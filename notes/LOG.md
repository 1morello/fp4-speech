- уст. vllm: резолвер откатился до vllm==0.1.2 (sdist, 2023) из-за конфликта torch-пинов
  → сборка упала на системном nvcc (CUDA 12.0 — не знает compute_120!)
  → фикс: --no-build + 'vllm>=0.10' + --torch-backend=auto
  → факт для статьи/админа: системный CUDA toolkit на сервере древнее Blackwell

- uv pip check: vllm хочет compressed-tensors==0.15.0.1, стоит 0.16.0 (притащил llmcompressor).
  Известное перетягивание. Решение отложено до дня 3; план Б — раздельные venv quant/serve.

# FP4-Speech Project Log

## День 0 — пт 12.06.2026 — аудит железа
- GPU: 2× RTX PRO 6000 Blackwell Max-Q Workstation Edition
- compute_cap: 12.0 (sm_120), драйвер 580.159.03, CUDA 13.0
- Power limit: default 300 Вт, min 250, max 325 → свип 250/275/300/325
- sudo: НЕТ → нужен админ для -pl, -lgc, NCU
- Соседи: Мамырбек (~2.4 ГБ на обеих картах, util ~0%)
- idle-мощности на глаз: GPU0 ~13.7 Вт, GPU1 ~11.4 Вт

## День 1 — пт 12.06.2026 — окружение
- env: torch 2.11.0+cu130, triton 3.6.0 ✓, vllm 0.22.1, llmcompressor 0.11.0
- грабли: vllm-резолвер уехал в 0.1.2 (фикс: --no-build + 'vllm>=0.10')
- системный nvcc = CUDA 12.0, не знает compute_120 → сборки из исходников невозможны
- datasets>=4 требует torchcodec для Audio-колонок
- compressed-tensors конфликт: vllm хочет 0.15.0.1, llmcompressor хочет 0.16.0
- Whisper large-v3 BF16 транскрибирует ✓ (Mr. Quilter на месте)
- договор: работаю на GPU 1, CUDA_VISIBLE_DEVICES в .bashrc

## День 2 — пн 15.06.2026 — fakequant + GEMM
- написал nvfp4_fakequant с block scaling (палитра E2M1, блоки по 16, FP8-скейлы)
- выброс-тест: 1000 в блоке из 16 → остальные 15 обнулились → ключевой инсайт проекта
- FP4 vs BF16 GEMM бенчмарки (torch._scaled_mm, cuBLAS):
  - 4096×4096: 1029 vs 302 TFLOPS → 3.4× ✓ FP4-ядра работают
  - 2048×5120×1280 (F5-TTS DiT FFN): 762 vs 268 → 2.8×
  - 1024×4096×1280 (DiT attention): 528 vs 228 → 2.3×
  - 128×1024×1024 (Whisper decode): 26 vs 19 → 1.4×
  - 256×1024×1024: 53 vs 44 → 1.2×
  - 512×1024×1024: 106 vs 101 → 1.0×
- cuBLAS NVFP4 не поддерживает M<128 → находка для статьи/vLLM
- gn-kernels не собирается (CUDA 12.0 mismatch) → написали свой бенч

## День 3 — вт 16.06.2026 — Orpheus BF16 бейзлайн
- orpheus-speech 0.1.0 установился; compressed-tensors сам откатился до 0.15.0.1
- llmcompressor теперь хочет 0.16.0 → конфликт перевернулся; отложено до квантизации
- Orpheus BF16 бейзлайн: 10 фраз, ~150-163 tok/s, RTF 0.53-0.60, 6.2 GiB VRAM
- качество: ~60-70% чисто, артефакты: растягивание гласных + повтор хвоста (поведение модели)
- днём: квантизация в NVFP4

## SM120 ecosystem findings (для статьи)
1. **nvcc CUDA 12.0 vs compute_120**: системный тулкит не знает sm_120 → все сборки из исходников падают. Нужен CUDA ≥12.8
2. **FlashInfer check_cuda_arch() баг**: сравнивает CC как строки → "12" < "75" → ошибка "requires sm75 or higher" на sm_120. Однострочный фикс → потенциальный PR
3. **cuBLAS NVFP4 minimum M=128**: матрицы с M<128 отвергаются → Whisper decode batch=1 невозможен через cuBLAS NVFP4. Не документировано
4. **vLLM V1 EngineDeadError**: дочерний процесс умирает молча, root cause не виден в parent. Workaround: VLLM_USE_FLASHINFER_SAMPLER=0 + enforce_eager + прямой LLM API
5. **vLLM zombie processes**: упавший EngineCore держит 90 ГБ GPU памяти. Надо pkill вручную
6. **"SM 12.x requires CUDA >= 12.9"**: warnings при запуске, не блокирующие
7. **VLLM_USE_V1, VLLM_MAX_MODEL_LEN**: env-переменные не распознаются в vllm 0.22.1
8. **orpheus_tts API**: модуль = orpheus_tts (не orpheus_speech), класс = OrpheusModel, generate_speech возвращает PCM int16 bytes chunks
EOF


- Субъективное наблюдение: W4A16 аудио звучит чище BF16 (~70% меньше артефактов).
  Гипотезы: (1) стохастичность генерации; (2) квантизационный шум как регуляризатор.
  TODO день 4: объективная проверка через WER (Whisper) + UTMOS.
  Если подтвердится — потенциальная находка для статьи (quant-as-regularizer для TTS).



## День 4 — чт 19.06.2026 — CUDA 13 + NVFP4 Orpheus ЗАГОВОРИЛ

### Главное событие
- Админ поставил CUDA 13.0 — стена FlashInfer JIT пала
- **Первый в мире аппаратный NVFP4-инференс TTS на Blackwell**

### Сводная таблица Orpheus-3B
| Метрика        | BF16   | W4A16 (Marlin) | NVFP4 eager | NVFP4 + CUDA graphs |
|----------------|--------|----------------|-------------|----------------------|
| tok/s          | ~152   | ~193           | ~82         | **~300**             |
| RTF            | 0.57   | 0.46           | 1.07        | **0.29**             |
| VRAM           | 6.2 GB | ~2.5 GB        | 2.44 GB     | **2.44 GB**          |
| Успешных фраз  | 10/10  | 10/10          | 6/10        | **7/10**             |
| vs BF16        | —      | 1.27×          | 0.5×        | **2.0×**             |

### Ключевые находки
1. CUDA graphs критичны для NVFP4: без них 82 tok/s, с ними 300 (3.7× разница)
   → NVFP4 имеет больший launch overhead, graphs его убирают
2. 3/10 пустых фраз в NVFP4: активации чувствительны к FP4, некоторые промпты
   попадают в "мёртвую зону" → sensitivity finding для статьи
3. Субъективная оценка качества (Арнур, проф. слух):
   S-tier: NVFP4, A-tier: BF16, B-tier: W4A16
   Гипотеза: квантизационный шум как регуляризатор подавляет артефакты
   (растягивание гласных, повтор хвоста, "Tara"-проговаривание)
   TODO: подтвердить CMOS-тестом на 500 фразах
4. Roofline подтверждён: Orpheus memory-bound → FP4 даёт 2× ускорение

### SM120 findings
9. FlashInfer CCCL headers incompatible с pip-installed nvcc — решено установкой системного CUDA 13
10. enforce_eager маскирует реальную производительность NVFP4 (82 vs 300 tok/s) —
    CUDA graphs обязательны для честного сравнения

## ср 08.07.2026 — ревизия кода перед перепрогоном
- whisper/sensitivity.py: refs/hyps наполнялись ВНЕ цикла — wer фактически считался
  по одному последнему сэмплу. дамми-числа дня 4 невалидны, в статью не брать.
  librispeech-500 гонялся уже исправленной версией, те цифры ок
- fakequant: clamp(1e-8) на скейл — нулевой блок давал деление на 0 -> nan
- калибровка v2: сгенерённый список фраз заменил на 10 своих + harvard sentences 1-4
  (фонетически сбалансированы) -> перегнать quantize_nvfp4_v2 и baseline_nvfp4_v2
- energy_sweep.py: -pl уходил через sudo без проверки; при отказе свип молча мерил
  все точки на одном лимите (потому и пришлось руками через админа).
  теперь sudo -n + сверка power.limit, непроставленная точка скипается
- f5tts_sensitivity: выкинул мёртвую переменную ref, на wer не влияло
- мусор из корня удалил (VLLM_MAX_MODEL_LEN — кривой редирект, *.log~unset)
