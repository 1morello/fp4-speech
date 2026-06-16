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

git add -A && git commit -m "docs(log): days 0-3 findings, sm120 ecosystem issues"

