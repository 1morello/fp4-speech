- уст. vllm: резолвер откатился до vllm==0.1.2 (sdist, 2023) из-за конфликта torch-пинов
  → сборка упала на системном nvcc (CUDA 12.0 — не знает compute_120!)
  → фикс: --no-build + 'vllm>=0.10' + --torch-backend=auto
  → факт для статьи/админа: системный CUDA toolkit на сервере древнее Blackwell

- uv pip check: vllm хочет compressed-tensors==0.15.0.1, стоит 0.16.0 (притащил llmcompressor).
  Известное перетягивание. Решение отложено до дня 3; план Б — раздельные venv quant/serve.


