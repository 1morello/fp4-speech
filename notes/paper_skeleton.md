# FP4 Inference for Speech Models: What Breaks and What Doesn't on NVIDIA Blackwell

---

## Abstract

NVIDIA Blackwell executes 4-bit floating-point (NVFP4) matrix multiplication in hardware, and the text-LLM stack adopted the format within months of launch. Whether speech models tolerate FP4 is, to our knowledge, unstudied. We evaluate hardware-native FP4 inference on three architectures: an autoregressive codec-LM TTS (Orpheus-3B), a flow-matching diffusion-transformer TTS (F5-TTS, 336M), and an encoder-decoder ASR (Whisper large-v3). Weight-only FP4 (W4A16) was benign in every configuration we tested: quantizing all 515 linear layers of Whisper moves WER on 500 LibriSpeech utterances from 3.37% to 3.36%, inside our noise floor. Quantizing activations as well (W4A4) is a different story. Orpheus emits no audio on 3-4 of 10 test prompts depending on the run, while F5-TTS keeps producing speech that merely sounds muffled; both effects trace back to block-max scaling zeroing small values. Speedup follows the compute regime: memory-bound Orpheus decode doubles (152 to ~300 tok/s, and only with CUDA graphs), while compute-bound F5-TTS gains nothing from weight compression and more from halving its NFE steps. A power sweep shows the 250 W cap delivers the same throughput at ~10% better tokens-per-joule. We close with ten SM120 ecosystem pitfalls absent from official documentation.

---

## 1. Introduction

1.1. Blackwell ships native FP4 tensor cores (NVFP4: E2M1 values, FP8 block scales). On paper, dense FP4 is ~4x BF16 tensor-core throughput; the best we measure on large GEMMs is 3.4x. vLLM, TensorRT-LLM and SGLang all shipped NVFP4 paths for text LLMs within months of the hardware.

1.2. Speech is missing from this picture, and the transfer is not obvious:
   - the output is a continuous signal, so quantization error becomes audible distortion, not a wrong token
   - the architecture zoo is wider than in text: AR codec LMs, flow-matching DiTs, encoder-decoder ASR
   - workloads sit on both sides of the roofline: AR decode at batch 1 is memory-bound, a 32-step DiT forward is compute-bound
   - error propagation differs: an AR model drags its mistakes through hundreds of decode steps, a diffusion model re-noises every step

1.3. Contributions:
   (a) an FP4 weight and activation sensitivity map across three speech architectures — to our knowledge the first; the sensitivity turns out to live in activations, not weights
   (b) evidence that W4A16 is safe in all three architectures tested, while W4A4 fails in architecture-specific ways
   (c) a roofline-grounded account of when FP4 actually pays off (memory-bound AR decode: 2x) and when it does not (compute-bound DiT: 1.0x)
   (d) a working W4A4 execution path for an audio DiT, with Python-side activation quantization identified as the current bottleneck
   (e) an energy analysis under power caps and ten documented SM120 pitfalls

[TODO before submission: proper prior-art pass — TensorRT-LLM / ModelOpt speech recipes, whisper.cpp INT4 users, any deployed FP4 TTS — so that every "first" above survives review.]

---

## 2. Background and Related Work

2.1. NVFP4 format
   - E2M1: 16 representable values in [-6, +6]
   - block scaling: 16 consecutive values share one FP8 (E4M3) scale
   - scale = max(|block|) / 6, cast to FP8
   - hardware path: cuBLAS NVFP4 GEMM on SM100/SM120 tensor cores

2.2. Quantization for text LLMs
   - GPTQ, AWQ, SqueezeLLM: weight-only INT4, pre-Blackwell
   - NVFP4 W4A4 paths in vLLM / TensorRT-LLM, decoder-only text models
   - nothing published on speech models under FP4 [re-check at submission time]

2.3. Models under study
   - Orpheus-3B: Llama-based AR model emitting SNAC codec tokens; sequential decode, memory-bound at batch 1; representative of the speech-codec-LM paradigm
   - F5-TTS (336M): flow-matching DiT over mel-spectrograms, 32 NFE steps by default; compute-bound through repeated dense forwards
   - Whisper large-v3 (1.5B): encoder-decoder ASR; encoder chews the full mel (M~1500 frames, compute-bound), decoder is AR text generation (memory-bound)

2.4. Roofline prediction
   - FP4 halves weight traffic and doubles tensor-core FLOPS (the latter only if both operands are FP4)
   - so: AR decode should gain from W4A16 already; a DiT should gain only from W4A4; Whisper sits in between
   - the experiments below test exactly this prediction

---

## 3. Experimental Setup

3.1. Hardware
   - GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB GDDR7), power cap range 250-325 W
   - CPU: AMD Threadripper TRX50, 256 GB RAM
   - power caps set via nvidia-smi -pl (requires admin; the sweep script verifies the cap actually applied)

3.2. Software
   - PyTorch 2.11.0+cu130, Triton 3.6.0, vLLM 0.22.1, llmcompressor 0.11.0
   - CUDA 13.0, driver 580.159.03, FlashInfer 0.6.11 (JIT CUTLASS NVFP4 kernels)

3.3. Quantization paths
   - fakequant: our NVFP4 quant-dequant (block 16, FP8 scales) applied per layer — measures sensitivity without deployment machinery
   - llmcompressor oneshot: calibrated NVFP4 / W4A16 checkpoints for Orpheus (calibration: 10 project phrases + Harvard sentences lists 1-4)
   - FP4Linear: drop-in nn.Module replacing Linear with torch._scaled_mm, our W4A4 path for F5-TTS

3.4. Metrics
   - WER via Whisper large-v3 transcription (lowercased, punctuation stripped). For TTS this is an intelligibility proxy, not a quality score
   - RTF: wall time / audio duration; tok/s for Orpheus
   - tok/J and audio-sec/J under power caps
   - listening so far is informal; a proper CMOS test is planned (see Remaining work)

3.5. Data
   - Whisper: 500 utterances of LibriSpeech test-clean
   - TTS: fixed sets of 5-10 English phrases, seed pinned where the stack allows
   - note: all TTS quality numbers below are sanity-scale (n = 5-10). Treat them as existence proofs; the 500-phrase eval is pending

---

## 4. Results

### 4.1. Does the FP4 path light up at all? (GEMM microbenchmarks)

Table 1: BF16 vs NVFP4 TFLOPS across speech-relevant shapes.

- 3.4x at 4096^3 — tensor cores confirmed active
- 2.8x at the F5-TTS DiT FFN shape (2048 x 5120 x 1280)
- 1.0-1.4x at decode-like shapes (M <= 512, K = 1024): too small to feed the FP4 pipes
- cuBLAS rejects NVFP4 GEMM for M < 128 outright (undocumented); our W4A4 path pads M up to a multiple of 128

So FP4 GEMM gains are shape-dependent: large-M forwards get close to the theoretical ratio, batch-1 decode gets nearly nothing at the kernel level — its win has to come from memory traffic instead.

### 4.2. Weights: nothing breaks

Quantizing all 515 linear layers of Whisper to FP4 changes WER on 500 LibriSpeech utterances from 3.37% to 3.36%. That is: nothing happens. Partial configurations (encoder only, decoder only, attention only, MLP only) land between 3.21% and 3.40% — a ±0.16 pp spread that brackets the baseline from both sides, which we read as the noise floor of a 500-sample WER estimate rather than as real differences.

Table 2: Whisper component-wise W4A16 sensitivity (500 samples).
Table 3: F5-TTS component x NFE grid.

F5-TTS behaves the same at sanity scale: WER flat at 4.7% in every cell of the component x NFE grid (167 layers, NFE 32 and 16; n = 5 per cell). Orpheus under W4A16 (Marlin kernels): 10/10 phrases, 193 tok/s, 1.27x over BF16.

Weight-only FP4 was safe in all three architectures we tested. This mirrors the text-LLM experience but had not been verified for speech.

### 4.3. Activations: two different failure modes

Orpheus (AR codec LM). Failure rates by run: 6/10 prompts survive with enforce_eager (4 failures, each emitting 0-2 tokens before a premature stop), 7/10 with CUDA graphs, 6/10 after re-calibrating with 50 prompts instead of 10. So the failure rate is 30-40% and insensitive to calibration budget — failed prompts die immediately rather than degrade. Layer-group fakequant localizes nothing: every 4-layer group individually preserves the top-1 next token (logit MSE 0.04-1.23 per group), yet quantizing all 28 layers flips it. The damage is cumulative across depth; there is no single culprit layer.

F5-TTS (flow-matching DiT). All 5/5 prompts produce audio, but it sounds low-pass filtered / muffled [TODO: spectrograms before claiming any cutoff frequency]. WER stays at 4.7% for both NFE 32 and NFE 16 — per-step quantization noise does not visibly accumulate across denoising steps, in sharp contrast to the AR case.

Why both, mechanically: the block scale is max(|block|)/6, so one outlier suppresses its 15 neighbors toward zero (in our synthetic test, a single 1000.0 in a block of 16 zeroes the other fifteen 0.01s). In an AR decoder this small per-layer error compounds over 28 layers times hundreds of decode steps until the sampled token shifts — in the worst case to <stop>. In a DiT, every denoising step restarts from a re-noised state, so the error stays bounded within the step; what is lost is the fine spectral detail inside each step, and that never returns.

### 4.4. Throughput and memory

Table 4: Orpheus-3B across regimes (like-for-like day-4 comparison; sampling temperature 0.4).

| Regime               | tok/s | RTF  | VRAM    | vs BF16 |
|----------------------|-------|------|---------|---------|
| BF16                 | 152   | 0.57 | 6.2 GB  | 1.0x    |
| W4A16 Marlin         | 193   | 0.46 | ~2.5 GB | 1.27x   |
| NVFP4, enforce_eager | 82    | 1.07 | 2.44 GB | 0.54x   |
| NVFP4 + CUDA graphs  | 300   | 0.29 | 2.44 GB | 2.0x    |

(The energy sweep, run later with different sampling settings, measured 316-318 tok/s on the same NVFP4 checkpoint; the table keeps the same-day, same-settings numbers.)

CUDA graphs are not an optimization here, they are a requirement: without them NVFP4 loses to BF16 (82 vs 152 tok/s) because kernel-launch overhead, not math, dominates. With graphs, the memory-bandwidth advantage finally shows: 2x throughput at 2.5x less VRAM.

Table 5: F5-TTS regimes (RTF = mean over 4 phrases after warmup).

| Regime | RTF  | Sounds like | Note                          |
|--------|------|-------------|-------------------------------|
| BF16   | 0.09 | clean       | baseline, 32 NFE              |
| W4A16  | 0.10 | clean       | compute-bound: nothing to gain|
| W4A4   | 0.65 | muffled     | Python act-quant dominates    |

F5-TTS is compute-bound, so W4A16 buys memory, not speed — as the roofline predicted. Our W4A4 path is 7x slower than BF16, but the overhead is Python-level activation quantization, not the FP4 GEMM itself; a fused Triton quantize kernel is the obvious next step [in progress].

### 4.5. NFE x quantization: the knobs compose

Table 6: F5-TTS under joint NFE reduction and FP4 weights.

Halving NFE from 32 to 16 halves RTF (0.09 to 0.055) with no WER change, and this still holds with all 167 layers in FP4. Weight quantization cuts the model in memory, NFE reduction cuts iterations, and at least at sanity scale (n = 5) they do not interact. Whether this orthogonality survives 500 phrases and UTMOS is exactly what the pending large-scale eval will tell.

### 4.6. Energy

Table 7: Orpheus NVFP4 under power caps.

| Cap   | tok/s | Avg draw | tok/J | vs 325 W |
|-------|-------|----------|-------|----------|
| 250 W | 316.1 | 231.3 W  | 1.367 | +10.4%   |
| 275 W | 316.8 | 240.4 W  | 1.318 | +6.5%    |
| 300 W | 317.7 | 245.8 W  | 1.292 | +4.4%    |
| 325 W | 318.1 | 257.0 W  | 1.238 | baseline |

Throughput varies by less than 1% across the whole cap range, and the average draw never reaches the cap. The workload is memory-bound, not power-limited: raising the cap buys heat, not tokens. The minimum 250 W cap is Pareto-optimal here — identical throughput, ~10% better tok/J.

---

## 5. Discussion

5.1. Why weights survive. Trained weight matrices are statistically tame — near-Gaussian, few extreme outliers — which is the same reason weight-only INT4 worked for text LLMs. FP8 block scales track such distributions well, and this has nothing to do with the training objective, which is presumably why all three architectures agree.

5.2. Why activations don't. Activation outliers are input-dependent and land wherever they want; one outlier per block of 16 wipes the small values around it. In speech those small values are not noise — they carry high-frequency harmonics and prosodic detail. The same mechanism costs an AR model its next token and a DiT its treble.

5.3. Sequential vs parallel error paths. An AR decoder composes its own errors: whatever FP4 does at step t is baked into the context for step t+1, across hundreds of steps. A DiT gets a fresh re-noised input every step, which acts as built-in error containment. This single architectural difference explains why identical per-layer noise produces total failure in one model and a muffled-but-intact signal in the other.

5.4. What we would actually deploy today
   - AR-TTS (Orpheus-like): W4A16 via Marlin. 1.27x speed, 2.5x memory, 10/10 reliability. NVFP4 W4A4 offers 2x but drops 30-40% of prompts — not shippable without mixed-precision fallbacks or better activation handling
   - DiT-TTS (F5-like): W4A16 for memory plus NFE 16 for speed; W4A4 pointless until the quantize step is fused into a kernel
   - ASR (Whisper-like): W4A16, no measurable cost. The hardware NVFP4 path is currently blocked upstream: llmcompressor has no encoder-decoder support

5.5. Limitations
   - TTS quality sets are tiny (5-10 phrases, one voice, one seed); WER is an intelligibility proxy; listening was informal. CMOS + UTMOS at 500 phrases are the fix, both planned
   - one GPU model tested (workstation Max-Q); datacenter Blackwell (B200/GB200) has a different power/bandwidth balance
   - our W4A4 timing is dominated by unfused Python quantization; conclusions about W4A4 *speed* (not quality) will change with a fused kernel
   - CosyVoice2 (hybrid LLM + flow) not yet covered

---

## 6. SM120 Ecosystem Notes

Ten pitfalls we hit on SM120 that are documented nowhere official; kept here both as a practical guide and as candidate upstream fixes.

1. System CUDA toolkits older than 12.8 do not know compute_120; every from-source build fails until the toolkit is upgraded
2. FlashInfer's check_cuda_arch() compares compute capability as strings, so "12" < "75" and sm_120 is rejected as "requires sm75 or higher" — one-line fix, PR-able
3. cuBLAS NVFP4 GEMM requires M >= 128 (and padding to multiples of 128); batch-1 decode cannot use it directly
4. vLLM V1 EngineDeadError: the engine child dies silently and the parent shows no root cause; workaround: VLLM_USE_FLASHINFER_SAMPLER=0, enforce_eager for debugging, direct LLM API
5. A crashed EngineCore leaves a zombie process holding ~90 GB of GPU memory; manual pkill required
6. "SM 12.x requires CUDA >= 12.9" warnings at startup are non-blocking noise
7. VLLM_USE_V1 and VLLM_MAX_MODEL_LEN env vars are silently unrecognized in vLLM 0.22.1
8. orpheus_tts API: module is orpheus_tts (not orpheus_speech), generate_speech yields raw PCM int16 chunks
9. FlashInfer's bundled CCCL headers are incompatible with a pip-installed nvcc; a system CUDA 13 install resolves JIT compilation
10. enforce_eager masks real NVFP4 performance (82 vs 300 tok/s); any NVFP4 benchmark without CUDA graphs undersells the format by ~4x

---

## 7. Conclusion

FP4 weights are a free lunch for speech: across an AR codec LM, a diffusion transformer and an encoder-decoder ASR we could not measure a quality cost, and the memory savings are immediate. FP4 activations are where the problem lives, and the failure mode is set by the architecture — an AR model collapses, a DiT loses detail. The speedup story is equally conditional: 2x for memory-bound decode (with CUDA graphs mandatory), nothing for compute-bound DiTs until activation quantization moves into a fused kernel. And on this hardware the cheapest optimization of all is turning the power cap down.

---

## Figures (to produce)
1. Block-scaling outlier diagram — one outlier zeroing its block, tied to the synthetic 1000-in-a-block test
2. Architecture comparison — Orpheus / F5-TTS / Whisper data flow side by side
3. Sensitivity heatmap — model x component x precision
4. Energy curve — tok/J vs power cap
5. Spectrograms — F5-TTS BF16 vs W4A4 (this also settles the "muffled" claim quantitatively)
6. Throughput bars — all models x all regimes

## Tables (data ready)
1. GEMM microbenchmark
2. Whisper sensitivity grid (500 samples)
3. F5-TTS NFE x component grid
4. Orpheus regime comparison
5. Energy sweep
6. Cross-model summary
7. SM120 notes

## Remaining work
- 500-phrase TTS eval: WER + UTMOS
- CMOS listening test (target: 15 listeners, 50 pairs)
- fused Triton activation-quantize kernel (Triton exercises started)
- prior-art pass over ModelOpt / TensorRT-LLM / whisper.cpp quantization for speech
- CosyVoice2 if time permits
- figures + writing
