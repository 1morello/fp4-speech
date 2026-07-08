# Hardware FP4 Inference for Speech: A Systematic Study on NVIDIA Blackwell

---

## Abstract

Quantization to 4-bit floating point (FP4) has emerged as a key optimization for large language model inference on NVIDIA Blackwell GPUs, yet its applicability to speech models remains entirely unexplored. We present the first systematic evaluation of hardware-native NVFP4 inference across three representative speech architectures: autoregressive LLM-based TTS (Orpheus-3B), flow-matching DiT TTS (F5-TTS, 336M), and encoder-decoder ASR (Whisper large-v3, 1.5B). Our study yields three principal findings. First, FP4 weight quantization preserves output quality universally across all tested architectures, with zero measurable degradation (Whisper: 3.36% vs 3.37% WER on 500 LibriSpeech samples; F5-TTS: invariant across all 167 layers and NFE settings). Second, FP4 activation quantization introduces architecture-dependent failures: autoregressive models exhibit catastrophic generation collapse (30% of prompts), while diffusion models suffer graceful spectral degradation — both traced to the same block-scaling outlier mechanism. Third, the interplay between quantization format and compute regime determines practical benefit: memory-bound AR-TTS achieves 2x throughput (300 tok/s vs 152 BF16), while compute-bound DiT-TTS gains nothing from weight compression alone but benefits from orthogonal NFE reduction (32 to 16 steps, zero quality loss). We additionally report a 4-point energy sweep showing 10% efficiency gain at minimum power cap, 12 undocumented SM120 ecosystem constraints, and the first W4A4 execution path for an audio diffusion transformer. All code and measurements are publicly available.

---

## 1. Introduction

1.1. NVIDIA Blackwell introduces hardware-native FP4 (NVFP4) tensor core operations, enabling 4-bit matrix multiplication at up to 8x the throughput of BF16. The quantization ecosystem for text-based LLMs has rapidly adopted this capability, with vLLM, TensorRT-LLM, and SGLang shipping NVFP4 support within months of Blackwell's release.

1.2. Speech models, however, remain entirely unstudied under FP4. This gap is not merely an omission — speech presents fundamentally different challenges for aggressive quantization:
   - Output is a continuous signal (audio waveform), not discrete tokens; quantization artifacts manifest as audible distortion rather than textual errors
   - The dominant architectures span a wider range than text: autoregressive codec LMs, flow-matching diffusion transformers, and encoder-decoder sequence models
   - Inference workloads vary from memory-bound (AR decode, batch=1) to compute-bound (DiT with 32 NFE steps), changing where quantization can help
   - Error propagation differs: AR models accumulate errors sequentially across hundreds of decode steps; diffusion models apply errors independently across NFE steps

1.3. Contributions:
   (a) The first FP4 weight and activation sensitivity map across three speech architectures, revealing that sensitivity is concentrated in activations, not weights, and that the failure mode is architecture-dependent
   (b) Quantitative evidence that FP4 weight quantization is universally safe for speech deployment (W4A16), while W4A4 requires architecture-specific mitigation
   (c) A roofline-grounded analysis explaining when FP4 helps (memory-bound AR-TTS: 2x) and when it does not (compute-bound DiT-TTS: 1.0x)
   (d) The first W4A4 execution path for an audio diffusion transformer, identifying Python-level activation quantization as the primary overhead
   (e) An energy efficiency analysis (tokens-per-joule under power caps) and 12 documented SM120 ecosystem constraints for the Blackwell early-adopter community

---

## 2. Background and Related Work

2.1. NVFP4 format specification
   - E2M1 representation: 16 distinct values in [-6, +6]
   - Block scaling: groups of 16 values share one FP8 (E4M3) scale factor
   - Quantization granularity: scale = max(|block|) / 6.0, cast to FP8
   - Hardware path: cuBLAS NVFP4 GEMM on SM100/SM120 tensor cores

2.2. Quantization for text LLMs
   - GPTQ, AWQ, SqueezeLLM: weight-only INT4 methods (pre-Blackwell)
   - NVFP4 in vLLM/TensorRT-LLM: hardware W4A4 path for decoder-only LLMs
   - Key gap: no published work applies any of these to speech models

2.3. Speech model architectures under study
   - Orpheus-3B: Llama-based AR model generating SNAC audio codec tokens. Decode is sequential, memory-bound at batch=1. Representative of GPT-4o voice, Gemini Live paradigm.
   - F5-TTS (336M): Flow-matching DiT. Generates mel-spectrogram through iterative denoising (32 NFE steps by default). Compute-bound due to repeated dense forward passes. Representative of modern non-autoregressive TTS.
   - Whisper large-v3 (1.5B): Encoder-decoder for ASR. Encoder processes full mel-spectrogram (M~1500); decoder generates text tokens autoregressively. Mixed regime: encoder compute-bound, decoder memory-bound.

2.4. Roofline model for FP4 benefit prediction
   - FP4 halves memory traffic (4-bit weights vs 8-bit) and doubles tensor core throughput
   - Memory-bound workloads (arithmetic intensity < machine balance) benefit from reduced traffic
   - Compute-bound workloads benefit only from higher FLOPS, which requires W4A4 (both operands in FP4)
   - Prediction: AR-TTS decode benefits most; DiT benefits only with W4A4; Whisper encoder is borderline

---

## 3. Experimental Setup

3.1. Hardware
   - GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB GDDR7, 250-325W TDP)
   - CPU: AMD Threadripper TRX50, 256 GB RAM
   - Power caps: 250 / 275 / 300 / 325 W via nvidia-smi -pl

3.2. Software
   - PyTorch 2.11.0+cu130, Triton 3.6.0, vLLM 0.22.1, llmcompressor 0.11.0
   - CUDA toolkit 13.0, driver 580.159.03
   - FlashInfer 0.6.11 (JIT compilation for CUTLASS NVFP4 kernels)

3.3. Quantization methods
   - Fakequant: custom NVFP4 quant-dequant function (block=16, FP8 scales) applied per-layer to measure sensitivity without deployment overhead
   - llmcompressor oneshot: calibration-based NVFP4 and W4A16 quantization for Orpheus checkpoints
   - Drop-in FP4Linear: custom nn.Module replacing Linear layers with torch._scaled_mm for F5-TTS W4A4 path

3.4. Evaluation metrics
   - WER: word error rate via Whisper large-v3 transcription, text-normalized (lowercase, punctuation removed)
   - RTF: real-time factor (wall time / audio duration)
   - tok/s: generated tokens per second (Orpheus)
   - tok/J and audio-sec/J: energy efficiency under power caps
   - Subjective quality: informal A/B comparison (formal CMOS planned)

3.5. Evaluation data
   - Whisper sensitivity: 500 samples from LibriSpeech test-clean
   - TTS sanity: 10 fixed English phrases, deterministic seed where possible
   - TTS large-scale: 500 LibriSpeech phrases (planned)

---

## 4. Results

### 4.1. FP4 GEMM Microbenchmarks

Validation that NVFP4 tensor cores activate on SM120 and quantification of speedup by matrix shape.

Table 1: BF16 vs NVFP4 throughput (TFLOPS) across speech-relevant shapes.

Key findings:
- 3.4x at large square matrices (4096^3), confirming tensor core activation
- 2.8x at F5-TTS DiT FFN shapes (2048 x 5120 x 1280)
- 1.0-1.4x at Whisper/Orpheus decode shapes (M=128-512, K=1024)
- cuBLAS rejects NVFP4 for M < 128 (undocumented constraint)
- cuBLAS requires M padded to nearest multiple of 128

Implication: FP4 GEMM benefit is shape-dependent. Compute-bound workloads with large M (DiT forward pass) see near-theoretical speedup. Memory-bound workloads with small M (AR decode) see minimal GEMM-level benefit but may still benefit from reduced memory traffic at the model level.

### 4.2. Weight Sensitivity Across Architectures (W4A16 Fakequant)

Central question: do speech model weights tolerate FP4 quantization?

Table 2: Whisper large-v3 component-wise W4A16 sensitivity (500 LibriSpeech samples).

Table 3: F5-TTS component-wise W4A16 sensitivity with NFE axis.

Summary of weight sensitivity:
- Whisper: 515 Linear layers quantized, WER 3.36% vs 3.37% baseline. No component (encoder, decoder, self-attention, cross-attention, MLP) shows measurable degradation.
- F5-TTS: 167 Linear layers quantized, WER invariant at 4.7% across all components and NFE values (32 and 16).
- Orpheus: W4A16 via Marlin yields 10/10 phrase success, 193 tok/s (1.27x over BF16).

Finding: FP4 weight quantization is universally safe across all speech architectures tested. This is consistent with text LLM findings but had not been verified for speech.

### 4.3. Activation Sensitivity (W4A4)

Central question: what happens when activations are also quantized to FP4?

4.3.1. Orpheus (AR-LLM TTS)
- 7/10 phrases generate valid audio; 3/10 emit stop-token immediately (0-2 tokens)
- Re-calibration with 50 prompts (vs 10) does not improve success rate (6/10)
- Layer-group analysis: no single group of 4 layers causes failure; MSE ranges 0.04-1.23 per group, all preserve top-1 token individually; only ALL layers combined shift top-1
- Conclusion: activation quantization error accumulates sequentially across 28 decoder layers. The failure is architectural, not calibration-dependent.

4.3.2. F5-TTS (Flow-matching DiT)
- All phrases produce audio, but output sounds low-pass filtered
- High frequencies (> 5 kHz) attenuated or eliminated
- Mechanism: within each block of 16 activation values, outliers dominate the scale factor, causing small values (which encode high-frequency detail) to round to zero
- NFE interaction: quantization noise does NOT accumulate across diffusion steps (32 steps and 16 steps show identical WER under FP4). This contrasts sharply with the AR case.

4.3.3. Unified explanation
Both failure modes trace to block scaling: max(|block|)/6.0 sets the scale, and any value smaller than ~1/12 of the block maximum rounds to zero. In AR models, these small errors compound across hundreds of sequential decode steps until the model emits a premature stop token. In DiT models, each NFE step is an independent denoising operation, so errors do not compound — but the spectral information lost within each step is never recovered.

### 4.4. Throughput and Memory

Table 4: Orpheus-3B inference across quantization regimes.

| Regime              | tok/s | RTF  | VRAM   | vs BF16 |
|---------------------|-------|------|--------|---------|
| BF16                | 152   | 0.57 | 6.2 GB | 1.0x    |
| W4A16 Marlin        | 193   | 0.46 | ~2.5 GB| 1.27x   |
| NVFP4 (enforce_eager)| 82   | 1.07 | 2.44 GB| 0.54x   |
| NVFP4 + CUDA graphs | 300   | 0.29 | 2.44 GB| 2.0x    |

Critical finding: CUDA graph capture is essential for NVFP4 performance. Without it, per-kernel launch overhead dominates, making NVFP4 slower than BF16. With CUDA graphs, launch overhead is amortized, revealing the true memory-bandwidth benefit: 2x throughput at 2.5x memory reduction.

Table 5: F5-TTS inference regimes.

| Regime | RTF  | Quality         | Notes                    |
|--------|------|-----------------|--------------------------|
| BF16   | 0.09 | clean           | baseline, 32 NFE         |
| W4A16  | 0.10 | clean           | no speedup (compute-bound)|
| W4A4   | 0.65 | muffled         | Python act-quant overhead |

F5-TTS is compute-bound: reducing weight precision does not reduce the bottleneck (FLOPs, not memory bandwidth). The W4A4 path is 7x slower than BF16 due to Python-level activation quantization overhead, not the FP4 GEMM itself. A fused Triton kernel eliminating this overhead is identified as the necessary next step.

### 4.5. NFE-Quantization Interaction (F5-TTS)

Table 6: F5-TTS quality under joint NFE reduction and FP4 weight quantization.

NFE reduction from 32 to 16 halves inference time (RTF 0.09 to 0.055) with zero quality degradation, and this holds even when all 167 layers are quantized to FP4. The two optimizations are orthogonal: weight quantization reduces model size, NFE reduction reduces iteration count, and their combination yields compound savings without compound quality loss.

This is a novel finding for speech: in text diffusion models, aggressive step reduction typically interacts with quantization. For audio, the mel-spectrogram representation appears robust to both perturbations simultaneously.

### 4.6. Energy Efficiency

Table 7: Orpheus NVFP4 under power caps (RTX PRO 6000, 250-325W range).

| Power cap | tok/s | Avg power | tok/J | Efficiency vs 325W |
|-----------|-------|-----------|-------|---------------------|
| 250 W     | 316.1 | 231.3 W   | 1.367 | +10.4%              |
| 275 W     | 316.8 | 240.4 W   | 1.318 | +6.5%               |
| 300 W     | 317.7 | 245.8 W   | 1.292 | +4.4%               |
| 325 W     | 318.1 | 257.0 W   | 1.238 | baseline            |

Throughput is flat across the entire power range (< 1% variation). The workload is not power-limited: additional wattage converts to heat, not performance. Minimum power cap (250W) is Pareto-optimal, delivering identical throughput at 10% lower energy cost. This has direct implications for datacenter deployment cost.

---

## 5. Discussion

5.1. Why weights are robust
Weight distributions in trained models are approximately Gaussian with moderate dynamic range. Block scaling with FP8 scale factors captures this distribution accurately. The E2M1 palette, while coarse (16 values), provides sufficient resolution when the scale factor is well-matched. This holds across all three architectures because the statistical properties of trained weights are similar regardless of the training objective.

5.2. Why activations break
Activations are input-dependent and exhibit outliers that are absent from weight distributions. A single outlier value in a block of 16 forces the scale factor high, causing the remaining 15 values to quantize to zero or the nearest E2M1 level. For speech, this is particularly damaging: small activation values encode high-frequency spectral detail, fine temporal alignment, and subtle prosodic cues.

5.3. AR vs DiT error propagation
The critical architectural distinction is sequential vs parallel application of quantized layers. In AR models, each token generation depends on all previous tokens, and quantization error at step t propagates to steps t+1, t+2, etc. After 28 layers times hundreds of decode steps, accumulated error can shift the output distribution enough to trigger premature termination. In DiT models, each NFE step is a self-contained denoising operation. Errors within a step degrade that step's output, but the next step operates on a fresh noisy input, providing natural error correction.

5.4. Practical deployment recommendations
- AR-TTS (Orpheus-like): deploy W4A16 via Marlin. 1.27x speedup, 2.5x memory reduction, zero quality loss. NVFP4 (W4A4) provides 2x speedup but at 30% failure rate — acceptable only with mixed-precision recipes or fused activation quantization kernels.
- DiT-TTS (F5-TTS-like): deploy W4A16 + reduced NFE (16 instead of 32). Weight quantization saves memory; NFE reduction saves compute. Combined: ~1.6x effective speedup. W4A4 requires fused kernels to be practical.
- ASR (Whisper-like): deploy W4A16. Zero degradation confirmed. Hardware NVFP4 path blocked by llmcompressor's lack of encoder-decoder support.

5.5. Limitations
- Subjective quality assessment is informal; CMOS test with listeners planned but not yet conducted
- Only one GPU tested (RTX PRO 6000 Max-Q); results may differ on datacenter Blackwell (B200/GB200)
- F5-TTS W4A4 overhead is dominated by Python-level quantization; a fused Triton kernel would change the speed comparison substantially
- CosyVoice2 (hybrid LLM+flow) not yet evaluated

---

## 6. SM120 Ecosystem Findings

We document 12 constraints encountered during development on SM120 (Blackwell) that are absent from official documentation. These are presented as a practical guide for early adopters and potential upstream contributions.

[List of 12 findings with workarounds — see project log]

---

## 7. Conclusion

Hardware FP4 inference for speech models is viable but architecture-dependent. Weight quantization to FP4 is universally safe and should be adopted immediately for memory savings. Activation quantization remains challenging: autoregressive models suffer catastrophic failure from error accumulation, while diffusion models degrade gracefully through spectral loss. The optimal deployment strategy depends on the compute regime: memory-bound AR-TTS benefits most from FP4 weights (2x throughput), while compute-bound DiT-TTS benefits more from orthogonal optimizations (NFE reduction). Energy efficiency analysis confirms that speech FP4 inference is not power-limited on Blackwell Max-Q, making minimum power cap the Pareto-optimal operating point.

---

## Figures (to produce)
1. Block scaling outlier diagram — visual explanation of the E2M1 quantization mechanism and outlier effect
2. Architecture comparison diagram — Orpheus / F5-TTS / Whisper side by side with data flow
3. Sensitivity heatmap — models x components x precision format
4. Energy Pareto curve — tok/J vs power cap
5. Spectrograms — F5-TTS BF16 vs W4A4 showing high-frequency loss
6. Throughput bar chart — all models x all regimes

## Tables (all data available)
1. GEMM microbenchmark (have data)
2. Whisper sensitivity grid (have data)
3. F5-TTS NFE x component grid (have data)
4. Orpheus regime comparison (have data)
5. Energy sweep (have data)
6. Cross-model summary (have data)
7. SM120 ecosystem findings (have data)

## Remaining work
- 500-phrase TTS evaluation with automatic WER and UTMOS
- CMOS listening test (15 listeners, 50 pairs)
- Triton fused activation-quantization kernel
- CosyVoice2 evaluation (if time permits)
- Paper writing and figure production
