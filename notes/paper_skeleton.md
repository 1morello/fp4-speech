# Hardware FP4 Inference for Speech: A Systematic Study on NVIDIA Blackwell

---

## Abstract

NVIDIA Blackwell GPUs run 4-bit floating point (FP4) matrix multiplication in hardware. The text-LLM ecosystem adopted this within months; speech models have not been tested at all. We evaluate hardware NVFP4 inference across three speech architectures: autoregressive LLM-based TTS (Orpheus-3B), flow-matching DiT TTS (F5-TTS, 336M), and encoder-decoder ASR (Whisper large-v3, 1.5B). Four findings. First, FP4 weights are safe everywhere we looked: quantizing all 515 Whisper layers moves WER from 3.37% to 3.36% on 500 LibriSpeech samples, and F5-TTS is likewise unaffected across all 167 layers and NFE settings. Second, FP4 activations are where things break, and how they break depends on the architecture: the autoregressive model stops generating on a fraction of prompts, the diffusion model loses harmonic structure to quantization noise. Both trace to the same block-scaling mechanism, including FP8 scale-factor overflow once activation outliers pass |2688|. Third, whether FP4 buys speed depends on the compute regime: memory-bound AR-TTS reaches 2x throughput (300 vs 152 tok/s) once CUDA graphs are enabled, while compute-bound DiT-TTS gains nothing from weight compression and more from halving NFE (32 to 16, no quality loss). Fourth, on 500 phrases NVFP4 had the highest generation success rate of any regime, including BF16 (51.6% vs 50.2%), at the cost of moderately higher WER. Quantization noise appears to suppress the repetition loops that derail autoregressive decoding. We also contribute two Triton kernels covering gaps in the SM120 stack (including the M<128 regime that cuBLAS NVFP4 rejects outright), a 4-point energy sweep showing the minimum power cap is 10% more efficient at under 1% speed cost, and 12 undocumented SM120 constraints, two of which we filed upstream (FlashInfer #3945, vLLM #48491). Code and measurements are public.

---

## 1. Introduction

1.1. Blackwell introduces hardware-native FP4 (NVFP4) tensor cores, up to 8x BF16 matmul throughput. vLLM, TensorRT-LLM, and SGLang shipped NVFP4 support for text LLMs within months of release.

1.2. No published work applies any of this to speech. Speech is worth testing separately for four reasons:
   - The output is a continuous waveform, not tokens. Quantization error becomes audible distortion, not a wrong word.
   - The architecture zoo is wider than in text: autoregressive codec LMs, flow-matching diffusion transformers, encoder-decoder sequence models.
   - The workloads span both roofline regimes. AR decode at batch=1 is memory-bound; a DiT running 32 denoising passes is compute-bound. FP4 helps these differently.
   - Errors propagate differently. An AR model carries its mistakes forward through hundreds of decode steps; a diffusion model starts each denoising step fresh.

1.3. Contributions:
   (a) The first FP4 weight and activation sensitivity map across three speech architectures. Sensitivity sits in activations, not weights, and the failure mode is architecture-dependent.
   (b) Evidence that weight-only FP4 (W4A16) is safe to deploy for speech today, while W4A4 needs architecture-specific handling. A 500-phrase evaluation additionally surfaces a quantization-as-regularizer effect in AR-TTS.
   (c) A mechanistic account of the activation failure: block-scaling outlier suppression, compounded by FP8 scale overflow (mid-network activations reach |3616|; divided by 6, the scale exceeds the E4M3 maximum of 448).
   (d) A roofline-grounded account of when FP4 pays off (memory-bound AR-TTS, 2x) and when it does not (compute-bound DiT-TTS, 1.0x).
   (e) The first W4A4 execution path for an audio diffusion transformer, plus two Triton kernels built on tl.dot_scaled: a fused-quantization GEMM that removes the Python-level activation quantization overhead (3.5x over the naive path), and a small-M kernel for the M<128 regime cuBLAS refuses.
   (f) Tokens-per-joule measurements under power caps, and 12 documented SM120 constraints, two filed upstream (FlashInfer #3945, vLLM #48491).

---

## 2. Background and Related Work

2.1. NVFP4 format
   - E2M1 values: 16 representable levels in [-6, +6]
   - Block scaling: every 16 consecutive values share one FP8 (E4M3) scale
   - Scale computation: max(|block|) / 6.0, cast to FP8
   - Hardware path: cuBLAS NVFP4 GEMM on SM100/SM120 tensor cores
   - Two consequences follow directly from the format. A value below roughly 1/12 of its block maximum rounds to zero. A block maximum above 6 x 448 = 2688 produces a scale the FP8 type cannot represent.

2.2. Quantization for text LLMs
   - GPTQ, AWQ, SqueezeLLM: weight-only INT4, pre-Blackwell
   - NVFP4 in vLLM and TensorRT-LLM: the hardware W4A4 path for decoder-only LLMs
   - None of it has been applied to speech models in published work.

2.3. The three architectures
   - Orpheus-3B: a Llama backbone generating SNAC audio codec tokens. Sequential decode, memory-bound at batch=1. Same paradigm as GPT-4o voice and Gemini Live.
   - F5-TTS (336M): flow-matching DiT producing mel-spectrograms through iterative denoising, 32 NFE steps by default. Compute-bound: the same dense network runs 32 times.
   - Whisper large-v3 (1.5B): encoder-decoder ASR. The encoder sees the whole mel-spectrogram at once (M around 1500, compute-bound); the decoder emits text tokens one at a time (memory-bound).

2.4. Roofline predictions
   - FP4 halves weight memory traffic and doubles tensor core throughput.
   - Memory-bound workloads benefit from the traffic reduction alone (W4A16 is enough).
   - Compute-bound workloads only benefit if both operands are FP4 (W4A4).
   - Prediction: AR-TTS decode gains the most; the DiT gains only under W4A4; the Whisper encoder is borderline.

---

## 3. Experimental Setup

3.1. Hardware
   - GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB GDDR7, 250-325W TDP)
   - CPU: AMD Threadripper TRX50, 256 GB RAM
   - Power caps set via nvidia-smi -pl: 250 / 275 / 300 / 325 W

3.2. Software
   - PyTorch 2.11.0+cu130, Triton 3.6.0, vLLM 0.22.1, llmcompressor 0.11.0
   - CUDA 13.0, driver 580.159.03
   - FlashInfer 0.6.11 (JIT-compiled CUTLASS NVFP4 kernels)

3.3. Quantization methods
   - Fakequant: our own NVFP4 quant-dequant (block 16, FP8 scales), applied per layer to measure sensitivity without touching the serving stack
   - llmcompressor oneshot: calibrated NVFP4 and W4A16 checkpoints for Orpheus (10 and 50 calibration prompts)
   - FP4Linear: a drop-in nn.Module replacing Linear layers with torch._scaled_mm for the F5-TTS W4A4 path, with M padded to multiples of 128 per the cuBLAS constraint
   - Triton kernels via tl.dot_scaled (e2m1) for the fused GEMM and the small-M case

3.4. Metrics
   - WER via Whisper large-v3 transcription, text-normalized (lowercase, punctuation stripped)
   - Generation success rate: fraction of prompts producing non-empty audio
   - RTF (wall time / audio duration), tok/s, tok/J, audio-sec/J
   - Spectrograms for characterizing perceptual failure

3.5. Data
   - Whisper sensitivity: 500 samples from LibriSpeech test-clean
   - TTS sanity set: 10 fixed English phrases, fixed seed where the stack allows
   - TTS at scale: 500 LibriSpeech test-clean transcripts as input text, identical across all regimes

---

## 4. Results

### 4.1. FP4 GEMM Microbenchmarks

First we check that the FP4 tensor cores actually engage on SM120, and measure the speedup by shape.

Table 1: BF16 vs NVFP4 throughput (TFLOPS) on speech-relevant shapes.

| Shape (MxNxK)        | BF16 | NVFP4 | Ratio |
|----------------------|------|-------|-------|
| 4096x4096x4096       | 302  | 1029  | 3.4x  |
| 2048x5120x1280 (F5)  | 268  | 762   | 2.8x  |
| 1024x4096x1280       | 228  | 528   | 2.3x  |
| 512x1024x1024        | 101  | 106   | 1.0x  |
| 128x1024x1024        | 19   | 26    | 1.4x  |

What we found:
- 3.4x on the large square case confirms the tensor cores engage.
- 2.8x on F5-TTS FFN shapes; 1.0-1.4x on decode-like shapes.
- cuBLAS rejects NVFP4 for M < 128 outright. This is documented nowhere; we filed it as vLLM #48491.
- M must additionally be padded to a multiple of 128.

The benefit is shape-dependent. Large-M compute-bound work sees near-theoretical gains. Small-M decode work sees little at the GEMM level, though it can still gain at the model level from reduced memory traffic.

### 4.2. Weight Sensitivity (W4A16 Fakequant)

The question: do speech model weights tolerate FP4?

Table 2: Whisper large-v3, component-wise, 500 LibriSpeech samples.

| Component     | Layers | WER   |
|---------------|--------|-------|
| baseline BF16 | 0      | 3.37% |
| encoder ALL   | 193    | 3.21% |
| decoder ALL   | 322    | 3.28% |
| ALL weights   | 515    | 3.36% |

Table 3: F5-TTS, component x NFE grid. Every cell reads 4.7% WER; RTF is 0.091 at NFE 32 and 0.056 at NFE 16, indifferent to quantization.

Across models:
- Whisper: all 515 Linear layers quantized, no measurable change in any component.
- F5-TTS: all 167 layers quantized, WER flat across components and both NFE values.
- Orpheus: W4A16 via Marlin, 10/10 on the sanity set, 193 tok/s (1.27x over BF16).

Weights survive FP4 in every architecture we tested.

### 4.3. Activation Sensitivity (W4A4)

What happens when activations are also FP4?

4.3.1. Orpheus (AR-LLM TTS)
- 7/10 sanity phrases produce audio; the failures emit a stop token within the first 0-2 tokens.
- Recalibrating with 50 prompts instead of 10 does not help (6/10). The failure is not a calibration artifact.
- Layer-group logit probe: quantizing the activations of any single group of 4 layers preserves the top-1 token (per-group logit MSE between 0.04 and 1.23). Only quantizing all 28 layers at once flips it.
- One mechanistic detail worth stating plainly: mid-network activations reach |3616|. The induced block scale, 3616/6 = 603, is above the FP8 E4M3 maximum of 448. In a naive implementation this is a NaN; in hardware paths it saturates.
- Conclusion: the error accumulates across the 28-layer decoder. No single layer is the culprit.

4.3.2. F5-TTS (flow-matching DiT)
- Every phrase produces audio, but it is audibly degraded.
- The spectrograms (Figure 5) show what happened: quantization noise floods the whole spectrum. The clean formant bands of BF16 are replaced by uniform noise texture, and even silent regions pick up a noise floor. Perceptually the damage concentrates at high frequencies, where the original energy is low and the noise floor overtakes it.
- NFE interaction: the noise does not accumulate across denoising steps. 32 steps and 16 steps give the same WER under FP4. This is the opposite of the AR case.

4.3.3. One mechanism, two failure modes
Block scaling sets scale = max(|block|)/6.0. Values below about 1/12 of the block maximum quantize to zero; block maxima above 2688 overflow the scale itself. In the AR model these per-step errors compound over hundreds of sequential decode steps until the output distribution drifts far enough to trigger a premature stop. In the DiT each denoising step starts from fresh input, so nothing compounds across steps, but the spectral detail lost inside a step never comes back. It shows up as a persistent noise floor instead of a crash.

### 4.4. Throughput and Memory

Table 4: Orpheus-3B across regimes.

| Regime               | tok/s | RTF  | VRAM    | vs BF16 |
|----------------------|-------|------|---------|---------|
| BF16                 | 152   | 0.57 | 6.2 GB  | 1.0x    |
| W4A16 Marlin         | 193   | 0.46 | ~2.5 GB | 1.27x   |
| NVFP4 (enforce_eager)| 82    | 1.07 | 2.44 GB | 0.54x   |
| NVFP4 + CUDA graphs  | 300   | 0.29 | 2.44 GB | 2.0x    |

CUDA graphs are not optional for NVFP4. Without them, per-kernel launch overhead dominates and NVFP4 is slower than BF16 (82 vs 152 tok/s, a 3.7x swing on the same checkpoint). With them, the memory-bandwidth advantage shows up: 2x throughput at 2.5x less VRAM.

Table 5: F5-TTS across regimes.

| Regime | RTF   | Quality                  | Notes                       |
|--------|-------|--------------------------|-----------------------------|
| BF16   | 0.09  | clean                    | baseline, 32 NFE            |
| W4A16  | 0.10  | clean                    | no speedup, compute-bound   |
| W4A4   | 0.65  | harmonic structure lost  | Python act-quant overhead   |

F5-TTS is compute-bound, so shrinking the weights does not move the bottleneck. The naive W4A4 path is 7x slower than BF16, and the slowdown is the Python-level activation quantization, not the FP4 GEMM. This is what motivated the fused kernel in Section 4.8.

### 4.5. NFE x Quantization (F5-TTS)

Halving NFE from 32 to 16 halves inference time (RTF 0.09 to 0.056) with no quality change, and this still holds with all 167 layers in FP4. The two optimizations do not interact: weight quantization shrinks the model, NFE reduction shrinks the iteration count, and stacking them stacks the savings without stacking the damage. We are not aware of a prior report of this for speech.

### 4.6. Energy Efficiency

Table 6: Orpheus NVFP4 under power caps.

| Power cap | tok/s | Avg power | tok/J | Efficiency vs 325W |
|-----------|-------|-----------|-------|---------------------|
| 250 W     | 316.1 | 231.3 W   | 1.367 | +10.4%              |
| 275 W     | 316.8 | 240.4 W   | 1.318 | +6.5%               |
| 300 W     | 317.7 | 245.8 W   | 1.292 | +4.4%               |
| 325 W     | 318.1 | 257.0 W   | 1.238 | baseline            |

Throughput is flat across the whole range (under 1% variation) while power draw climbs from 231 to 257 W. The workload is not power-limited; the extra wattage becomes heat. Running at the minimum cap is a free 10% on the energy bill.

### 4.7. TTS at Scale (500 LibriSpeech Phrases)

To get past sanity-scale numbers, all three Orpheus regimes generated the same 500 LibriSpeech transcripts; Whisper large-v3 transcribed the outputs and we scored against the source text.

Table 7: Orpheus-3B, 500 phrases.

| Regime | Successful | Success rate | WER (on successful) |
|--------|-----------|--------------|----------------------|
| BF16   | 251/500   | 50.2%        | 41.5%                |
| W4A16  | 221/500   | 44.2%        | 50.4%                |
| NVFP4  | 258/500   | 51.6%        | 47.2%                |

Three things to read out of this table:
- Absolute numbers are low for every regime, BF16 included. That is a domain mismatch: Orpheus is trained on short conversational text, and LibriSpeech transcripts are long literary sentences. The cross-regime comparison on identical inputs is still valid.
- NVFP4 has the highest success rate of any regime, above BF16 itself (258 vs 251). This matches, at scale, what we first noticed by ear: FP4 noise suppresses the repetition and vowel-elongation loops that derail AR generation.
- The stability is not free. NVFP4 WER sits 5.7 points above BF16. The model generates more often and transcribes less precisely. W4A16, the "safe" choice from the weight-sensitivity analysis, unexpectedly loses on both axes at once.

So the picture is a genuine trade-off, not a ranking: BF16 for per-utterance accuracy, NVFP4 for generation stability, and W4A16 dominated by both on this benchmark.

### 4.8. Triton Kernels for the SM120 Gaps

Triton 3.6.0's tl.dot_scaled with e2m1 operands compiles and runs correctly on SM120. We built two kernels on it.

4.8.1. Fused-quantization W4A16 GEMM
A tiled matmul taking BF16 activations and pre-packed FP4 weights (e8m0 block scales) in one kernel. On the F5-TTS FFN shape (2048x5120x1280): 0.201 ms, against 0.098 ms for cuBLAS BF16 and 0.037 ms for cuBLAS NVFP4. Roughly half of cuBLAS BF16 throughput, but 3.5x faster than the Python W4A4 path it replaces. We did not pursue the remaining gap to cuBLAS; closing it is pipelining and memory-layout work, and the point of this kernel is removing the Python overhead identified in Section 4.4.

4.8.2. Small-M kernel (M < 128)
cuBLAS NVFP4 rejects every GEMM with M < 128, which is exactly where speech decode lives (batch=1 means M=1 per step). Our kernel runs correctly at M = 1 through 64. Throughput is 0.5-0.7x of cuBLAS BF16 at these sizes, and that is expected: at M=1 the launch overhead (around 10 us) dwarfs the arithmetic (around 100 ns), so the number format is irrelevant to latency. The honest conclusion: at small M, FP4 buys memory capacity (2.5x model compression), not speed. Speed at small M comes from launch amortization, which is CUDA graphs (Section 4.4), not the GEMM.

---

## 5. Discussion

5.1. Why weights survive
Trained weight distributions are roughly Gaussian with moderate dynamic range. A per-16-block FP8 scale tracks that distribution well, and the coarse E2M1 palette is enough once the scale is right. This is a property of trained weights in general, which is why it held for all three architectures.

5.2. Why activations break
Activations are input-dependent and carry outliers that weights do not. One outlier in a block of 16 pushes the scale up and the other 15 values toward zero; an outlier past 2688 breaks the scale itself. For speech the small values matter: they carry high-frequency detail, timing, and prosody.

5.3. AR vs DiT error propagation
The distinction that matters is whether quantized computation is applied sequentially or in parallel. In the AR model, the error at step t feeds every later step; over 28 layers times hundreds of steps the drift is enough to trigger a premature stop, even though no single layer group is individually responsible (Section 4.3.1). In the DiT, each denoising step starts fresh. Errors hurt that step's output and go no further.

5.4. Quantization as a regularizer
The 500-phrase run turns a listening impression into a number: NVFP4 has the highest success rate of any regime, above BF16. Our best explanation is that the quantization noise perturbs the token distribution enough to break the self-reinforcing loops (vowel elongation, tail repetition) that AR sampling falls into. It behaves like implicit noise injection. The elevated WER says the perturbation is not free. Mapping the trade-off curve (noise magnitude against stability against accuracy) is a concrete follow-up.

5.5. What to deploy
- AR-TTS (Orpheus-like): W4A16 via Marlin for conservative deployments (1.27x, 2.5x memory, clean on the sanity set). NVFP4 with CUDA graphs where throughput and stability matter more than per-utterance accuracy (2x, highest success rate). One caveat: W4A16 lost on both axes in the 500-phrase run, so validate the regime choice on in-domain text before committing.
- DiT-TTS (F5-TTS-like): W4A16 plus NFE 16. Quantization saves memory, NFE reduction saves compute, together about 1.6x with no quality loss. W4A4 needs fused kernels and outlier handling before it is usable.
- ASR (Whisper-like): W4A16, no degradation measured. The hardware NVFP4 path is blocked by llmcompressor not supporting encoder-decoder models. That is a toolchain gap, not a model property.

5.6. Limitations
- Perceptual evaluation rests on automatic metrics and spectrograms. We skipped the formal CMOS test on purpose: with success rates around 50% on out-of-domain text, the listening sample would be biased toward the phrases that happened to generate, and a 15-listener panel cannot resolve effects of this size. In-domain CMOS is future work.
- Absolute Orpheus quality on LibriSpeech text is poor in every regime. The deltas between regimes on identical inputs are still interpretable; the absolute numbers are not.
- One GPU model tested (RTX PRO 6000 Max-Q). Datacenter Blackwell (B200/GB200) may dispatch cuBLAS differently and has different power behavior.
- CosyVoice2 (hybrid LLM+flow) was not evaluated.

---

## 6. SM120 Ecosystem Findings

Twelve constraints we hit on SM120 that appear in no official documentation, collected as a practical guide for anyone running early Blackwell cards. Two are now upstream issues with minimal reproducers: FlashInfer #3945 (check_cuda_arch compares compute capability as strings, so SM 12.0 fails the "sm75 or higher" check) and vLLM #48491 (cuBLAS NVFP4 rejects M < 128, so decode workloads permanently fall back to Marlin).

[Full list of 12, with workarounds: project log / appendix]

---

## 7. Conclusion

Hardware FP4 for speech works, with an asterisk that depends on the architecture. Weight quantization is safe everywhere and deployable now. Activation quantization is the open problem: the AR model fails by accumulating error across sequential steps, the diffusion model by losing spectral detail within each step. The mechanism is the same; the symptom depends on how errors propagate. Deployment follows the roofline: memory-bound AR-TTS gets 2x from FP4 weights plus CUDA graphs, compute-bound DiT-TTS gets more from halving NFE than from any quantization. At scale, the quantization noise itself acts as a regularizer for AR generation, trading per-utterance accuracy for stability, an effect we have not seen reported for speech. And the SM120 stack is young: two of our twelve findings are now upstream issues, and the Triton kernels show the M < 128 gap can be covered without waiting for the vendor.

---

## Figures
1. Block scaling diagram: E2M1 mechanism, outlier suppression, FP8 scale overflow
2. Architecture comparison: Orpheus / F5-TTS / Whisper data flow
3. Sensitivity heatmap: models x components x precision
4. Energy Pareto curve: tok/J vs power cap
5. Spectrograms, F5-TTS BF16 vs W4A4 (produced): harmonic structure vs noise flooding
6. Throughput comparison: all models x all regimes
7. 500-phrase evaluation: success rate and WER by regime

## Tables (all data collected)
1. GEMM microbenchmark
2. Whisper sensitivity grid
3. F5-TTS NFE x component grid
4. Orpheus regime comparison
5. F5-TTS regime comparison
6. Energy sweep
7. 500-phrase evaluation
8. Triton kernel benchmarks
9. SM120 ecosystem findings (appendix)

## Remaining work

Internship deliverables (weeks 5-6):
- Final internship presentation and demo (slides from Tables 1-8 + Figures; live Orpheus NVFP4 demo)
- Repository cleanup and README (the abstract claims public availability; that requires reproduction instructions, pinned environment, checkpoint links)
- Figure production: 6 of 7 need rendering (spectrograms done). Highest priority: sensitivity heatmap, 500-phrase bar chart, energy Pareto curve
- Remaining upstream reports: 2 of 12 findings filed. FlashInfer CCCL-header incompatibility and the root-owned zombie EngineCore are the next actionable candidates

Paper track (deadlines: NeurIPS workshop ~Aug 29, ICASSP Sep 16):
- Prose for all sections (skeleton complete, all data collected)
- Related Work pass: verify no FP4-speech work appeared since project start

Optional (cut without regret if time runs short):
- In-domain 500-phrase eval with conversational text (interpretable absolute numbers)
- CosyVoice2 as fourth architecture

Post-internship:
- CMOS listening test on in-domain text
- Noise-magnitude ablation for the regularizer effect
