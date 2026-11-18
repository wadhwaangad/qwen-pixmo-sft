# Qwen3.5-4B SFT on PixMo-Points

Full-weight SFT (no LoRA) of `Qwen/Qwen3.5-4B` on the gated dataset `sahsah19/pixmo-points-sft`,
on a single 80 GB GPU.

| Setting | Value |
|---|---|
| Method | Full fine-tune, thinking on; LLM + merger trained, ViT frozen |
| Loss | Assistant turn only, including `<\|im_end\|>`; mean over the step's tokens |
| Optimizer | AdamW, betas 0.9/0.999, eps 1e-8, weight decay 0 |
| Learning rate | 1e-5, cosine, 3% warmup |
| Grad clip | 1.0 |
| Batch | 64 per optimizer step (micro-batch x gradient accumulation), 2 epochs |
| Precision | bf16 weights; FP32 AdamW moments; Kahan-compensated updates |
| Image cap | `max_pixels = 1048576` |
| Max length | 4096 tokens; longer samples dropped |
| Eval loss | 2% of train held out by image |
| Seed | 0 |

Memory on one GPU: bf16 weights (9 GB) + bf16 grads (8.4 GB) + FP32 moments (34 GB)
+ bf16 compensation (8.4 GB) + activations. `scripts/gpu_check.py` measures the peak on the longest samples.

## Run

    # From the repo root with the title-filter env active
    export HF_HOME=/gscratch/krishna/$USER/hf-cache
    python scripts/smoke_test.py                       # CPU, no weights
    python scripts/download_data.py                    # gated: hf auth login first
    SFT_ENTRY=scripts/gpu_check.py sbatch --time=01:00:00 scripts/klone_train.sbatch   # memory check
    sbatch scripts/klone_train.sbatch                  # training

Metrics go to `outputs/run1/metrics.jsonl`, the model to `outputs/run1/final`.
