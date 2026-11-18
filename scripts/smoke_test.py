"""CPU check of the template, masking, freezing, optimizer and step math. No weights needed."""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText
from prepare_data import split_by_image
from train import Collator, KahanAdamW, freeze_vision, load_processor, lr_factor, steps_per_epoch

cfg = json.load(open(ROOT / "config.json"))
processor = load_processor(cfg)
reply = "<think>\nI see <point>[[100,200],[300,400]]</point>.\n</think>\n\n2"
rows = [{"messages": [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"How many cats?{'?' * i}"}]},
    {"role": "assistant", "content": [{"type": "text", "text": reply}]}],
    "image": Image.new("RGB", (448 + 56 * i, 336), "white")} for i in range(2)]
batch = Collator(processor)(rows)
for i in range(2):
    ids = batch["labels"][i]
    decoded = processor.tokenizer.decode(ids[ids != -100])
    assert "<point>[[100,200],[300,400]]</point>" in decoded and decoded.endswith("2<|im_end|>")
    assert "How many cats" not in decoded and "<|image_pad|>" not in decoded
print("labels OK:", repr(decoded))

big = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[[Image.new("RGB", (3000, 2000))]], return_tensors="pt")
pixels = big["image_grid_thw"].prod().item() * processor.image_processor.patch_size ** 2
assert pixels <= cfg["max_pixels"]
print("image cap OK:", pixels)

with torch.device("meta"):
    model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(cfg["model"]))
freeze_vision(model)
groups = {}
for name, p in model.named_parameters():
    key = ("merger" if ".merger." in name else "vision") if ".visual." in name else "language"
    g = groups.setdefault(key, [0, 0]); g[p.requires_grad] += p.numel()
assert groups["vision"][1] == 0 and groups["merger"][0] == 0 and groups["language"][0] == 0
print("freezing OK:", groups)

# KahanAdamW equals torch AdamW in FP32, and keeps 1e-5 updates that plain bf16 loses.
torch.manual_seed(0)
a = torch.randn(1000, requires_grad=True); b = a.detach().clone().requires_grad_()
ref = torch.optim.AdamW([a], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
mine = KahanAdamW([b], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
for _ in range(20):
    g = torch.randn(1000)
    a.grad, b.grad = g.clone(), g.clone()
    ref.step(); mine.step()
assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
# Realistic weights (+-0.02): plain bf16 loses every 1e-5 update; Kahan tracks FP32 closely.
w0 = torch.linspace(-0.02, 0.02, 64)
plain = w0.to(torch.bfloat16).requires_grad_(); kahan = w0.to(torch.bfloat16).requires_grad_()
exact = w0.to(torch.bfloat16).float().requires_grad_()
po = torch.optim.AdamW([plain], lr=1e-5, weight_decay=0.0); ko = KahanAdamW([kahan], lr=1e-5)
eo = torch.optim.AdamW([exact], lr=1e-5, weight_decay=0.0)
torch.manual_seed(1)
for _ in range(300):
    g = torch.randn(64) + 0.5
    plain.grad, kahan.grad, exact.grad = g.to(torch.bfloat16), g.to(torch.bfloat16), g.to(torch.bfloat16).float()
    po.step(); ko.step(); eo.step()
start = w0.to(torch.bfloat16).float()
moved_exact = (start - exact.detach()).abs().mean().item()
true_kahan = kahan.detach().float() + ko.state[kahan]["compensation"].float()
plain_err = (plain.detach().float() - exact.detach()).abs().mean().item()
kahan_err = (true_kahan - exact.detach()).abs().mean().item()
print(f"optimizer OK: exact moved {moved_exact:.2e}; error plain bf16 {plain_err:.2e}, Kahan {kahan_err:.2e}")
assert kahan_err < 0.03 * moved_exact and plain_err > 0.3 * moved_exact
fake = [{"images": [f"images/{i}.jpg"]} for i in range(1000) for _ in range(2)]
train, held = split_by_image(fake, 0.02, cfg["seed"])
assert len(held) == 40 and not {r["images"][0] for r in train} & {r["images"][0] for r in held}

total = 2 * steps_per_epoch(9800, 64)
warmup = math.ceil(0.03 * total)
assert 2 * steps_per_epoch(14390, 64) == 450
print("steps at 9800 rows:", total, "warmup:", warmup, "start lr:", lr_factor(0, warmup, total))
print("SMOKE TEST PASSED")
