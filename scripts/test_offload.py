"""GPU self-test: offloaded KahanAdamW must match on-GPU KahanAdamW bit for bit, including across
a checkpoint save/resume. Runs in seconds; the training job aborts if it fails."""
import sys, tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train

CFG = dict(warmup_frac=0.1, lr=1e-2, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0)


def make(offload):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 1)).to("cuda", torch.bfloat16)
    for p in m[0].parameters():
        p.requires_grad = False
    params = [p for p in m.parameters() if p.requires_grad]
    o, s, _ = train.make_optimizer(dict(CFG, offload_moments=offload), params, 20)
    o.CHUNK = 7  # force many chunks per tensor
    return m, params, o, s


def step(m, o, s, i):
    g = torch.Generator().manual_seed(i)
    x = torch.randn(16, 64, generator=g).to("cuda", torch.bfloat16)
    m(x).float().pow(2).mean().backward()
    o.step(); s.step(); o.zero_grad()


def run(offload, n=10):
    m, params, o, s = make(offload)
    for i in range(n):
        step(m, o, s, i)
    return params, o


ref, o_ref = run(False)
off, o_off = run(True)
assert o_off.state[off[0]]["exp_avg"].device.type == "cpu" and o_off.state[off[0]]["exp_avg"].is_pinned()
assert o_ref.state[ref[0]]["exp_avg"].device.type == "cuda"
for a, b in zip(ref, off):
    assert torch.equal(a, b), (a.float() - b.float()).abs().max()
for a, b in zip(ref, off):
    for k in ("exp_avg", "exp_avg_sq", "compensation"):
        assert torch.equal(o_ref.state[a][k].cpu(), o_off.state[b][k].cpu()), k

# save at step 6 with offload, resume into a fresh offloaded optimizer, finish -> must equal reference
out = Path(tempfile.mkdtemp())
m, params, o, s = make(True)
for i in range(6):
    step(m, o, s, i)
train.save_checkpoint(out, m, o, s, 6, params)
m, params, o, s = make(True)
assert train.load_checkpoint(out, m, o, s, params, "cuda") == 6
for i in range(6, 10):
    step(m, o, s, i)
for a, b in zip(ref, params):
    assert torch.equal(a, b), (a.float() - b.float()).abs().max()
print("offload self-test OK", flush=True)
