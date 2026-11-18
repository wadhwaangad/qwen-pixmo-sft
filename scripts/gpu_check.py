"""Two optimizer steps on the 64 longest training samples: finds a micro-batch that fits."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from train import Collator, JsonlDataset, label_tokens, make_optimizer, setup_model, train_step

cfg = json.load(open(ROOT / "config.json"))
torch.manual_seed(cfg["seed"])
device = torch.device("cuda")
props = torch.cuda.get_device_properties(device)
print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB", flush=True)

processor, model, params = setup_model(cfg, device)
print(f"weights loaded: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
ds = JsonlDataset(ROOT / cfg["data_dir"], "prepared_train")
ds.rows = sorted(ds.rows, key=lambda r: -r["_len"])[: cfg["batch_size"]]
print("longest sample lengths:", [r["_len"] for r in ds.rows[:5]], flush=True)
collate = Collator(processor)

one = collate([ds[0]])
n = label_tokens(one)
batch = {k: v.to(device) for k, v in one.items()}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    mean = model(**batch, use_cache=False).loss.item()
    halved = model(**batch, use_cache=False, num_items_in_batch=2 * n).loss.item()
print(f"num_items_in_batch check: mean={mean:.4f} with 2x count={halved:.4f} ratio={halved / mean:.3f}", flush=True)
assert abs(halved / mean - 0.5) < 0.01, "model ignores num_items_in_batch"

optim, sched, _ = make_optimizer(cfg, params, total=308)
for micro in sorted({cfg["micro_batch_size"], 4, 2, 1}, reverse=True):
    if cfg["batch_size"] % micro:
        continue
    try:
        optim.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        micros = list(DataLoader(ds, batch_size=micro, collate_fn=collate, num_workers=cfg["num_workers"]))
        for s in range(2):
            t = time.time()
            loss = train_step(model, micros, optim, sched, params, cfg, device)
            torch.cuda.synchronize()
            print(f"micro={micro} step={s + 1} loss={loss:.4f} sec={time.time() - t:.1f}", flush=True)
        print(f"FITS micro={micro} peak={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"of {props.total_memory / 2**30:.1f}", flush=True)
        break
    except torch.OutOfMemoryError:
        print(f"OOM micro={micro}", flush=True)
else:
    sys.exit("nothing fits")
