"""Download the gated dataset (run `hf auth login` or set HF_TOKEN first)."""
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(__file__).resolve().parent.parent / "data" / "pixmo-points-sft"
snapshot_download("sahsah19/pixmo-points-sft", repo_type="dataset", local_dir=target, max_workers=8)
# local_dir keeps one metadata file per download; drop them to save file quota.
shutil.rmtree(target / ".cache", ignore_errors=True)
print("rows files:", sorted(p.name for p in target.glob("*.jsonl")))
print("images:", sum(1 for _ in target.glob("images/*/*")))
