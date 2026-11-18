"""Hold out 2% of train by image and drop over-length samples. Safe to rerun."""
import json, os, random, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image
from train import load_processor, render, strip_image_refs


def split_by_image(rows, frac, seed):
    images = sorted({r["images"][0] for r in rows})
    random.Random(seed).shuffle(images)
    held = set(images[:round(len(images) * frac)])
    return [r for r in rows if r["images"][0] not in held], [r for r in rows if r["images"][0] in held]


def row_length(processor, row, data_dir):
    image = Image.open(Path(data_dir) / row["images"][0]).convert("RGB")
    full, _ = render(processor, strip_image_refs(row["messages"]))
    return processor(text=[full], images=[[image]], return_tensors="pt")["input_ids"].shape[1]


def init(cfg_):
    global cfg, processor
    cfg, processor = cfg_, load_processor(cfg_)


def length(row):
    return row_length(processor, row, ROOT / cfg["data_dir"])


def main():
    conf = json.load(open(ROOT / "config.json"))
    data = ROOT / conf["data_dir"]
    if (data / "prepared_stats.json").exists():
        print("already prepared:", (data / "prepared_stats.json").read_text())
        return
    rows = [json.loads(l) for l in open(data / "train.jsonl")]
    train, held = split_by_image(rows, conf["heldout_frac"], conf["seed"])
    stats = {"input": len(rows)}
    with Pool(len(os.sched_getaffinity(0)), init, (conf,)) as pool:
        for name, part in (("train", train), ("heldout", held)):
            lens = pool.map(length, part, chunksize=32)
            kept = [{**r, "_len": n} for r, n in zip(part, lens) if n <= conf["max_seq_len"]]
            stats[name] = {"rows": len(kept), "dropped_too_long": len(part) - len(kept), "max_len": max(lens)}
            with open(data / f"prepared_{name}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in kept)
    (data / "prepared_stats.json").write_text(json.dumps(stats))
    print(stats)


if __name__ == "__main__":
    main()
