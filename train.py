"""Full-weight SFT of Qwen3.5-4B on PixMo-Points on one GPU (vision encoder frozen)."""
import argparse, itertools, json, math, os, shutil, time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class JsonlDataset(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        with open(self.root / f"{split}.jsonl") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(self.root / row["images"][0]).convert("RGB")
        return {"messages": row["messages"], "image": image}


def strip_image_refs(messages):
    """The template only needs an image placeholder; pixels go to the processor."""
    out = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list):
            c = [{"type": "image"} if p.get("type") == "image" else p for p in c]
        out.append({"role": m["role"], "content": c})
    return out


def load_processor(cfg):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg["model"])
    image_processor = processor.image_processor
    if isinstance(image_processor.size, dict):
        image_processor.size["longest_edge"] = cfg["max_pixels"]
    else:
        image_processor.size.longest_edge = cfg["max_pixels"]
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = cfg["max_pixels"]
    return processor


def render(processor, msgs):
    """Return (full conversation ending at <|im_end|>, prompt with generation header)."""
    end = "<|im_end|>"
    full = processor.apply_chat_template(msgs, tokenize=False)
    full = full[: full.rindex(end) + len(end)]
    prompt = processor.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prompt):
        raise ValueError(f"Prompt is not a prefix of the conversation:\n{prompt[-200:]!r}\n{full[:400]!r}")
    return full, prompt


class Collator:
    """Tokenizes full conversations and masks everything before the assistant reply."""

    def __init__(self, processor):
        self.processor = processor
        processor.tokenizer.padding_side = "right"

    def __call__(self, batch):
        p = self.processor
        msgs = [strip_image_refs(b["messages"]) for b in batch]
        images = [[b["image"]] for b in batch]
        full, prompt = map(list, zip(*(render(p, m) for m in msgs)))
        enc = p(text=full, images=images, padding=True, return_tensors="pt")
        prompt_len = p(text=prompt, images=images, padding=True, return_tensors="pt")["attention_mask"].sum(1)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, n in enumerate(prompt_len.tolist()):
            labels[i, :n] = -100
        enc["labels"] = labels
        return enc


def lr_factor(step, warmup, total):
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def steps_per_epoch(n_rows, batch_size):
    return math.ceil(n_rows / batch_size)


def freeze_vision(model):
    """Train the language model and connector (visual.merger); freeze the rest of vision."""
    for name, param in model.named_parameters():
        dotted = f".{name}."
        param.requires_grad = ".visual." not in dotted or ".merger." in dotted


class KahanAdamW(torch.optim.Optimizer):
    """torch.optim.AdamW's update with FP32 moments and Kahan-compensated weights.

    bf16 weights alone would round 1e-5-sized updates away; the bf16 compensation
    buffer carries the lost remainder into the next step.
    """

    CHUNK = 1 << 24  # bounds the FP32 scratch memory per parameter

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, offload=False):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay))
        # offload=True keeps the FP32 moments in pinned CPU memory and streams them through the GPU
        # chunk by chunk; the arithmetic is identical, it only saves ~34 GB of GPU memory.
        self.offload = offload

    def _moment_home(self, p, t):
        """Place a moment tensor where it lives between steps (pinned CPU if offloading)."""
        if self.offload:
            if t.device.type != "cpu" or (torch.cuda.is_available() and not t.is_pinned()):
                t = t.cpu().pin_memory() if torch.cuda.is_available() else t.cpu()
            return t
        return t.to(p.device)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["compensation"] = torch.zeros_like(p)
                for k in ("exp_avg", "exp_avg_sq"):  # also fixes placement after a checkpoint load
                    state[k] = self._moment_home(p, state[k])
                state["compensation"] = state["compensation"].to(p.device)
                state["step"] += 1
                bias1 = 1 - beta1 ** state["step"]
                bias2_sqrt = math.sqrt(1 - beta2 ** state["step"])
                flat = [t.view(-1) for t in (p, p.grad, state["exp_avg"], state["exp_avg_sq"], state["compensation"])]
                for w, g, m_home, v_home, c in zip(*(t.split(self.CHUNK) for t in flat)):
                    m, v = m_home.to(p.device, non_blocking=True), v_home.to(p.device, non_blocking=True)
                    g, old = g.float(), w.float()
                    m.lerp_(g, 1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    if m is not m_home:
                        m_home.copy_(m)
                        v_home.copy_(v)
                    update = m / (v.sqrt() / bias2_sqrt).add_(eps) * (-lr / bias1)
                    if wd:
                        update.sub_(old, alpha=lr * wd)
                    update.add_(c.float())
                    new = (old + update).to(w.dtype)
                    c.copy_(update - (new.float() - old))
                    w.copy_(new)


def setup_model(cfg, device):
    from transformers import AutoModelForImageTextToText

    processor = load_processor(cfg)
    dtype = getattr(torch, cfg["weight_dtype"])
    model = AutoModelForImageTextToText.from_pretrained(cfg["model"], dtype=dtype).to(device)
    freeze_vision(model)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    return processor, model, params


def make_optimizer(cfg, params, total):
    warmup = math.ceil(cfg["warmup_frac"] * total)
    offload = cfg.get("offload_moments", "auto")
    if offload == "auto":  # moments need ~34 GB; full on-GPU training peaks at ~68 GB
        offload = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 75 * 2**30
    print(f"optimizer moments on {'CPU (offloaded)' if offload else 'GPU'}", flush=True)
    optim = KahanAdamW(params, lr=cfg["lr"], betas=cfg["betas"], eps=cfg["eps"], weight_decay=cfg["weight_decay"],
                       offload=bool(offload))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_factor(s, warmup, total))
    return optim, sched, warmup


def save_checkpoint(out, model, optim, sched, step, params):
    """Save trainable weights + optimizer/scheduler state; atomically replaces out/ckpt.

    Frozen (vision) weights are reloaded from the base model on resume. Failures (e.g. full disk)
    are logged and training continues.
    """
    tmp, dst = out / "ckpt.tmp", out / "ckpt"
    t0 = time.time()
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        names = {id(p): n for n, p in model.named_parameters()}
        torch.save({names[id(p)]: p.detach() for p in params}, tmp / "trainable.pt")
        # Per-param state saved directly: Optimizer.load_state_dict would cast the FP32 moments to bf16.
        torch.save({"optim_state": [optim.state[p] for p in params], "sched": sched.state_dict(), "step": step,
                    "cuda_rng": torch.cuda.get_rng_state(), "cpu_rng": torch.get_rng_state()}, tmp / "state.pt")
        (tmp / "DONE").write_text(str(step))
        old = out / "ckpt.old"
        shutil.rmtree(old, ignore_errors=True)
        if dst.exists():
            os.replace(dst, old)
        os.replace(tmp, dst)
        shutil.rmtree(old, ignore_errors=True)
        print(f"checkpoint saved at step {step} in {time.time() - t0:.0f}s", flush=True)
    except Exception as e:  # never kill a long run over a checkpoint
        print(f"WARNING: checkpoint at step {step} failed: {e!r}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)


def load_checkpoint(out, model, optim, sched, params, device):
    """Return the step to resume from (0 if no complete checkpoint)."""
    for d in (out / "ckpt", out / "ckpt.old"):
        if (d / "DONE").exists():
            weights = torch.load(d / "trainable.pt", map_location=device)
            missing = set(weights) - {n for n, _ in model.named_parameters()}
            if missing:
                raise RuntimeError(f"checkpoint has unknown params: {sorted(missing)[:5]}")
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in weights:
                        p.copy_(weights[n])
            del weights
            # Load to CPU; KahanAdamW.step moves each tensor where it belongs (avoids a 50 GB GPU spike).
            st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
            if len(st["optim_state"]) != len(params):
                raise RuntimeError("checkpoint optimizer state does not match trainable params")
            for p, ps in zip(params, st["optim_state"]):
                optim.state[p] = ps
            sched.load_state_dict(st["sched"])
            for group, lr in zip(optim.param_groups, sched.get_last_lr()):
                group["lr"] = lr
            st["cuda_rng"], st["cpu_rng"] = st["cuda_rng"].cpu(), st["cpu_rng"].cpu()
            torch.cuda.set_rng_state(st["cuda_rng"])
            torch.set_rng_state(st["cpu_rng"])
            print(f"resumed from {d} at step {st['step']}", flush=True)
            return st["step"]
    return 0


def label_tokens(batch):
    return int((batch["labels"][:, 1:] != -100).sum())


def train_step(model, micros, optim, sched, params, cfg, device):
    """One optimizer step over all micro-batches; loss is the mean over the step's label tokens."""
    n_items = sum(label_tokens(b) for b in micros)
    step_loss = 0.0
    for batch in micros:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False, num_items_in_batch=n_items).loss
        loss.backward()
        step_loss += loss.item()
    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
    optim.step()
    sched.step()
    optim.zero_grad(set_to_none=True)
    return step_loss


def evaluate(model, loader, device):
    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            n = label_tokens(batch)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_sum += model(**batch, use_cache=False).loss.item() * n
            tokens += n
    model.train()
    return loss_sum / tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    cfg = json.load(open(ap.parse_args().config))

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda")
    if cfg["batch_size"] % cfg["micro_batch_size"]:
        raise ValueError("batch_size must be a multiple of micro_batch_size")
    accum = cfg["batch_size"] // cfg["micro_batch_size"]

    processor, model, params = setup_model(cfg, device)
    train_ds = JsonlDataset(cfg["data_dir"], "prepared_train")
    val_ds = JsonlDataset(cfg["data_dir"], "prepared_heldout")
    kw = dict(batch_size=cfg["micro_batch_size"], num_workers=cfg["num_workers"],
              collate_fn=Collator(processor), pin_memory=True)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    per_epoch = steps_per_epoch(len(train_ds), cfg["batch_size"])
    total = cfg["epochs"] * per_epoch
    optim, sched, warmup = make_optimizer(cfg, params, total)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "metrics.jsonl", "a", buffering=1)
    print(f"train={len(train_ds)} heldout={len(val_ds)} steps={total} warmup={warmup} "
          f"micro={cfg['micro_batch_size']}x{accum} trainable={sum(p.numel() for p in params):,}", flush=True)

    resume_step = load_checkpoint(out, model, optim, sched, params, device)
    ckpt_every = cfg.get("checkpoint_every", 0)
    step, start = resume_step, time.time()
    for epoch in range(cfg["epochs"]):
        if step >= (epoch + 1) * per_epoch:
            continue  # epoch finished before the checkpoint
        # Deterministic per-epoch order so a resumed run skips exactly the micro-batches already trained on.
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.randperm(len(train_ds), generator=generator).tolist()
        skip = (step - epoch * per_epoch) * accum * cfg["micro_batch_size"]
        batches = iter(DataLoader(train_ds, sampler=order[skip:], **kw))
        while step < (epoch + 1) * per_epoch:
            micros = list(itertools.islice(batches, accum))
            if not micros:
                break
            loss = train_step(model, micros, optim, sched, params, cfg, device)
            step += 1
            if step == 1 or step % cfg["log_every"] == 0:
                rec = dict(step=step, loss=loss, lr=sched.get_last_lr()[0], elapsed=time.time() - start,
                           peak_gb=torch.cuda.max_memory_allocated() / 2**30)
                print(rec, flush=True)
                log.write(json.dumps(rec) + "\n")
            if ckpt_every and step % ckpt_every == 0 and step < total:
                save_checkpoint(out, model, optim, sched, step, params)
        step = (epoch + 1) * per_epoch
        rec = dict(step=step, epoch=epoch + 1, eval_loss=evaluate(model, val_loader, device))
        print(rec, flush=True)
        log.write(json.dumps(rec) + "\n")

    model.save_pretrained(out / "final")
    processor.save_pretrained(out / "final")


if __name__ == "__main__":
    main()
