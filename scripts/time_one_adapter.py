#!/usr/bin/env python
"""
time_one_adapter.py (v2) -- calibrate the AdapTrack compute budget.

Three fixes, all driven by your T4 measurements:

1. DTYPE BY ARCHITECTURE. bf16 tensor cores arrived with Ampere (sm_80).
   A T4 is Turing (sm_75): it has fp16 tensor cores but NO bf16 tensor-core
   path, so bf16 matmuls fall back to unaccelerated kernels. Note that
   torch.cuda.is_bf16_supported() returns True on Turing anyway -- it reports
   whether bf16 can be ALLOCATED, not accelerated. v1 hard-coded bf16 and was
   running your T4 off the tensor cores entirely.

2. DYNAMIC PADDING. v1 used padding="max_length", so every sequence cost
   max_len tokens regardless of content. Your two runs gave 439 and 467 tok/s
   -- essentially flat -- the signature of a workload whose cost tracks PADDED
   tokens. v2 tokenizes without padding, pads per batch to a multiple of 8,
   and groups by length so batches are internally uniform.

3. FIXED PROJECTION. v1 multiplied the measured run by --n-adapters
   regardless of --n-examples, so your 500-example run projected 6.2 GPU-h
   against a real 3000-example figure of ~36. v2 scales train time to
   --target-examples and amortizes the base-model load across the pool
   (you load the base once and re-init a fresh LoRA per adapter).

  python time_one_adapter.py --dataset medmcqa --n-examples 3000
"""

import argparse
import json
import time

import torch


def human(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def pick_dtype(force=None):
    """Return (dtype, use_fp16_flag, use_bf16_flag, reason)."""
    major, minor = torch.cuda.get_device_capability(0)
    cc = f"sm_{major}{minor}"
    if force == "fp32":
        return torch.float32, False, False, f"forced fp32 on {cc}"
    if force == "bf16":
        return torch.bfloat16, False, True, f"forced bf16 on {cc}"
    if force == "fp16":
        return torch.float16, True, False, f"forced fp16 on {cc}"
    if major >= 8:
        return torch.bfloat16, False, True, f"{cc} >= sm_80, bf16 tensor cores present"
    return torch.float16, True, False, (
        f"{cc} < sm_80: no bf16 tensor cores (Turing/Volta), using fp16")


def build_texts(dataset, n, seed):
    from datasets import load_dataset
    if dataset == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [f"Question: {r['question']}\nOptions: A) {r['opa']} B) {r['opb']} "
                f"C) {r['opc']} D) {r['opd']}\nAnswer: {'ABCD'[r['cop']]}" for r in ds]
    if dataset == "pubmedqa":
        # pqa_artificial (211,269 rows). Question + long_answer only, NO context:
        # PQA-A contexts average ~238 tokens and would roughly triple compute for
        # a domain signal the question and conclusion already carry.
        ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [f"Question: {r['question']}\nAnswer: {r['long_answer']}" for r in ds]
    if dataset == "cnndm":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [f"Article: {r['article'][:2000]}\nSummary: {r['highlights']}" for r in ds]
    if dataset == "ledgar":
        ds = load_dataset("coastalcph/lex_glue", "ledgar")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [f"Provision: {r['text']}\nCategory: {r['label']}" for r in ds]
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset", default="medmcqa",
                    choices=["medmcqa", "pubmedqa", "cnndm", "ledgar"])
    ap.add_argument("--n-examples", type=int, default=3000)
    ap.add_argument("--target-examples", type=int, default=None,
                    help="project to this many; defaults to --n-examples")
    ap.add_argument("--max-len", type=int, default=512,
                    help="TRUNCATION cap only. With dynamic padding this no "
                         "longer sets the cost of every sequence.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--static-padding", action="store_true",
                    help="reproduce v1 behaviour, for the A/B")
    ap.add_argument("--no-group-by-length", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--n-adapters", type=int, default=80)
    ap.add_argument("--cu-observed", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--keep", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no GPU -- this measurement is meaningless on CPU"
    gpu = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    dtype, use_fp16, use_bf16, why = pick_dtype(
        None if args.dtype == "auto" else args.dtype)

    print(f"GPU        : {gpu}  sm_{major}{minor}  ({total_vram:.1f} GB)")
    print(f"dtype      : {str(dtype).replace('torch.', '')}  -- {why}")
    print(f"padding    : {'static max_length (v1 mode)' if args.static_padding else 'dynamic, multiple of 8'}")

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).cuda()
    load_s = time.time() - t_load

    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))

    # fp16 + GradScaler raises "Attempting to unscale FP16 gradients" when the
    # trainable params are themselves fp16. Keep the base in fp16, promote only
    # the LoRA params to fp32 -- 2.25M params, ~9 MB, free.
    if use_fp16:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params: {trainable:,}")

    texts = build_texts(args.dataset, args.n_examples, args.seed)

    if args.static_padding:
        ds = Dataset.from_dict({"text": texts}).map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_len,
                          padding="max_length"),
            batched=True, remove_columns=["text"])
        collator = DataCollatorForLanguageModeling(tok, mlm=False)
    else:
        ds = Dataset.from_dict({"text": texts}).map(
            lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
            batched=True, remove_columns=["text"])
        collator = DataCollatorForLanguageModeling(tok, mlm=False,
                                                   pad_to_multiple_of=8)

    lens = [len(x) for x in ds["input_ids"]]
    srt = sorted(lens)
    p50, p95 = srt[len(srt) // 2], srt[int(len(srt) * 0.95)]
    real_tokens = sum(lens)
    print(f"examples   : {len(texts)} from {args.dataset}")
    print(f"token len  : p50={p50} p95={p95} max={max(lens)} "
          f"mean={real_tokens / len(lens):.0f}")
    if args.static_padding:
        waste = 1 - real_tokens / (len(lens) * args.max_len)
        print(f"pad waste  : {100 * waste:.0f}% of compute is padding "
              f"at max_length={args.max_len}")

    group = not (args.no_group_by_length or args.static_padding)
    if group:
        ds = ds.add_column("length", lens)

    targs = TrainingArguments(
        output_dir=args.keep or "/tmp/_timing_adapter",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        fp16=use_fp16, bf16=use_bf16,
        group_by_length=group,
        logging_steps=25, save_strategy="no", report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    out = trainer.train()
    torch.cuda.synchronize()
    train_s = time.time() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_resv = torch.cuda.max_memory_reserved() / 1e9
    if args.keep:
        model.save_pretrained(args.keep)

    steps = out.global_step
    print("\n" + "=" * 68 + "\nMEASURED\n" + "=" * 68)
    print(f"model load        : {human(load_s)}")
    print(f"train wall-clock  : {human(train_s)}  ({train_s:.0f}s)")
    print(f"  optimizer steps : {steps}  ({train_s / max(steps, 1):.2f} s/step)")
    print(f"  real-token rate : {real_tokens * args.epochs / train_s:,.0f} tok/s")
    print("    (non-padding tokens only -- this is the number to compare across")
    print("     configs; padded tok/s stays flat and tells you nothing)")
    print(f"peak VRAM alloc   : {peak_alloc:.2f} GB")
    print(f"peak VRAM reserved: {peak_resv:.2f} GB   <-- must fit")

    target = args.target_examples or args.n_examples
    scale = target / len(texts)
    train_at_target = train_s * scale
    pool_h = (load_s + train_at_target * args.n_adapters) / 3600

    print("\n" + "=" * 68)
    print(f"PROJECTION: {args.n_adapters} adapters x {target} examples")
    print("=" * 68)
    if scale != 1.0:
        print(f"measured on {len(texts)} ex, scaled x{scale:.2f} to {target} ex")
        print("  (linear in examples -- valid only if p50 length is comparable)")
    print(f"train per adapter : {human(train_at_target)}")
    print(f"base load         : {human(load_s)} ONCE, not x{args.n_adapters}")
    print(f"POOL TOTAL        : {pool_h:.1f} GPU-hours")

    if args.cu_observed:
        rate = args.cu_observed / (train_s / 3600)
        print(f"\nCU rate on {gpu}: {rate:.2f} CU/hr -> pool ~{rate * pool_h:.0f} CU")
    else:
        print("\nCU: read the Colab counter before/after, re-run with --cu-observed")

    print("\n" + "=" * 68 + "\nVERDICTS\n" + "=" * 68)
    print(f"fits 16GB T4      : {'YES' if peak_resv < 15.0 else 'NO'} ({peak_resv:.2f} GB)")
    fits8 = peak_resv < 7.5
    print(f"fits 8GB RTX 4060 : {'YES' if fits8 else 'NO'}"
          f"{'' if fits8 else '  -> --gradient-checkpointing or --batch-size 2'}")
    print(f"budget <= 25 GPU-h: {'YES' if pool_h <= 25 else 'NO'} ({pool_h:.1f} h)")

    tag = args.tag or (f"{args.dataset}_"
                       f"{'static' if args.static_padding else 'dyn'}_"
                       f"{str(dtype).split('.')[-1]}")
    with open(f"timing_{tag}.json", "w") as f:
        json.dump({
            "tag": tag, "gpu": gpu, "sm": f"{major}{minor}", "dtype": str(dtype),
            "static_padding": args.static_padding, "group_by_length": group,
            "n_examples": len(texts), "max_len": args.max_len,
            "tok_p50": p50, "tok_p95": p95, "real_tokens": real_tokens,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "load_s": load_s, "train_s": train_s, "steps": steps,
            "s_per_step": train_s / max(steps, 1),
            "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_resv,
            "target_examples": target, "pool_hours": pool_h,
        }, f, indent=2)
    print(f"\nwrote timing_{tag}.json")


if __name__ == "__main__":
    main()