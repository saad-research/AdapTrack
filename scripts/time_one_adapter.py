#!/usr/bin/env python
"""
time_one_adapter.py -- calibrate the number the whole AdapTrack plan rests on.

Trains exactly ONE LoRA adapter under the roadmap's config (r=8, alpha=16,
q/k/v/o, 1 epoch, ~3k examples) and reports wall-clock, peak VRAM, and the
projection to an 80-adapter pool.

  python time_one_adapter.py --dataset pubmedqa --n-examples 3000

Read Colab compute units off the UI immediately before and after the run --
that is the only trustworthy CU measurement. Pass the delta back in with
--cu-observed to get a CU projection printed alongside the wall-clock one.

Everything below is instrumentation, not science: it writes no adapter you
should keep. Use --keep to save one for verify_merge_semantics.py.
"""

import argparse
import json
import os
import time

import torch


def human(sec):
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def build_texts(dataset, n, seed):
    from datasets import load_dataset
    if dataset == "pubmedqa":
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [f"Question: {r['question']}\nAnswer: {r['long_answer']}" for r in ds]
    if dataset == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [
            f"Question: {r['question']}\nOptions: A) {r['opa']} B) {r['opb']} "
            f"C) {r['opc']} D) {r['opd']}\nAnswer: {'ABCD'[r['cop']]}" for r in ds
        ]
    if dataset == "cnndm":
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0")["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
        return [f"Article: {r['article'][:2000]}\nSummary: {r['highlights']}" for r in ds]
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset", default="pubmedqa",
                    choices=["pubmedqa", "medmcqa", "cnndm"])
    ap.add_argument("--n-examples", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="cuts VRAM ~30-40%%, costs ~20-30%% time. Try both.")
    ap.add_argument("--load-4bit", action="store_true",
                    help="QLoRA. Almost certainly unnecessary for 1.1B in 16GB.")
    ap.add_argument("--n-adapters", type=int, default=80,
                    help="pool size to project to")
    ap.add_argument("--cu-observed", type=float, default=None,
                    help="compute units this run consumed, read off the Colab UI")
    ap.add_argument("--keep", default=None, help="path to save the adapter")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no GPU visible -- this measurement is meaningless on CPU"
    gpu = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU        : {gpu}  ({total_vram:.1f} GB)")
    print(f"torch      : {torch.__version__}")

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
    kw = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.base, **kw).cuda()
    load_s = time.time() - t_load

    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params: {trainable:,} "
          f"({trainable * 4 / 1e6:.1f} MB fp32, ~{trainable * 8 / 1e6:.1f} MB AdamW state)")

    texts = build_texts(args.dataset, args.n_examples, args.seed)
    print(f"examples   : {len(texts)} from {args.dataset}")
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=args.seq_len,
                      padding="max_length"),
        batched=True, remove_columns=["text"])

    outdir = args.keep or "/tmp/_timing_adapter"
    targs = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False))

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

    # ---------------- report ----------------
    steps = out.global_step
    tokens = len(texts) * args.seq_len * args.epochs
    print("\n" + "=" * 68)
    print("MEASURED")
    print("=" * 68)
    print(f"model load        : {human(load_s)}")
    print(f"train wall-clock  : {human(train_s)}  ({train_s:.0f}s)")
    print(f"  optimizer steps : {steps}   ({train_s / max(steps,1):.2f} s/step)")
    print(f"  throughput      : {tokens / train_s:,.0f} tok/s")
    print(f"peak VRAM alloc   : {peak_alloc:.2f} GB")
    print(f"peak VRAM reserved: {peak_resv:.2f} GB   <-- the number that must fit")
    print(f"headroom on this  : {total_vram - peak_resv:.2f} GB free of {total_vram:.1f}")

    per_adapter_s = load_s + train_s
    total_h = per_adapter_s * args.n_adapters / 3600
    print("\n" + "=" * 68)
    print(f"PROJECTION to {args.n_adapters} adapters")
    print("=" * 68)
    print(f"per adapter (load+train): {human(per_adapter_s)}")
    print(f"pool total              : {total_h:.1f} GPU-hours")
    print(f"  + DP-LoRA sweep (~x1.5 on a subset, roadmap est. 10-20h)")
    print(f"  ~ grand total          : {total_h * 1.3 + 15:.0f}-{total_h * 1.5 + 20:.0f} GPU-hours")

    if args.cu_observed:
        cu_rate = args.cu_observed / (per_adapter_s / 3600)
        print(f"\nCU: {args.cu_observed:.2f} for this run -> {cu_rate:.2f} CU/hr on {gpu}")
        print(f"    {args.n_adapters} adapters ~ {cu_rate * total_h:.0f} CU")
        print(f"    Pro = 100 CU/mo, Pro+ = 500 CU/mo")
    else:
        print("\nCU not measured. Read the Colab UI counter before and after this run")
        print("and re-run with --cu-observed <delta>. Do not trust published CU rates;")
        print("they change and are not officially documented per-GPU.")

    print("\n" + "=" * 68)
    print("VERDICTS")
    print("=" * 68)
    print(f"fits 16GB T4      : {'YES' if peak_resv < 15.0 else 'NO'} "
          f"(peak reserved {peak_resv:.2f} GB)")
    fits8 = peak_resv < 7.5
    print(f"fits 8GB RTX 4060 : {'YES' if fits8 else 'NO'}"
          f"{'' if fits8 else '  -> try --gradient-checkpointing, or --seq-len 256, or --batch-size 2'}")
    if total_h > 40:
        print(f"\nWARNING: {total_h:.0f} GPU-h exceeds the roadmap's 20-40h estimate.")
        print("Cut in this order (cheapest loss of scientific value first):")
        print("  1. dataset size 3000 -> 2000        (~33% off, weakest adapters still fine)")
        print("  2. adapters 80 -> 60                (~25% off, keeps 20/family for 3 families)")
        print("  3. epochs 1 -> 0.5                  (~50% off, but weakens the signal you")
        print("                                       are trying to detect -- do this last)")
        print("  Do NOT cut rank below 8: the spectral features ARE the paper.")
    else:
        print(f"\nOK: {total_h:.0f} GPU-h is within the roadmap's 20-40h envelope.")

    json.dump({
        "gpu": gpu, "total_vram_gb": total_vram,
        "load_s": load_s, "train_s": train_s, "steps": steps,
        "peak_alloc_gb": peak_alloc, "peak_reserved_gb": peak_resv,
        "n_examples": len(texts), "seq_len": args.seq_len,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_4bit": args.load_4bit,
        "projected_pool_hours": total_h,
    }, open("timing_result.json", "w"), indent=2)
    print("\nwrote timing_result.json -- commit this, it is your compute-budget evidence")


if __name__ == "__main__":
    main()