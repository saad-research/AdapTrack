"""
src/train/adapter.py -- train exactly one LoRA adapter.

Locked config from the A/B: fp16 on sm_75 (no bf16 tensor-core path on Turing),
dynamic padding to a multiple of 8, group_by_length, max_len as a truncation cap
only. That combination measured 0.67 s/step against 18.17 for the v1 defaults.
"""

from __future__ import annotations

import time

import torch


def pick_dtype(force: str | None = None):
    major, _ = torch.cuda.get_device_capability(0)
    if force == "fp32":
        return torch.float32, False, False
    if force == "fp16":
        return torch.float16, True, False
    if force == "bf16":
        return torch.bfloat16, False, True
    if major >= 8:
        return torch.bfloat16, False, True
    return torch.float16, True, False   # Turing/Volta: bf16 is not accelerated


def load_base(base_id: str, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype).cuda()
    return model, tok


def train_one_adapter(base, tok, texts, out_dir, *, r=8, alpha=16, dropout=0.05,
                      lr=2e-4, epochs=1.0, batch_size=4, grad_accum=4,
                      max_len=512, seed=0, use_fp16=True, use_bf16=False,
                      save=True):
    """Wrap `base` in a fresh LoRA, train, save, and unwrap.

    Returns a dict of run stats. `base` is returned to its pre-call state via
    unload(), with an assertion that no LoRA layer survives -- state leaking
    between adapters would make pool adapters correlated and silently inflate
    every AUC downstream.
    """
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import (DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    torch.manual_seed(seed)
    model = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"), adapter_name="default")

    # fp16 + GradScaler raises "Attempting to unscale FP16 gradients" when the
    # trainable params are themselves fp16. Base stays fp16, LoRA goes fp32:
    # ~2.25M params, ~9 MB.
    if use_fp16:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, max_length=max_len),
        batched=True, remove_columns=["text"])
    lens = [len(x) for x in ds["input_ids"]]
    ds = ds.add_column("length", lens)

    args = TrainingArguments(
        output_dir=f"{out_dir}/_trainer", per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, num_train_epochs=epochs,
        learning_rate=lr, fp16=use_fp16, bf16=use_bf16, group_by_length=True,
        logging_steps=50, save_strategy="no", report_to=[], seed=seed,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(
            tok, mlm=False, pad_to_multiple_of=8))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train()
    torch.cuda.synchronize()
    wall = time.time() - t0

    if save:
        model.save_pretrained(out_dir)

    stats = dict(
        wall_s=wall, steps=out.global_step,
        s_per_step=wall / max(out.global_step, 1),
        train_loss=float(out.training_loss),
        peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
        n_examples=len(texts), tok_p50=int(sorted(lens)[len(lens) // 2]),
    )

    base_restored = model.unload()          # strips LoRA, keeps base weights
    del model, trainer
    leftover = [n for n, m in base_restored.named_modules()
                if isinstance(m, LoraLayer)]
    assert not leftover, f"LoRA layers survived unload(): {leftover[:3]}"
    torch.cuda.empty_cache()
    return stats, base_restored