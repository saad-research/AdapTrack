#!/usr/bin/env python
"""
verify_merge_semantics.py  --  AdapTrack week-one gate.

Question: does peft's `add_weighted_adapter(combination_type=...)` produce the
task-arithmetic delta

      D_TA = sum_i  lambda_i * (alpha_i / r_i) * (B_i @ A_i)

or something else? Anything else means the spectral features in AdapTrack are
computed on the wrong object.

What this prints, per target module:
  ||D_TA||_F            direct task-arithmetic delta (the object you want)
  ||D_peft||_F          what peft actually applies at inference
  ||D_peft - D_TA||_F   absolute divergence
  rel                   divergence / ||D_TA||_F
  cos                   cosine similarity of the two flattened deltas
  ||D_cross||_F         predicted spurious cross-term mass (see note below)

Prediction being tested (peft 0.20.0, combination_type="linear"): peft scales
each *factor* by sqrt(lambda_i * s_i) and sums the factors separately, so the
applied delta is

      D_peft = s_new * (sum_i sqrt(l_i s_i) B_i)(sum_j sqrt(l_j s_j) A_j)
             = sum_i l_i s_i B_i A_i   +   sum_{i != j} sqrt(l_i s_i l_j s_j) B_i A_j
             = D_TA + D_cross

`add_weighted_adapter` sets the merged adapter's lora_alpha = r, so s_new = 1
and the diagonal terms come out correct; the entire error is the cross terms
B_i A_j for i != j, which are products of factors from *different* adapters and
have no meaning. If ||D_cross|| is comparable to ||D_TA||, materialize the delta
yourself.

Usage
-----
  # day one, no training needed, no GPU, no downloads:
  python verify_merge_semantics.py --tiny-random

  # the real check, once you have two trained adapters:
  python verify_merge_semantics.py \
      --base TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --adapter-a runs/adapter_pubmed_seed0 \
      --adapter-b runs/adapter_cnndm_seed0 \
      --lambdas 0.5 0.5 --combination-type linear

Run it in float32. fp16 rounding noise is ~1e-3 relative and would muddy the
signal you are trying to measure.
"""

import argparse
import math

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_model(args):
    """Return a PeftModel holding two adapters named 'a' and 'b'."""
    if args.tiny_random:
        # Tiny randomly-initialised Llama. No network, no checkpoint. Use this to
        # confirm the *arithmetic*; the divergence is a property of peft's merge
        # code, not of the weights, so a small model settles the question.
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=256,
        )
        torch.manual_seed(0)
        base = LlamaForCausalLM(cfg).to(torch.float32)
    else:
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.float32, device_map="cpu"
        )

    lcfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        target_modules=TARGETS,
        task_type="CAUSAL_LM",
    )

    if args.adapter_a and args.adapter_b:
        model = PeftModel.from_pretrained(base, args.adapter_a, adapter_name="a")
        model.load_adapter(args.adapter_b, adapter_name="b")
    else:
        model = get_peft_model(base, lcfg, adapter_name="a")
        model.add_adapter("b", lcfg)
        # LoRA initialises B to zeros, so an untrained adapter has delta exactly 0
        # and every comparison below would be trivially 0 == 0. Fill B with noise
        # to stand in for a trained adapter.
        torch.manual_seed(1)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoraLayer):
                    for name in ("a", "b"):
                        if name in module.lora_B:
                            module.lora_B[name].weight.normal_(0, 0.02)
                            module.lora_A[name].weight.normal_(0, 0.02)
    return model


def lora_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and "a" in module.lora_A:
            yield name, module


def deltas_for_layer(module, adapters, lambdas, merged_name):
    """Return (D_TA, D_peft, D_cross) for one LoRA-targeted module."""
    A = {n: module.lora_A[n].weight.data.double() for n in adapters}
    B = {n: module.lora_B[n].weight.data.double() for n in adapters}
    s = {n: module.scaling[n] for n in adapters}

    # The object AdapTrack's threat model is about: sum_i lambda_i * s_i * B_i A_i
    d_ta = sum(lam * s[n] * (B[n] @ A[n]) for n, lam in zip(adapters, lambdas))

    # What peft actually applies. get_delta_weight returns B @ A * scaling for
    # the merged adapter, i.e. exactly what is added to the frozen base weight.
    d_peft = module.get_delta_weight(merged_name).data.double()

    # Predicted spurious mass: the i != j outer products.
    w = {n: math.sqrt(abs(lam * s[n])) for n, lam in zip(adapters, lambdas)}
    d_cross = torch.zeros_like(d_ta)
    for i in adapters:
        for j in adapters:
            if i != j:
                d_cross += w[i] * w[j] * (B[i] @ A[j])

    return d_ta, d_peft, d_cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter-a")
    ap.add_argument("--adapter-b")
    ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lambdas", type=float, nargs=2, default=[0.5, 0.5])
    ap.add_argument(
        "--combination-type",
        default="linear",
        help="linear | ties | dare_linear | dare_ties | svd | ties_svd | dare_linear_svd",
    )
    ap.add_argument(
        "--density",
        type=float,
        default=0.5,
        help="required by ties/dare_*/magnitude_prune; peft's default is None and crashes",
    )
    ap.add_argument("--max-layers", type=int, default=8, help="rows to print")
    args = ap.parse_args()

    adapters = ["a", "b"]
    merged = "merged"

    model = build_model(args)
    model.add_weighted_adapter(
        adapters=adapters,
        weights=list(args.lambdas),
        adapter_name=merged,
        combination_type=args.combination_type,
        density=args.density,
    )

    mcfg = model.peft_config[merged]
    print(f"\ncombination_type = {args.combination_type}   lambdas = {args.lambdas}")
    print(f"per-adapter scaling s_i = alpha/r = {args.alpha}/{args.r} = {args.alpha / args.r}")
    print(f"merged adapter config: r={mcfg.r}, lora_alpha={mcfg.lora_alpha}  "
          f"-> s_merged = {mcfg.lora_alpha / mcfg.r}")

    hdr = f"\n{'module':<44} {'||D_TA||':>10} {'||D_peft||':>11} {'||diff||':>10} {'rel':>8} {'cos':>7} {'||cross||':>10}"
    print(hdr)
    print("-" * len(hdr))

    rels, coss, printed = [], [], 0
    cross_explains, best_scales, rels_after_scale = [], [], []
    for name, module in lora_layers(model):
        if merged not in module.lora_A:
            continue
        d_ta, d_peft, d_cross = deltas_for_layer(module, adapters, args.lambdas, merged)

        n_ta = d_ta.norm().item()
        diff = (d_peft - d_ta)
        rel = diff.norm().item() / max(n_ta, 1e-12)
        cos = torch.nn.functional.cosine_similarity(
            d_peft.flatten(), d_ta.flatten(), dim=0
        ).item()
        rels.append(rel)
        coss.append(cos)
        # is the divergence fully explained by the predicted cross terms?
        cross_explains.append((diff - d_cross).norm().item() / max(n_ta, 1e-12))

        # Separate "wrong scalar factor" from "wrong subspace": find the c that
        # best matches D_peft ~ c * D_TA, then see what error survives.
        c = (d_peft.flatten() @ d_ta.flatten()).item() / max(n_ta**2, 1e-12)
        best_scales.append(c)
        rels_after_scale.append((d_peft - c * d_ta).norm().item() / max(n_ta, 1e-12))

        if printed < args.max_layers:
            short = name.replace("base_model.model.model.layers.", "L")
            print(f"{short:<44} {n_ta:>10.4f} {d_peft.norm().item():>11.4f} "
                  f"{diff.norm().item():>10.4f} {rel:>8.3f} {cos:>7.3f} "
                  f"{d_cross.norm().item():>10.4f}")
            printed += 1

    n = len(rels)
    print(f"\n{n} LoRA-targeted modules compared.")
    print(f"mean relative divergence ||D_peft - D_TA|| / ||D_TA|| : {sum(rels)/n:.4f}")
    print(f"mean cosine(D_peft, D_TA)                             : {sum(coss)/n:.4f}")
    print(f"mean residual after subtracting predicted cross terms : {sum(cross_explains)/n:.2e}")
    print(f"best-fit scalar c in D_peft ~ c * D_TA                : {sum(best_scales)/n:.4f}")
    print(f"mean relative residual after rescaling by c           : {sum(rels_after_scale)/n:.4f}")
    print("  (c != 1 means a wrong scale factor; residual after rescaling means a")
    print("   genuinely different matrix, not just a differently-scaled one.)")

    tol = 1e-6
    if sum(rels) / n < tol:
        print("\nVERDICT: peft matches task arithmetic for this combination_type.")
        print("You may extract spectral features straight from the peft merge.")
    else:
        print("\nVERDICT: peft does NOT produce the task-arithmetic delta.")
        if sum(cross_explains) / n < 1e-6:
            print("The divergence is fully accounted for by cross terms B_i A_j (i != j):")
            print("peft sums the A factors and the B factors separately, so the product")
            print("picks up outer products between factors of *different* adapters.")
        print("Materialize D = sum_i lambda_i * (alpha_i/r_i) * (B_i @ A_i) yourself")
        print("and extract spectral features from that.")


if __name__ == "__main__":
    main()