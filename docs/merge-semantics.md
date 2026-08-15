# Why we materialize the delta ourselves
 
peft's `add_weighted_adapter(combination_type="linear")` scales each FACTOR by
sqrt(λᵢ·sᵢ) and sums the A factors and B factors separately. The applied delta
is therefore
 
    Δ_peft = Σᵢ λᵢsᵢBᵢAᵢ  +  Σ_{i≠j} sqrt(λᵢsᵢλⱼsⱼ)·BᵢAⱼ
 
The diagonal terms are correct (the merged adapter gets lora_alpha = r, so
s_merged = 1). The error is entirely the cross terms BᵢAⱼ, i≠j. Measured on a
2-adapter merge at λ=[0.5,0.5], r=8, α=16: relative divergence 0.999,
cos(Δ_peft, Δ_TA) = 0.711, residual after subtracting the predicted cross
terms 9.4e-08 (i.e. the identity is exact).
 
The `*_svd` variants have a different bug: they multiply weights by
target.scaling and then apply them to get_delta_weight(), which already
includes scaling -- a double application of α/r -- and then truncate to rank
max(rᵢ) when the true sum of two rank-8 deltas is rank ≤16.
 
TIES/DARE are worse than wrong, they are ill-posed: peft applies sign
consensus and magnitude pruning to the A and B factors separately, but the
LoRA factorization is only unique up to A→QA, B→BQ⁻¹. Elementwise operations
on factors are gauge-dependent. Apply them to Δ.
 
Re-run scripts/verify_merge_semantics.py after any peft upgrade.