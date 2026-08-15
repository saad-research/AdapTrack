# Setup handoff — Aug 1, 2026

Paste into `roadmap.md` and `venues.md`. Everything below was verified during
setup; items marked **UNRESOLVED** were not.

---

## 1. Merge semantics — CLOSED

`peft.add_weighted_adapter` does **not** produce the task-arithmetic delta.

- `linear`: applied delta = `Σᵢλᵢsᵢ BᵢAᵢ + Σ_{i≠j}√(λᵢsᵢλⱼsⱼ)·BᵢAⱼ`. The
  diagonal is correct (merged adapter gets `lora_alpha = r`, so `s_merged = 1`);
  the whole error is spurious cross terms between factors of *different*
  adapters. Measured: rel divergence 0.9994, cos 0.7107, residual after
  subtracting predicted cross terms 9.38e-08, best-fit scalar c = 1.0105.
- `svd` on real TinyLlama adapters (88 modules): rel 1.0000, cos 0.7643,
  c = 1.1691, cross-term residual 1.41 — a *different* failure. The svd path
  multiplies weights by `target.scaling` and applies them to
  `get_delta_weight()`, which already includes scaling: α/r applied twice, then
  rank-truncated to `max(rᵢ)` when the true sum of two rank-8 deltas is rank ≤16.
- `ties`/`dare_*` are worse than wrong, they are ill-posed: peft applies sign
  consensus and magnitude pruning to the A and B factors separately, but the
  LoRA factorization is only defined up to `A→QA, B→BQ⁻¹`. Elementwise ops on
  factors are gauge-dependent.

**Decision:** materialize `Δ = Σᵢ λᵢ(αᵢ/rᵢ)BᵢAᵢ` in `src/merge/delta.py` and
apply all merge operators to Δ. `delta.py` ships a test asserting (a) our delta
equals the hand formula exactly (max err 0.00e+00) and (b) peft equals ours plus
the cross terms (max err 1.39e-09), plus a gauge-invariance test. Re-run
`verify_merge_semantics.py` after any peft upgrade.

Gotcha: `ties`/`dare_*`/`magnitude_prune` crash in peft unless `density=` is
passed explicitly (peft's default is `None`).

---

## 2. Final dataset list

| Family | Dataset | Split | Rows | License | Status |
|---|---|---|---|---|---|
| A: summarization | `abisee/cnn_dailymail` (3.0.0) | train | 287,113 | Apache-2.0 | ✅ parquet |
| B: medical QA | `openlifescienceai/medmcqa` | train | 182,822 | Apache-2.0 | ✅ parquet |
| B′: medical QA (transfer) | `qiaojin/PubMedQA` `pqa_artificial` | train | 211,269 | MIT | ✅ parquet |
| C: legal | `coastalcph/lex_glue` `ledgar` | train | ~60,000 **verify** | CC-BY-4.0 | ✅ parquet |

**Dropped.** `qiaojin/PubMedQA` `pqa_labeled` — only 1,000 rows, unusable for 20
adapters. Use `pqa_artificial` (211,269). Note PQA-A labels are 92.8% yes /
7.2% no / 0% maybe, and contexts average ~238 tokens — train on
`question + long_answer` only, no context.

**Dropped.** `takala/financial_phrasebank` — `main` still carries
`financial_phrasebank.py`; hard-fails under `datasets>=4.0`, and
`refs/convert/parquet` does not exist. Even `sentences_50agree` is only 4,846
rows, so the overlap problem was never fixed by the allagree→50agree switch.

**Rejected as replacement.** `FinGPT/fingpt-sentiment-train` (76,772 rows,
parquet, DOI 10.57967/hf/3856) — **no license declared on the dataset card**,
and it is an aggregation of existing financial sentiment corpora (very likely
including financial_phrasebank itself), so "distinct dataset" claims are unsafe.
Label balance across its 9 fine-grained classes was **not verified**.

**Confirmed sizes:** `lex_glue` `case_hold` train = 45,000 (verified);
`ecthr_a` train = 9,000 (verified); `ledgar` train size **UNRESOLVED** — the
LexGLUE paper reports 60k but I did not see it in the dataset metadata. Confirm
on load; if <60,000, use 2,500-example subsets or switch to `case_hold`.

**Subset discipline:** 20 adapters × 3,000 examples = 60,000 rows needed per
family for fully disjoint subsets. cnn_dailymail, medmcqa, and pqa_artificial
all clear this with large margin. `ledgar` is exactly at the line.

---

## 3. Cross-dataset generalization test — REDESIGNED

MedMCQA (4-option exam MCQ) vs PubMedQA PQA-A (yes/no from abstracts) differ in
*format* as much as in *content*, so the test as written is format-confounded:
a transfer failure would be unattributable.

**Fix:** render both through one shared prompt template so format is held
constant and only domain content varies. Add a **format-matched negative
control**: LEDGAR rendered in the *same* template. If the classifier still fires
on the negative control, it is detecting template structure, not domain.

The test is then a 2×2: {medical, legal} × {dataset 1, dataset 2}, all in one
template. Transfer within medical + no transfer to legal = task provenance.
Transfer everywhere = format artifact. Transfer nowhere = dataset fingerprinting.

---

## 4. Compute — dtype and padding were both wrong

**bf16 has no tensor-core path on a T4.** bf16 tensor cores arrived with Ampere
(sm_80); Turing is sm_75. `torch.cuda.is_bf16_supported()` returns True on
Turing anyway — it reports whether bf16 can be *allocated*, not accelerated;
TorchInductor gates bf16 at sm_80+. **Use fp16 on T4.** With fp16, promote LoRA
params to fp32 after `get_peft_model`, or Trainer raises "Attempting to unscale
FP16 gradients".

**Padding was dominating.** v1 used `padding="max_length"`. Measured 439 tok/s
at seq 512 and 467 tok/s at seq 256 — flat, the signature of cost tracking
*padded* tokens. MedMCQA questions average ~13 tokens. v2 uses dynamic padding
to a multiple of 8 plus `group_by_length`.

**Projection arithmetic was wrong** in v1 (multiplied the measured run by 80
regardless of `--n-examples`; the 500-example run projected 6.2 GPU-h against a
real ~36). Fixed, and base-model load is now amortized once across the pool
rather than charged 80 times.

Baseline measurements to beat (T4, medmcqa, bs4×ga4, r=8 α=16 q/k/v/o,
2,252,800 trainable params):

| config | s/step | wall | peak reserved | pool @80 |
|---|---|---|---|---|
| 3000 ex, seq 512, bf16, static pad | 18.60 | 58m16s | 10.88 GB | 77.7 h |
| 500 ex, seq 256, bf16, static pad | 8.57 | 4m34s | 5.82 GB | (~36 h scaled) |

---

## 5. Detector availability — Arabic / Urdu / Bengali

**No PII-redaction system covers any of the three.** Verified:

- `urchade/gliner_multi_pii-v1` — 6 languages: en, fr, de, es, pt, it.
- `fastino/gliner2-privacy-filter-PII-multi` (GLiNER2-PII, arXiv:2605.09973,
  Zaratiana et al., May 2026) — 7 languages: en, fr, es, de, it, pt, nl. All
  Latin-script European. Span-F1 0.477 on SPY even in-scope. (Card says 205M
  params, press release says 300M — cite the paper.)
- Presidio ships a real `GLiNERRecognizer` (PR #1916, `gliner` extra confirmed
  in wheel metadata), but its default config is `supported_languages: [en]`,
  Presidio's docs state each recognizer supports one language, and the only
  wired model is the 6-Latin-language one. You can force a `ur`/`bn` tag; it
  runs, with no training signal behind it and no spaCy tokenizer for those
  scripts. **Report as an architectural finding, not a baseline.**

What *does* exist — all generic PER/LOC/ORG NER, no notion of name components:

| Script | Tool | Notes |
|---|---|---|
| Arabic | CAMeL Tools `ner-arabert` | 4 classes; `camel_data -i ner-arabert` |
| Urdu | Stanza `ur`, package `ilner` | added in Stanza 1.10.0, from `ltrc/IL-NER` |
| Bengali | `ai4bharat/IndicNER` (MIT) | **gated — request access early**; not in Stanza |

No single system covers both Urdu and Bengali: Stanza has Urdu but not Bengali;
IndicNER has Bengali but not Urdu.

---

## 6. IsmBench central claim — REWRITTEN

Component-level PRR is now the **main result**, not the Gate B1 fallback. The
gap is absence, not degradation, and PRR is the only metric that can measure it:
entity-level recall scores a partial redaction as a success.

**Claim:**
> No production PII-redaction system supports Arabic, Urdu, or Bengali personal
> names, and the generic named-entity models that do cover these scripts
> recognize a name as a single opaque span — so compositional names are
> routinely *partially* redacted, leaving residual identifying components that
> entity-level recall scores as a success.

**Abstract opening:**
> Personal-name redaction is evaluated as a binary: an entity is either caught
> or missed. We show this framing conceals a systematic failure on names built
> compositionally from multiple components — given name, patronymic chain,
> teknonym, epithet, and origin descriptor — where redaction systems catch some
> components and silently leave others, producing outputs that score as
> successful redactions while remaining re-identifiable.

Leads with the metric critique rather than the coverage gap, which travels
better at MPLR-FM and InfPriv topic 06. Avoids "Muslim names" in the core claim:
the property is compositional naming; the representation-harm framing layers on
for MusIML Track 1 without the claim depending on it. The residual-k
re-identification proxy carries more weight now — it converts "partial
redaction" into a quantified privacy failure and is what makes the InfPriv hook
land.

---

## 7. Environments

`transformers==4.57.6` in **both** envs. Not conservatism:
`presidio-analyzer[transformers]` pulls `spacy_huggingface_pipelines`, which
pins `transformers<5.0.0`; `gliner 0.2.28` requires `>=4.51.3,<5.14.0`. 4.57.6
is the only version satisfying both. Verified functionally with peft 0.20.0,
not just resolvable.

Resolver-verified conflicts:
- **mergekit 0.1.4 pins `accelerate~=1.6.0`** — incompatible with modern
  accelerate. Dropped; not needed (LoRA-level merges, deltas materialized by hand).
- **camel-tools 1.6.0 requires `dill>=0.4.0`; datasets <4.8 caps `dill<0.3.9`.**
  Forces `datasets>=4.8.5`.
- camel-tools 1.6.0 requires Python >=3.11, numpy>=2.0, scikit-learn>=1.8.
- presidio-analyzer requires Python >=3.10,<3.15.

Two separate envs (`requirements-adaptrack.txt`, `requirements-ismbench.txt`) so
a spaCy or camel-tools pin can never break GPU runs during the Aug 12–20 crunch.

### torchao gotcha
Colab preinstalls **torchao 0.10.0, which breaks peft 0.20.0 with an
ImportError**. Run `pip uninstall -y torchao` at the top of **every** session.
Put it in the first cell of every notebook, not in a setup script you might skip.

---

## 8. LaTeX

`neurips_2026.sty` verified by compiling (`\ProvidesPackage[2026-01-29]`,
"last revision: January 2026"). **Provenance caveat: obtained from a
third-party GitHub mirror, not neurips.cc — re-download from the official site
before submitting.**

- Bare `\usepackage{neurips_2026}` → "Anonymous Author(s)" ✅, line numbers on,
  footer "Submitted to 40th Conference … Do not distribute." Correct for
  submission; `venues.md` §7 confirmed.
- **New:** the sty has a `dblblindworkshop` option and a `\workshoptitle{}`
  command. The workshop name renders only in `[final]`. Camera-ready incantation
  is `\usepackage[final,dblblindworkshop]{neurips_2026}` — verified to produce
  "…(NeurIPS 2026). Workshop: <name>."
- The sty auto-hides its `\ack` environment in submission mode. Don't rely on
  it; write no acknowledgments until accepted.
- `\@neuripslocation` is hardcoded to **Sydney**. Not printed in submission
  mode. If the city-fork lands on Paris, check the camera-ready footer.
- `checklist.tex` inputs cleanly (adds 8 pages). InfPriv states the checklist is
  not required — keep the file, `\input` only where required.

### Engine decision for non-Latin scripts
**pdflatex cannot typeset Arabic, Urdu Nastaliq, or Bengali.** IsmBench must
either (a) build with **XeLaTeX or LuaLaTeX** + `fontspec` + Noto Nastaliq Urdu
/ Noto Sans Bengali / Amiri, or (b) stay on pdflatex with all names as
transliterations and native-script forms as a **figure image**. Decide before
writing examples. Recommendation: (b) for the 4-page MPLR-FM/InfPriv versions
(one figure, no engine risk near a deadline), (a) only for the MusIML 8-page
version where script fidelity is part of the contribution.

---

## 9. Still UNRESOLVED

- `lex_glue` `ledgar` exact train row count — confirm on load (need ≥60,000).
- `FinGPT/fingpt-sentiment-train` license and label balance — unverified; not used.
- Colab CU consumption rates per GPU — undocumented and changing. Read the UI
  counter before/after a run; do not trust published figures.
- arXiv IDs `2601.22285`, `2601.11207`, `2511.06390` — still never independently
  opened. Citation-verification block, before any citing.
- CODEC-FoMo page limits; InfPriv public CFP; MusIML Sydney deadline/limits.
- Open emails: LIGHT page limit, MusIML deadline schedule, MPLR-FM scope
  (Version A vs B — decides the Paris paper assignment), CODEC-FoMo scope.
- MedMCQA `test` split label reliability — use train + validation only until
  confirmed.