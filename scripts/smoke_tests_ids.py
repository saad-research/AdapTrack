#!/usr/bin/env python
"""
smoke_test_ids.py -- load every ID in the plan, print what came back, keep going.

Run this FIRST, on Aug 1, before anything else. Each check is isolated: one
failure never stops the rest, so a single run tells you the full damage.

  python smoke_test_ids.py                 # everything
  python smoke_test_ids.py --only datasets # or: models, nlp
  python smoke_test_ids.py --skip-heavy    # skips the two multi-GB downloads

NOTE ON PROVENANCE: I verified each ID against its Hugging Face page, but I could
not execute this script against the Hub (no network egress to huggingface.co from
where it was written). Treat the pass/fail column as a prediction until you run it.
Everything except financial_phrasebank is predicted to pass.
"""

import argparse
import sys
import traceback

RESULTS = []


def check(name, fn, heavy=False):
    """Run one probe, record outcome, never raise."""
    print(f"\n{'=' * 72}\n### {name}\n{'=' * 72}")
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail or ""))
        print(f"--> PASS  {detail or ''}")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        RESULTS.append((name, "FAIL", msg[:300]))
        print(f"--> FAIL  {msg[:300]}")
        if "--traceback" in sys.argv:
            traceback.print_exc()


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

def _describe(ds, text_fields, n=1):
    """Print split sizes, features, and a truncated sample row."""
    from datasets import DatasetDict

    if isinstance(ds, DatasetDict):
        sizes = {k: len(v) for k, v in ds.items()}
        first = ds[list(ds.keys())[0]]
    else:
        sizes = {"(single split)": len(ds)}
        first = ds
    print("splits    :", sizes)
    print("features  :", list(first.features.keys()))
    row = first[0]
    for k in (text_fields or list(row.keys())[:3]):
        if k in row:
            v = str(row[k]).replace("\n", " ")
            print(f"  {k:<14}: {v[:160]}{'...' if len(v) > 160 else ''}")
    return f"splits={sizes}"


def ds_cnn_dailymail():
    from datasets import load_dataset
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
    return _describe(ds, ["article", "highlights"])


def ds_pubmedqa():
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    return _describe(ds, ["question", "long_answer", "final_decision"])


def ds_medmcqa():
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa")
    # NOTE: use train + validation. The test split is the AIIMS holdout and its
    # labels are not reliable for supervised use -- confirm before touching it.
    return _describe(ds, ["question", "opa", "cop", "subject_name"])


def ds_financial_phrasebank():
    """EXPECTED TO FAIL. main still carries financial_phrasebank.py."""
    from datasets import load_dataset
    ds = load_dataset("takala/financial_phrasebank", "sentences_50agree")
    return _describe(ds, ["sentence", "label"])


def ds_financial_phrasebank_parquet_fallback():
    """Workaround A: read the auto-converted parquet branch directly."""
    from datasets import load_dataset
    ds = load_dataset(
        "takala/financial_phrasebank",
        "sentences_50agree",
        revision="refs/convert/parquet",
    )
    return _describe(ds, ["sentence", "label"])


def ds_replacement_candidates():
    """Workaround B: a corpus actually large enough for 20 disjoint subsets.

    financial_phrasebank sentences_50agree is 4,846 rows. Twenty adapters drawing
    2-4k rows each would overlap almost completely -- the exact failure your own
    roadmap note warns about ("classifier may learn these sentences, not this
    task"). Switching allagree->50agree does NOT fix that. These are bigger.
    I have NOT verified these IDs; this probe is how you check them.
    """
    from datasets import load_dataset
    out = []
    for cand in [
        ("zeroshot/twitter-financial-news-sentiment", None),
        ("FinGPT/fingpt-sentiment-train", None),
    ]:
        try:
            ds = load_dataset(cand[0], cand[1]) if cand[1] else load_dataset(cand[0])
            n = sum(len(v) for v in ds.values()) if hasattr(ds, "values") else len(ds)
            print(f"  {cand[0]}: OK, {n} rows total")
            out.append(f"{cand[0]}={n}")
        except Exception as e:
            print(f"  {cand[0]}: unavailable ({type(e).__name__})")
    if not out:
        raise RuntimeError("no replacement candidate loaded")
    return "; ".join(out)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def _describe_model(model_id, heavy=True):
    from transformers import AutoConfig, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    print("arch      :", getattr(cfg, "architectures", None))
    print("hidden    :", getattr(cfg, "hidden_size", None),
          "| layers:", getattr(cfg, "num_hidden_layers", None),
          "| heads:", getattr(cfg, "num_attention_heads", None),
          "| kv heads:", getattr(cfg, "num_key_value_heads", None))
    print("vocab     :", getattr(cfg, "vocab_size", None),
          "| max_pos:", getattr(cfg, "max_position_embeddings", None))
    print("tokenizer :", type(tok).__name__, "| pad:", tok.pad_token, "| eos:", tok.eos_token)
    if heavy:
        import torch
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        n = sum(p.numel() for p in m.parameters())
        print(f"params    : {n / 1e9:.3f}B  (~{n * 2 / 1e9:.2f} GB in fp16)")
        del m
        return f"{n / 1e9:.3f}B params"
    return "config+tokenizer only"


def model_tinyllama(heavy):
    return _describe_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", heavy)


def model_qwen(heavy):
    # Replaces Llama-3.2-1B specifically to avoid the Meta license gate.
    return _describe_model("Qwen/Qwen2.5-1.5B", heavy)


def model_indicner():
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    mid = "ai4bharat/IndicNER"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForTokenClassification.from_pretrained(mid)
    print("labels    :", mdl.config.id2label)
    nlp = pipeline("ner", model=mdl, tokenizer=tok, aggregation_strategy="simple")
    # Bengali: "Muhammad Abdullah lives in Dhaka."
    sample = "মুহাম্মদ আব্দুল্লাহ ঢাকায় থাকেন।"
    print("sample    :", sample)
    print("entities  :", nlp(sample))
    return f"{len(mdl.config.id2label)} labels"


# --------------------------------------------------------------------------
# nlp toolchains
# --------------------------------------------------------------------------

def nlp_stanza_urdu():
    """Stanza gained Urdu NER in 1.10.0 (package 'ilner', from ltrc/IL-NER)."""
    import stanza
    stanza.download("ur", processors="tokenize,ner", verbose=False)
    nlp = stanza.Pipeline("ur", processors="tokenize,ner", verbose=False)
    # "Muhammad Abdullah lives in Karachi."
    doc = nlp("محمد عبداللہ کراچی میں رہتے ہیں۔")
    ents = [(e.text, e.type) for s in doc.sentences for e in s.ents]
    print("entities  :", ents)
    return f"{len(ents)} entities"


def nlp_stanza_bengali_expected_fail():
    """EXPECTED TO FAIL: Stanza has no pretrained Bengali NER model."""
    import stanza
    stanza.download("bn", processors="tokenize,ner", verbose=False)
    nlp = stanza.Pipeline("bn", processors="tokenize,ner", verbose=False)
    doc = nlp("মুহাম্মদ আব্দুল্লাহ ঢাকায় থাকেন।")
    return f"{sum(len(s.ents) for s in doc.sentences)} entities"


def nlp_camel_arabic():
    """Requires `camel_data -i ner-arabert` to have been run first."""
    from camel_tools.ner import NERecognizer
    from camel_tools.tokenizers.word import simple_word_tokenize
    ner = NERecognizer.pretrained()
    # "Muhammad Abdullah bin Ahmad lives in Cairo."
    sent = simple_word_tokenize("محمد عبد الله بن أحمد يعيش في القاهرة")
    labels = ner.predict_sentence(sent)
    print("tokens    :", sent)
    print("labels    :", labels)
    return f"{sum(1 for l in labels if l != 'O')} tagged tokens"


def nlp_presidio():
    from presidio_analyzer import AnalyzerEngine
    eng = AnalyzerEngine()
    res = eng.analyze(text="Contact Muhammad Abdullah at m.abdullah@example.com",
                      language="en")
    print("results   :", [(r.entity_type, r.score) for r in res])
    return f"{len(res)} entities"


def nlp_gliner():
    from gliner import GLiNER
    m = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
    labels = ["person", "email", "address"]
    txt = "Contact Muhammad Abdullah bin Ahmad al-Ansari at m.a@example.com"
    print("entities  :", m.predict_entities(txt, labels))
    # Same sentence in Arabic script -- this model was trained on 6 Latin-script
    # European languages, so treat any output here as unsupported zero-shot.
    print("arabic    :", m.predict_entities("محمد عبد الله بن أحمد الأنصاري", labels))
    return "loaded"


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["datasets", "models", "nlp"])
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip full weight downloads (TinyLlama, Qwen)")
    ap.add_argument("--traceback", action="store_true")
    args = ap.parse_args()

    if args.only in (None, "datasets"):
        check("dataset: abisee/cnn_dailymail [3.0.0]", ds_cnn_dailymail)
        check("dataset: qiaojin/PubMedQA [pqa_labeled]", ds_pubmedqa)
        check("dataset: openlifescienceai/medmcqa", ds_medmcqa)
        check("dataset: takala/financial_phrasebank [sentences_50agree] (EXPECTED FAIL)",
              ds_financial_phrasebank)
        check("  fallback: financial_phrasebank @ refs/convert/parquet",
              ds_financial_phrasebank_parquet_fallback)
        check("  replacement candidates (larger finance corpora)",
              ds_replacement_candidates)

    if args.only in (None, "models"):
        heavy = not args.skip_heavy
        check("model: TinyLlama/TinyLlama-1.1B-Chat-v1.0", lambda: model_tinyllama(heavy))
        check("model: Qwen/Qwen2.5-1.5B", lambda: model_qwen(heavy))
        check("model: ai4bharat/IndicNER (Bengali)", model_indicner)

    if args.only in (None, "nlp"):
        check("stanza: Urdu NER (ilner)", nlp_stanza_urdu)
        check("stanza: Bengali NER (EXPECTED FAIL)", nlp_stanza_bengali_expected_fail)
        check("camel-tools: Arabic NER (ner-arabert)", nlp_camel_arabic)
        check("presidio: default English engine", nlp_presidio)
        check("gliner: urchade/gliner_multi_pii-v1", nlp_gliner)

    print(f"\n\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    w = max(len(n) for n, _, _ in RESULTS)
    for name, status, detail in RESULTS:
        print(f"{status:<5} {name:<{w}}  {detail[:70]}")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"\n{len(RESULTS) - n_fail} passed, {n_fail} failed.")
    print("Two failures are expected by design: financial_phrasebank and Stanza bn.")


if __name__ == "__main__":
    main()