"""CPU checks of the two model-dependent helpers in the notebook against independent references.

1. bertscore_f1 must match the bert_score package (same model, layer and baseline). A small RoBERTa
   (distilroberta-base) is used so that the test runs on a CPU; the code path is the same as for
   roberta-large in the notebook.
2. preference_scores must match a direct log-probability computation, and must drop pairs whose
   prompt alone fills max_length.

Usage:  python tests/check_models.py llama_sft_dpo.ipynb
Needs:  torch, transformers 4.x and bert-score 0.3.13 (bert_score does not work with transformers 5).
        Only small models are downloaded; no GPU is needed.
"""
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

nb = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
src = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("%%writefile"))
tmp = Path(tempfile.mkdtemp())
(tmp / "sft_dpo.py").write_text(src.split("\n", 1)[1], encoding="utf-8")
spec = importlib.util.spec_from_file_location("sft_dpo", tmp / "sft_dpo.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# ---- 1. BERTScore against the bert_score package
import bert_score  # noqa: E402

MODEL, LAYER = "distilroberta-base", 5          # bert_score's default layer for this model
base_file = Path(bert_score.__file__).parent / "rescale_baseline/en" / f"{MODEL}.tsv"
baseline_f = float(pd.read_csv(base_file).iloc[LAYER]["F"])
cands = ["The cat sat on the mat.", "A completely unrelated sentence about stock markets.",
         "Paris is the capital of France.", "", "Water boils at 100 degrees Celsius at sea level."]
refs = ["A cat was sitting on the mat.", "The weather was sunny all week.",
        "The capital of France is Paris.", "Some reference text.", "At sea level, water boils at 100 C."]
_, _, ref_f = bert_score.score(cands, refs, model_type=MODEL, num_layers=LAYER, lang="en",
                               rescale_with_baseline=True, device="cpu", verbose=False)
mine = m.bertscore_f1(cands, refs, MODEL, LAYER, baseline_f, batch_size=2, device="cpu")
for a, b in zip(mine, ref_f.tolist()):
    print(f"  mine {a: .9f}   bert_score {b: .9f}   difference {abs(a - b):.1e}")
assert all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(mine, ref_f.tolist())), "BERTScore mismatch"
print("bertscore_f1 matches bert_score")

# ---- 2. preference_scores against a direct computation
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
tok = AutoTokenizer.from_pretrained(TINY)
tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] | trim }}</s>{% endfor %}"
                     "{% if add_generation_prompt %}<|assistant|>{% endif %}")
lm = AutoModelForCausalLM.from_pretrained(TINY).eval()
pairs = [
    {"prompt": [{"role": "user", "content": "Is the sky blue?"}],
     "chosen": [{"role": "assistant", "content": "Yes, on a clear day."}],
     "rejected": [{"role": "assistant", "content": "No idea, whatever."}]},
    {"prompt": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "word " * 300}],
     "chosen": [{"role": "assistant", "content": "ok"}], "rejected": [{"role": "assistant", "content": "no"}]},
]
MAXLEN = 64
df = m.preference_scores(lm, tok, pairs, MAXLEN, device="cpu")
assert list(df["pair"]) == [0], "the pair whose prompt fills max_length must be dropped"


def direct(ex, key):
    p = tok(tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True),
            add_special_tokens=False)["input_ids"]
    full = tok(tok.apply_chat_template(ex["prompt"] + ex[key], tokenize=False), add_special_tokens=False)["input_ids"]
    assert full[:len(p)] == p
    ids = torch.tensor([full[:MAXLEN]])
    with torch.no_grad():
        logits = lm(input_ids=ids).logits[0]
    lp = [torch.log_softmax(logits[t - 1].float(), -1)[ids[0, t]].item() for t in range(len(p), ids.shape[1])]
    return sum(lp) / len(lp), len(lp)


for key in ("chosen", "rejected"):
    want, n = direct(pairs[0], key)
    got = df.loc[0, f"{key}_logp"]
    print(f"  {key}: mine {got:.6f}  direct {want:.6f}  tokens {df.loc[0, f'{key}_tokens']}/{n}")
    assert math.isclose(got, want, abs_tol=1e-5) and df.loc[0, f"{key}_tokens"] == n
assert df.loc[0, "prefers_chosen"] == (df.loc[0, "chosen_logp"] > df.loc[0, "rejected_logp"])
print("preference_scores matches the direct computation")
print("ALL MODEL CHECKS PASSED")
