"""Dataset sources: each turns a public dataset into benchmark requests.

A source is `fn(n, rng, tok, lengths, **kw) -> list[dict]`, registered with @source.
Each dict holds kind, group, messages, max_tokens, ignore_eos, labels, reference and
scorer; workload.build renders the messages into a prompt. `lengths` is "fixed"
(generate requests decode exactly max_tokens, ignore_eos=True) or "natural" (max_tokens
is only a cap). Classify requests always stop naturally.

Adding a dataset means adding one function here. huggingface_hub and pyarrow are
imported lazily so the rest of the package runs without them.
"""
import json
import math
import os
import re
import urllib.request
from pathlib import Path

SOURCES = {}
CACHE = Path(os.environ.get("BENCH_CACHE", Path.home() / ".cache" / "bench"))
LETTERS = "ABCD"


def source(name, license, url):
    def register(fn):
        fn.license, fn.url = license, url
        SOURCES[name] = fn
        return fn
    return register


def req(kind, messages, max_tokens, *, group=None, ignore_eos=False, labels=None, reference=None, scorer=None):
    return dict(kind=kind, group=group, messages=messages, max_tokens=max_tokens, ignore_eos=ignore_eos,
                labels=labels, reference=reference, scorer=scorer)


def gen_len(lengths, fixed, cap):
    """(max_tokens, ignore_eos) for a generate request."""
    return (fixed, True) if lengths == "fixed" else (cap, False)


def hf_table(repo, filename, columns=None):
    """One parquet file from a HF dataset repo; never downloads a whole config."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    return pq.read_table(hf_hub_download(repo, filename, repo_type="dataset"), columns=columns)


def hf_rows(repo, filename, columns=None):
    return hf_table(repo, filename, columns).to_pylist()


def url_jsonl(url):
    path = CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").rename(path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ntok(tok, text):
    return len(tok.encode(text, add_special_tokens=False))


def group_by(rows, key):
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return groups


def mc_question(question, options):
    return question.strip() + "\n" + "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options))


@source("mmlu", license="MIT", url="https://huggingface.co/datasets/cais/mmlu")
def mmlu(n, rng, tok, lengths, per_subject=24, shots=5):
    """5-shot multiple choice; the few-shot block is shared by every question of a subject."""
    test = group_by(hf_rows("cais/mmlu", "all/test-00000-of-00001.parquet"), "subject")
    dev = group_by(hf_rows("cais/mmlu", "all/dev-00000-of-00001.parquet"), "subject")
    subjects = rng.sample(sorted(test), min(len(test), math.ceil(n / per_subject)))
    out = []
    for subject in subjects:
        prefix = [{"role": "system", "content": f"The following are multiple choice questions about "
                   f"{subject.replace('_', ' ')}. Answer with the letter of the correct option only."}]
        for ex in dev[subject][:shots]:
            prefix += [{"role": "user", "content": mc_question(ex["question"], ex["choices"])},
                       {"role": "assistant", "content": LETTERS[ex["answer"]]}]
        for q in rng.sample(test[subject], min(per_subject, len(test[subject]))):
            out.append(req("classify", prefix + [{"role": "user", "content": mc_question(q["question"], q["choices"])}], 1,
                           group=f"mmlu/{subject}", labels=list(LETTERS), reference=LETTERS[q["answer"]],
                           scorer="choice"))
    return out[:n]


@source("clinc", license="CC-BY-3.0", url="https://huggingface.co/datasets/clinc/clinc_oos")
def clinc(n, rng, tok, lengths):
    """Intent classification: one long instruction listing all 151 intents, shared by every request."""
    table = hf_table("clinc/clinc_oos", "plus/test-00000-of-00001.parquet")
    names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    system = ("Classify the user's message into exactly one of the intents below. Reply with the intent name only, "
              "exactly as written. Use oos if no intent applies.\n\nIntents:\n" + "\n".join(names))
    max_tokens = max(ntok(tok, name) for name in names) + 1
    return [req("classify", [{"role": "system", "content": system}, {"role": "user", "content": row["text"]}], max_tokens,
                group="clinc", labels=names, reference=names[row["intent"]], scorer="label")
            for row in rng.sample(table.to_pylist(), n)]


QUALITY_URL = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev"


@source("quality", license="CC-BY-4.0", url="https://nyu-mll.github.io/quality/")
def quality(n, rng, tok, lengths, per_article=8):
    """Long-document multiple choice: a ~5k-token article shared by per_article questions."""
    articles = {}
    for line in url_jsonl(QUALITY_URL):  # one line per (article, question writer); an article spans several lines
        articles.setdefault(line["article_id"], {"article": line["article"], "questions": []})["questions"] += line["questions"]
    eligible = sorted(aid for aid, a in articles.items() if len(a["questions"]) >= per_article)
    out = []
    for aid in rng.sample(eligible, min(len(eligible), math.ceil(n / per_article))):
        system = ("Read the article, then answer the multiple choice question about it with the letter of the "
                  "correct option only.\n\n" + articles[aid]["article"].strip())
        for q in rng.sample(articles[aid]["questions"], per_article):
            out.append(req("classify", [{"role": "system", "content": system},
                                        {"role": "user", "content": mc_question(q["question"], q["options"])}], 1,
                           group=f"quality/{aid}", labels=list(LETTERS), reference=LETTERS[q["gold_label"] - 1],
                           scorer="choice"))
    return out[:n]


@source("gsm8k", license="MIT", url="https://huggingface.co/datasets/openai/gsm8k")
def gsm8k(n, rng, tok, lengths, shots=8):
    """8-shot chain-of-thought math; one few-shot block shared by every request, answers scored."""
    train = hf_rows("openai/gsm8k", "main/train-00000-of-00001.parquet")
    test = hf_rows("openai/gsm8k", "main/test-00000-of-00001.parquet")
    prefix = [{"role": "system", "content": "Solve the math word problem step by step. "
               "End with '#### ' followed by the final numeric answer."}]
    for ex in train[:shots]:
        prefix += [{"role": "user", "content": ex["question"]},
                   {"role": "assistant", "content": re.sub(r"<<[^>]*>>", "", ex["answer"])}]  # drop calculator notes
    max_tokens, ignore_eos = gen_len(lengths, 256, 512)
    return [req("generate", prefix + [{"role": "user", "content": q["question"]}], max_tokens, group="gsm8k",
                ignore_eos=ignore_eos, reference=q["answer"].split("####")[-1].strip().replace(",", ""),
                scorer="number")
            for q in rng.sample(test, n)]


@source("wildchat", license="ODC-BY", url="https://huggingface.co/datasets/allenai/WildChat-1M")
def wildchat(n, rng, tok, lengths, max_prompt=2048, max_output=512):
    """Real first-turn chat prompts; fixed-mode length is the logged reply's token count."""
    import pyarrow.compute as pc
    table = hf_table("allenai/WildChat-1M", "data/train-00000-of-00014.parquet", columns=["conversation", "language"])
    table = table.filter(pc.equal(table["language"], "English"))
    order = list(range(table.num_rows))
    rng.shuffle(order)
    out, seen = [], set()
    for i in order:
        if len(out) == n:
            break
        conv = table["conversation"][i].as_py()
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        prompt = conv[0]["content"].strip()
        if not prompt or prompt in seen or ntok(tok, prompt) > max_prompt:
            continue
        seen.add(prompt)
        max_tokens, ignore_eos = gen_len(lengths, min(max(ntok(tok, conv[1]["content"]), 16), max_output), max_output)
        out.append(req("generate", [{"role": "user", "content": prompt}], max_tokens, ignore_eos=ignore_eos))
    return out


def vocab_words(tok):
    """Vocab entries that decode to ' <word>', so joined words re-tokenize to about one token each."""
    decoded = (tok.decode([i]) for i in sorted(tok.get_vocab().values()))
    return sorted({w[1:] for w in decoded if re.fullmatch(r" [a-z]{3,10}", w)})


@source("synthetic", license="n/a", url="")
def synthetic(n, rng, tok, lengths, prompt_len=1024, shared_frac=0.5, groups=8, output_len=128):
    """Controlled sharing: `groups` shared prefixes of shared_frac * prompt_len words, unique suffixes."""
    words = vocab_words(tok)
    shared = round(shared_frac * prompt_len)
    prefixes = [" ".join(rng.choices(words, k=shared)) for _ in range(groups)]
    kind = "classify" if output_len <= 8 else "generate"
    max_tokens, ignore_eos = gen_len(lengths, output_len, output_len) if kind == "generate" else (output_len, False)
    out = []
    for i in range(n):
        messages = [{"role": "system", "content": prefixes[i % groups]}] if shared else []
        messages.append({"role": "user", "content": " ".join(rng.choices(words, k=prompt_len - shared))})
        out.append(req(kind, messages, max_tokens, group=f"synthetic/{i % groups}" if shared else None,
                       ignore_eos=ignore_eos))
    return out
