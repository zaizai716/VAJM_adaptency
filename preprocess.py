#!/usr/bin/env python3
import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/mitul/VAJM_adaptency/prm800k-vajm/prm800k/data"
OUT_DIR    = "/Users/mitul/VAJM_adaptency/processed/train_dataset"
MODEL_NAME = "internlm/internlm2-math-base-20b"

os.makedirs(OUT_DIR, exist_ok=True) 

# ─── 1) LOAD RAW SPLITS ─────────────────────────────────────────────────────────
files = {
    "train": os.path.join(DATA_DIR, "phase2_train.jsonl"),
    "test":  os.path.join(DATA_DIR, "phase2_test.jsonl")
}
raw: DatasetDict = load_dataset("json", data_files=files)

# ─── 2) FLATTEN OUT GOOD STEPS IN PYTHON ─────────────────────────────────────────
def extract_good_steps(example):
    out = []
    prob = example["question"]["problem"]
    gt = example["question"]["ground_truth_answer"]

    for step in example["label"]["steps"]:
        completions = step.get("completions") or []
        for comp in completions:
            if comp.get("rating") == 1:
                out.append({
                    "problem": prob,
                    "ground_truth_answer": gt,
                    "completion": comp.get("text", ""),
                    "rating": 1
                })

        human = step.get("human_completion")
        if human is not None:
            text = human.get("text") if isinstance(human, dict) else human
            if text:
                out.append({
                    "problem": prob,
                    "ground_truth_answer": gt,
                    "completion": text,
                    "rating": 1
                })
    return out

flat_splits = {}
for split, ds in raw.items():
    flat_list = []
    for example in ds:
        flat_list.extend(extract_good_steps(example))
    flat_splits[split] = Dataset.from_list(flat_list)
    print(f"{split}: flattened to {len(flat_list)} records")

flat = DatasetDict(flat_splits)

# ─── 3) TOKENIZE INPUTS & LABELS ───────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, torch_dtype="auto", trust_remote_code=True, use_fast=True
)

def tokenize_batch(batch):
    texts = [
        f"Question: {p}\nAnswer: {gt}\nStep: {c}"
        for p, gt, c in zip(batch["problem"], batch["ground_truth_answer"], batch["completion"])
    ]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # preserve rating for downstream use
    enc["rating"] = batch["rating"]
    return enc

tok_splits = {}
for split, ds in flat.items():
    tok_splits[split] = ds.map(
        tokenize_batch,
        batched=True,
        remove_columns=["problem", "ground_truth_answer", "completion"]
    )
    print(f"{split}: tokenized {len(tok_splits[split])} records")

tok = DatasetDict(tok_splits)

# ─── 4) SAVE TOKENIZED DATASETS AS JSONL ────────────────────────────────────────
for split, dataset in tok.items():
    out_path = os.path.join(OUT_DIR, f"phase2_{split}.jsonl")
    dataset.to_json(out_path, orient="records", lines=True)
    print(f"Saved {split} dataset to {out_path}")

print("✅ Preprocessing and saving complete.")
