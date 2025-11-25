import json
import argparse
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from labels import ID2LABEL, label_is_pii


# ---------- BIO → span decoding ----------

def bio_to_spans(text, offsets, label_ids):
    """
    Convert per-token BIO labels to character-level spans.
    offsets: list of (start, end) for each token
    label_ids: list of label IDs aligned with offsets
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # Skip special / padding tokens (offset (0, 0))
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(int(lid), "O")

        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)

        if prefix == "B":
            # close previous span
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            # continue same entity type, otherwise start a new span
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


# ---------- Post-processing helpers (precision boosts) ----------

DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

MONTH_WORDS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def text_to_digits(span_text: str) -> str:
    """
    Convert a noisy STT span into a string of digits.
    Handles spelled-out digits like 'one two three'.
    """
    t = span_text.lower()
    tokens = re.split(r"\s+", t.strip())
    digits = []

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            digits.append(tok)
        elif tok in DIGIT_WORDS:
            digits.append(DIGIT_WORDS[tok])

    return "".join(digits)


def luhn_check(number: str) -> bool:
    """
    Luhn checksum for validating credit card-like sequences.
    """
    if not number.isdigit():
        return False
    total = 0
    reverse_digits = number[::-1]
    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def is_valid_email(span_text: str) -> bool:
    t = span_text.lower().strip()
    # Direct email pattern (if punctuation survives STT)
    if "@" in t:
        return True
    # STT-style: "john doe at gmail dot com"
    if " at " in t and " dot " in t:
        return True
    return False


def is_valid_phone(span_text: str) -> bool:
    digits = text_to_digits(span_text)
    # Require a reasonable phone length
    return 7 <= len(digits) <= 15


def is_valid_credit_card(span_text: str) -> bool:
    digits = text_to_digits(span_text)
    # Typical card length 13–19 digits
    if not (13 <= len(digits) <= 19):
        return False
    # Optional but strong precision boost
    return luhn_check(digits)


def is_likely_date(span_text: str) -> bool:
    t = span_text.lower()
    if any(m in t for m in MONTH_WORDS):
        return True
    # Fallback: sequence with at least ~4 digits after normalization
    digits = text_to_digits(span_text)
    return len(digits) >= 4


def is_likely_person(span_text: str) -> bool:
    t = span_text.strip()
    # Very short strings are rarely good names in this setting
    if len(t) < 3:
        return False
    # Heuristic: if mostly digits, it's probably not a person
    num_digits = sum(c.isdigit() for c in t)
    if num_digits >= max(1, len(t) // 2):
        return False
    return True


def filter_spans(text: str, spans):
    """
    Apply label-specific validation rules to increase PII precision.
    We **only remove** spans predicted by the model – no new spans are created.
    """
    filtered = []
    for start, end, label in spans:
        span_text = text[start:end]

        if label == "EMAIL" and not is_valid_email(span_text):
            continue
        if label == "PHONE" and not is_valid_phone(span_text):
            continue
        if label == "CREDIT_CARD" and not is_valid_credit_card(span_text):
            continue
        if label == "DATE" and not is_likely_date(span_text):
            continue
        if label == "PERSON_NAME" and not is_likely_person(span_text):
            continue

        # CITY and LOCATION are kept as-is (non-PII but still useful entities)
        filtered.append((start, end, label))

    return filtered


# ---------- Main prediction script ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    model_name_or_dir = args.model_dir if args.model_name is None else args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            spans = filter_spans(text, spans)

            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
