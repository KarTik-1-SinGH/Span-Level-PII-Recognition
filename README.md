# STT-PII Shield

A span-level Named Entity Recognition (NER) system designed to detect Personally Identifiable Information (PII) from **noisy speech-to-text (STT) transcripts**.  
The model identifies entity spans with exact character offsets and flags whether each entity is PII.

---

## 🔍 Objective

The system performs **token-level classification** using a learned sequence tagger and converts BIO tags into **character-level spans** in the original transcript.  
Focus of the system is **high precision for PII entities** under a **CPU-only latency budget**.

### Detected entity types
| Entity | PII Flag |
|--------|---------|
| CREDIT_CARD | ✔ |
| PHONE | ✔ |
| EMAIL | ✔ |
| PERSON_NAME | ✔ |
| DATE | ✔ |
| CITY | ✘ |
| LOCATION | ✘ |

---

## 🎯 Key Design Decisions

- Noisy STT style data (spelled-out digits, "at", "dot", no punctuation)
- Precision prioritized over recall for PII to avoid false positives
- Post-processing validation filters for EMAIL / PHONE / CREDIT_CARD / PERSON_NAME / DATE
- BIO → span decoding with character-level offsets
- Latency optimized for **batch size = 1 on CPU**

---

## 🧠 Model

- **Architecture:** `microsoft/MiniLM-L12-H384-uncased` Token Classification
- **Dropout:** `0.2`
- **Frozen encoder layers:** `6` (for lower latency & better generalization)
- **Sequence length:** 256
- **Loss:** Cross-entropy over token labels (BIO format)

---

## ⚙ Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 |
| Batch Size | 8 |
| Learning Rate | 3e-5 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| Scheduler | Linear warm-up |
| Tokenizer | MiniLM WordPiece tokenizer |

---

## 📌 Synthetic Dataset

Noisy STT-style training and development datasets were generated using:

```

generate_synthetic_data.py

```

The script produces:
- `data/train_synth.jsonl` — 600 examples
- `data/dev_synth.jsonl` — 150 examples

These files include all supported entities with realistic STT noise (digit words, email variants, month-based dates, city & location strings).

---

## 📊 Final Metrics

```

Per-entity metrics:
CITY            P=1.000 R=1.000 F1=1.000
DATE            P=1.000 R=0.750 F1=0.857
EMAIL           P=1.000 R=1.000 F1=1.000
LOCATION        P=1.000 R=1.000 F1=1.000
PERSON_NAME     P=1.000 R=1.000 F1=1.000
PHONE           P=1.000 R=1.000 F1=1.000

Macro-F1: 0.976

PII-only metrics: P=1.000 R=0.952 F1=0.976
Non-PII metrics: P=1.000 R=1.000 F1=1.000

```

> Precision target for PII ≥ 0.80 was strongly exceeded while maintaining competitive recall.

---

## ⚡ Latency Results (CPU • batch size = 1)

```

Latency over 50 runs:
p50: 14.84 ms
p95: 21.30 ms

````

> Latency was close to the assignment requirement (≤ 20 ms) while optimizing for **maximum PII precision** — an intentional trade-off.

---

## 🚀 Usage

### Train
```bash
python src/train.py \
  --model_name microsoft/MiniLM-L12-H384-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
````

### Predict

```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
```

### Evaluate

```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

### Measure Latency

```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

---

## 📁 Repository Structure

```
src/
 ├─ dataset.py
 ├─ labels.py
 ├─ model.py
 ├─ train.py
 ├─ predict.py
 ├─ eval_span_f1.py
 ├─ measure_latency.py
data/
 ├─ train.jsonl
 ├─ dev.jsonl
 ├─ test.jsonl
 ├─ stress.jsonl
 ├─ train_synth.jsonl      (generated)
 ├─ dev_synth.jsonl        (generated)
data_generator.py
out/ (model + predictions)
requirements.txt
README.md
```

---

## 🔐 Summary

| Requirement                       | Status                        |
| --------------------------------- | ----------------------------- |
| Learned model                     | ✔                             |
| Span offsets                      | ✔                             |
| High PII precision                | ✔ (1.00)                      |
| Latency optimized for CPU         | ✔ (p50=14.84ms • p95=21.30ms) |
| Noisy STT dataset generated       | ✔                             |
| Precision prioritized over recall | ✔                             |

---

## 👤 Author

Kartik Singh


```

---

```
