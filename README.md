# Kurdish NER

A Streamlit app and Hugging Face model for Named Entity Recognition (NER) in Kurmanji Kurdish (Hawar Latin script).

---

## 📖 Overview

This repository provides a fine-tuned xlm-roberta-base model for NER over three entity types: PERSON (PER), LOCATION (LOC), and ORGANIZATION (ORG). It includes:

- A Streamlit demo (`app.py`) for interactive inference
- A model card with training details and evaluation results
- Scripts to reproduce training, evaluation, and error analysis in Colab

---

## 🤗 Model Card

- **Model URI**: [hf.co/akam-ot/ku-ner-xlmr](https://hf.co/akam-ot/ku-ner-xlmr)
- **Base**: xlm-roberta-base (270M params, 12 layers, 768 hidden, 12 heads)
- **Head**: token classification linear layer predicting 7 BIO tags: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG

## 🛠 Training Details

| Setting | Value |
|---------|-------|
| Epochs | 5 |
| Batch size | 16 |
| Max sequence length | 128 tokens |
| Optimizer | AdamW (weight_decay=0.01) |
| Learning rate | 2e-5 |
| Warmup steps | 500 |
| Gradient clipping | 1.0 |

## 📊 Evaluation

| Entity | Precision | Recall | F1 Score |
|--------|-----------|--------|----------|
| PER | 0.872 | 0.867 | 0.869 |
| LOC | 0.882 | 0.883 | 0.882 |
| ORG | 0.728 | 0.793 | 0.759 |
| **Overall** | **0.833** | **0.851** | **0.841** |

---

## 🚀 Usage

```bash
pip install transformers streamlit
```

```python
from transformers import pipeline

# load
ner = pipeline(
    "ner",
    model="akam-ot/ku-ner-xlmr",
    tokenizer="akam-ot/ku-ner-xlmr",
    aggregation_strategy="simple"
)

# inference
tokens = ner("Serokê Komara Fransa, Emanuel Macron, di Parisê de got...")
print(tokens)
```

To launch the demo:

```bash
streamlit run app.py
```

---

## License

Released under Apache 2.0.
