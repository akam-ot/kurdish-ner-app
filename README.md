# Kurdish NER  

**Demo:** https://kurdish-ner-app-dhkvzatwygtk8rvvgwbty3.streamlit.app/

A fine-tuned XLM-RoBERTa model for Named Entity Recognition (NER) on Kurmanji Kurdish.

## 📖 Overview

- **Base model:** xlm-roberta-base
- **Fine-tuned on:** ~8,146 sentences (train/test 80%/20%)
- **Supported entity types:**
  - PER (person)
  - LOC (location)
  - ORG (organization)

## 🛠️ Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/akam-ot/kurdish-ner-app.git
cd kurdish-ner-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run locally

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 🚀 Usage

1. Paste or type a Kurmanji Kurdish sentence (Latin script) into the text area.
2. Click **Analyze**.
3. See detected entities (word, label, confidence).

## 📊 Model Performance

| Entity | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------| 
| PER    | 0.872     | 0.867  | 0.869 | 652     |
| LOC    | 0.882     | 0.882  | 0.882 | 1,047   |
| ORG    | 0.728     | 0.793  | 0.759 | 739     |
| **Overall** | **0.833** | **0.851** | **0.841** | — |

*(Metrics on held-out test set.)*

## 📝 Model Card

See [akam-ot/ku-ner-xlmr](https://huggingface.co/akam-ot/ku-ner-xlmr) on Hugging Face for full details, tags, and license.


## 📄 License

Apache 2.0
