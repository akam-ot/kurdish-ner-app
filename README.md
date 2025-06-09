# 🏷️ Kurdish NER

**Demo:** https://kurdish-ner-app-dhkvzatwygtk8rvvgwbty3.streamlit.app/

A fine-tuned XLM-RoBERTa model for Named Entity Recognition (NER) on Kurmanji Kurdish (Hawar Latin alphabet).

## 📖 Overview

- **Base model:** xlm-roberta-base
- **Fine-tuned on:** Manually annotated Kurmanji Kurdish text dataset
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

The app will automatically open in your browser, or you can manually navigate to the URL shown in the terminal (typically http://localhost:8501).

## 🚀 Usage

1. Paste or type a Kurmanji Kurdish sentence (Latin script) into the text area.
2. Click **Analyze**.
3. See detected entities (word, label, confidence).

## 📊 Model Performance

| Entity | Precision | Recall | F1 |
|--------|-----------|--------|----|
| PER    | 0.872     | 0.867  | 0.869 |
| LOC    | 0.882     | 0.882  | 0.882 |
| ORG    | 0.728     | 0.793  | 0.759 |
| **Overall** | **0.833** | **0.851** | **0.841** |

*(Metrics on held-out test set.)*

## 📝 Model Card

See [akam-ot/ku-ner-xlmr](https://huggingface.co/akam-ot/ku-ner-xlmr) on Hugging Face for full details, tags, and license.


## 📄 License

Apache 2.0
