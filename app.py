import os
import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Kurdish NER", layout="centered")
st.title("🧠 Kurdish NER")
st.markdown(
    "This app uses a fine-tuned XLM-RoBERTa model to recognize named entities in **Kurmanji Kurdish**."
)

# Load model pipeline
@st.cache_resource
def load_pipeline():
    return pipeline(
        "ner",
        model="akam-ot/ku-ner-xlmr",
        tokenizer="akam-ot/ku-ner-xlmr",
        aggregation_strategy="simple"
    )

ner_pipe = load_pipeline()

# Text input
text = st.text_area(
    "✍️ Enter a Kurmanji Kurdish sentence in Latin alphabet:",
    height=150,
    placeholder="Navê min Hejar e û ez li Hewlêr dijîm."
)

# Analyze button
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            raw_results = ner_pipe(text)

            # Apply confidence threshold (keep only >0.85) & remove single punctuation
            filtered = []
            for ent in raw_results:
                word_clean = ent["word"].strip().strip(".,!?\"'()")
                if ent["score"] > 0.85 and word_clean and not all(c in ".,!?\"'()" for c in word_clean):
                    filtered.append({
                        "word": word_clean,
                        "entity_group": ent["entity_group"],
                        "score": ent["score"],
                    })

        if not filtered:
            st.info("No high-confidence entities detected.")
        else:
            st.subheader("🔍 Detected Entities:")
            for ent in filtered:
                st.write(f"• **{ent['word']}** → {ent['entity_group']} (score: {ent['score']:.2f})")
