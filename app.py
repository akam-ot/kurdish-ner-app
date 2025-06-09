import os
import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Kurdish NER", layout="centered")
st.title("🧠 Kurdish NER")
st.markdown("This app uses a fine-tuned XLM-RoBERTa model to recognize named entities in **Kurmanji Kurdish**.")

# Load model pipeline
@st.cache_resource
def load_pipeline():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    return pipeline(
        "ner",
        model="akam-ot/ku-ner-xlmr",
        tokenizer="akam-ot/ku-ner-xlmr",
        aggregation_strategy="simple",
        use_auth_token=token
    )

ner_pipe = load_pipeline()

# Text input
text = st.text_area("✍️ Enter a Kurmanji Kurdish sentence in Latin alphabet:", height=150)

# Analyze
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            results = ner_pipe(text)
            filtered = [ent for ent in results if ent['word'].strip().isalnum()]

        if not filtered:
            st.info("No entities detected.")
        else:
            st.subheader("🔍 Detected Entities:")
            for ent in filtered:
                st.write(f"• **{ent['word']}** → {ent['entity_group']} (score: {ent['score']:.2f})")
