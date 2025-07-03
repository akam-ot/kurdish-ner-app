import os
import streamlit as st
from transformers import pipeline
from sentence_splitter import SentenceSplitter
from supabase import create_client, Client

# 1) Supabase connection for storing user feedback
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("SUPABASE_URL or SUPABASE_ANON_KEY not set as environment variables / secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# 2) Streamlit page setup
st.set_page_config(page_title="Kurdish NER", layout="centered")
st.title("🧠 Kurdish NER")
st.markdown(
    "This app uses a fine-tuned **XLM-RoBERTa** model to recognise named entities in **Kurmanji Kurdish**. "
    "You can also **correct predictions** to help us improve the system!"
)

# 3) Cached resources
@st.cache_resource
def load_pipeline():
    return pipeline(
        "ner",
        model="akam-ot/ku-ner-xlmr",
        tokenizer="akam-ot/ku-ner-xlmr",
        aggregation_strategy="simple",
    )

ner_pipe = load_pipeline()

@st.cache_resource
def get_splitter():
    return SentenceSplitter(language="en")

splitter = get_splitter()

# 4) Main app
text = st.text_area(
    "✍️ Enter a Kurmanji Kurdish paragraph or sentences (Latin alphabet):",
    height=150,
    placeholder="Navê min Hejar e û ez li Hewlêr dijîm."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Analyzing..."):
        sentences = splitter.split(text)
        entities = []

        for sent in sentences:
            for ent in ner_pipe(sent):
                # Clean & filter
                token = ent["word"].strip(" .,!?:;\"'()")
                if (
                    ent["score"] > 0.85
                    and token
                    and not all(c in ".,!?\"'()" for c in token)
                ):
                    entities.append(
                        {
                            "sentence": sent.strip(),
                            "word": token,
                            "pred": ent["entity_group"],
                            "score": ent["score"],
                        }
                    )

    #Results
    if not entities:
        st.info("No high-confidence entities detected.")
        st.stop()

    st.subheader("🔍 Detected entities (click to correct):")

    for idx, ent in enumerate(entities):
        st.write(f"**Sentence:** {ent['sentence']}")
        st.write(
            f"• **{ent['word']}** → {ent['pred']} "
            f"(score {ent['score']:.2f})"
        )

        # Feedback form for each entity
        with st.form(f"form_{idx}"):
            corrected = st.selectbox(
                "Correct label (if wrong):",
                ["PER", "LOC", "ORG", "O"],
                index=["PER", "LOC", "ORG", "O"].index(ent["pred"])
                if ent["pred"] in ["PER", "LOC", "ORG"]
                else 3,
            )
            submitted = st.form_submit_button("Submit correction")
            if submitted:
                data = {
                    "sentence": ent["sentence"],
                    "word": ent["word"],
                    "model_prediction": ent["pred"],
                    "corrected_label": corrected,
                    "confidence": ent["score"],
                }
                res = supabase.table("entity_feedback").insert(data).execute()
                if res.data:
                    st.success("✅ Correction saved — thank you!")
                else:
                    st.error("❌ Could not save correction (check Supabase).")
