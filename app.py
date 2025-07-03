import streamlit as st
from transformers import pipeline
from sentence_splitter import SentenceSplitter
from supabase import create_client, Client
import hashlib

# ─────────────────────────────────────────────
# 1) Supabase connection
# ─────────────────────────────────────────────
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("❌ SUPABASE_URL or SUPABASE_ANON_KEY not set in Streamlit secrets.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ─────────────────────────────────────────────
# 2) Streamlit page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Kurdish NER", layout="centered")
st.title("🧠 Kurdish NER")
st.markdown(
    "This app uses a fine-tuned **XLM-RoBERTa** model to recognize named entities in **Kurmanji Kurdish**. "
    "You can also **correct predictions** to help improve the system!"
)

# ─────────────────────────────────────────────
# 3) Cached resources
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return pipeline(
        "ner",
        model="akam-ot/ku-ner-xlmr",
        tokenizer="akam-ot/ku-ner-xlmr",
        aggregation_strategy="simple",
    )

@st.cache_resource
def get_splitter():
    return SentenceSplitter(language="en")

ner_pipe = load_pipeline()
splitter = get_splitter()

# ─────────────────────────────────────────────
# 4) Helper functions
# ─────────────────────────────────────────────
def create_entity_key(sentence, word):
    """Create a unique key for an entity to prevent duplicates"""
    return hashlib.md5(f"{sentence.strip()}_{word.strip()}".encode()).hexdigest()

def deduplicate_entities(entities):
    """Remove duplicate entities based on sentence and word"""
    seen = set()
    unique_entities = []
    
    for ent in entities:
        key = create_entity_key(ent["sentence"], ent["word"])
        if key not in seen:
            seen.add(key)
            unique_entities.append(ent)
    
    return unique_entities

# ─────────────────────────────────────────────
# 5) Initialize session state
# ─────────────────────────────────────────────
if "submitted_corrections" not in st.session_state:
    st.session_state.submitted_corrections = set()

# ─────────────────────────────────────────────
# 6) Main App Logic
# ─────────────────────────────────────────────
text = st.text_area(
    "✍️ Enter a Kurmanji Kurdish paragraph or sentences (Latin alphabet):",
    height=150,
    placeholder="Navê min Hejar e û ez li Hewlêr dijîm."
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    # Reset submitted corrections for new analysis
    st.session_state.submitted_corrections = set()

    with st.spinner("Analyzing..."):
        sentences = splitter.split(text)
        entities = []

        for sent in sentences:
            try:
                for ent in ner_pipe(sent):
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
            except Exception as e:
                st.error(f"Error processing sentence: {sent[:50]}... - {e}")
                continue

    # Remove duplicates
    entities = deduplicate_entities(entities)

    if not entities:
        st.info("No high-confidence entities detected.")
        st.stop()

    st.subheader("🔍 Detected Entities (click to correct):")

    for idx, ent in enumerate(entities):
        entity_key = create_entity_key(ent["sentence"], ent["word"])
        
        st.write(f"**Sentence:** {ent['sentence']}")
        st.write(f"• **{ent['word']}** → {ent['pred']} (score: {ent['score']:.2f})")

        # Check if this entity has been corrected
        if entity_key in st.session_state.submitted_corrections:
            st.success("✅ Correction already submitted for this entity!")
        else:
            # Feedback form per entity
            with st.form(f"form_{idx}"):
                corrected = st.selectbox(
                    "Correct label (if wrong):",
                    ["PER", "LOC", "ORG", "O"],
                    index=["PER", "LOC", "ORG", "O"].index(ent["pred"])
                    if ent["pred"] in ["PER", "LOC", "ORG"]
                    else 3,
                    key=f"select_{idx}"
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
                    
                    # Show immediate feedback
                    with st.spinner("Saving correction..."):
                        try:
                            result = supabase.table("entity_feedback").insert(data).execute()
                            
                            st.session_state.submitted_corrections.add(entity_key)
                            st.success(f"✅ Correction saved! {ent['word']} → {corrected}")
                            
                            st.info("💡 Refresh the page to see updated status")
                            
                        except Exception as e:
                            # Handle specific error types
                            error_msg = str(e)
                            if "duplicate key" in error_msg.lower():
                                st.warning("⚠️ This correction has already been submitted.")
                            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                                st.error("❌ Network error. Please check your connection and try again.")
                            else:
                                st.error(f"❌ Error saving correction: {error_msg}")
                                with st.expander("🔧 Full error details"):
                                    st.code(str(e))
        
        st.divider()

