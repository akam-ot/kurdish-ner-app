import streamlit as st
from transformers import pipeline
from sentence_splitter import SentenceSplitter
from supabase import create_client, Client
import time

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
# 4) Initialize session state
# ─────────────────────────────────────────────
if "entities" not in st.session_state:
    st.session_state.entities = []
if "feedback_messages" not in st.session_state:
    st.session_state.feedback_messages = []
if "current_text" not in st.session_state:
    st.session_state.current_text = ""

# ─────────────────────────────────────────────
# 5) Helper functions
# ─────────────────────────────────────────────
def add_feedback_message(msg_type, message):
    """Add a feedback message to session state"""
    st.session_state.feedback_messages.append({"type": msg_type, "message": message})

def clear_feedback_messages():
    """Clear all feedback messages"""
    st.session_state.feedback_messages = []

def display_feedback_messages():
    """Display all feedback messages"""
    for msg in st.session_state.feedback_messages:
        if msg["type"] == "success":
            st.success(msg["message"])
        elif msg["type"] == "error":
            st.error(msg["message"])
        elif msg["type"] == "warning":
            st.warning(msg["message"])
        elif msg["type"] == "info":
            st.info(msg["message"])

def save_correction(sentence, word, model_pred, corrected_label, confidence):
    """Save correction to Supabase"""
    data = {
        "sentence": sentence,
        "word": word,
        "model_prediction": model_pred,
        "corrected_label": corrected_label,
        "confidence": float(confidence),  # Convert numpy float32 to Python float
    }
    
    try:
        result = supabase.table("entity_feedback").insert(data).execute()
        add_feedback_message("success", f"✅ Correction saved: '{word}' → {corrected_label}")
        return True
    except Exception as e:
        error_msg = str(e)
        if "duplicate key" in error_msg.lower():
            add_feedback_message("warning", f"⚠️ Correction for '{word}' already exists")
        else:
            add_feedback_message("error", f"❌ Failed to save correction: {error_msg}")
        return False

def process_text(text):
    """Process text and extract entities"""
    if not text.strip():
        return []
    
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
                    entities.append({
                        "sentence": sent.strip(),
                        "word": token,
                        "pred": ent["entity_group"],
                        "score": float(ent["score"]),  # Convert numpy float32 to Python float
                    })
        except Exception as e:
            add_feedback_message("error", f"Error processing sentence: {str(e)}")
    
    return entities

# ─────────────────────────────────────────────
# 6) Main Interface
# ─────────────────────────────────────────────

# Display any feedback messages
display_feedback_messages()

# Input text area
text = st.text_area(
    "✍️ Enter a Kurmanji Kurdish paragraph or sentences (Latin alphabet):",
    height=150,
    placeholder="Navê min Hejar e û ez li Hewlêr dijîm.",
    key="input_text"
)

# Analyze button
if st.button("🔍 Analyze Text", type="primary"):
    clear_feedback_messages()
    
    if not text.strip():
        add_feedback_message("warning", "Please enter some text to analyze.")
    else:
        st.session_state.current_text = text
        
        with st.spinner("Analyzing text..."):
            entities = process_text(text)
            
            # Remove duplicates
            unique_entities = []
            seen = set()
            for ent in entities:
                key = f"{ent['sentence']}_{ent['word']}"
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(ent)
            
            st.session_state.entities = unique_entities
            
            if not unique_entities:
                add_feedback_message("info", "No high-confidence entities detected.")
            else:
                add_feedback_message("success", f"Found {len(unique_entities)} entities!")

# ─────────────────────────────────────────────
# 7) Display entities and correction interface
# ─────────────────────────────────────────────
if st.session_state.entities:
    st.subheader("🔍 Detected Entities")
    
    for idx, ent in enumerate(st.session_state.entities):
        with st.container():
            st.markdown(f"**Sentence:** {ent['sentence']}")
            st.markdown(f"**Word:** `{ent['word']}` → **{ent['pred']}** (confidence: {ent['score']:.2f})")
            
            # Create correction interface
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                correction_key = f"correction_{idx}"
                corrected_label = st.selectbox(
                    "Correct label:",
                    options=["PER", "LOC", "ORG", "O"],
                    index=["PER", "LOC", "ORG", "O"].index(ent["pred"]) if ent["pred"] in ["PER", "LOC", "ORG"] else 3,
                    key=correction_key
                )
            
            with col2:
                if st.button("💾 Save Correction", key=f"save_{idx}"):
                    clear_feedback_messages()
                    
                    if save_correction(
                        ent["sentence"], 
                        ent["word"], 
                        ent["pred"], 
                        corrected_label, 
                        ent["score"]
                    ):
                        # Force a rerun to show the success message
                        st.rerun()
            
            with col3:
                if corrected_label != ent["pred"]:
                    st.markdown("🔄 **Changed**")
                else:
                    st.markdown("✅ **Same**")
            
            st.divider()

# ─────────────────────────────────────────────
# 8) Sidebar information
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    **Entity Types:**
    - **PER**: Person names
    - **LOC**: Locations
    - **ORG**: Organizations
    - **O**: Not an entity
    
    **How to use:**
    1. Enter Kurdish text
    2. Click "Analyze Text"
    3. Review detected entities
    4. Correct any wrong predictions
    5. Click "Save Correction"
    """)
    
    if st.session_state.entities:
        st.markdown(f"**Current Analysis:**")
        st.markdown(f"- {len(st.session_state.entities)} entities found")
        
        # Count by type
        entity_counts = {}
        for ent in st.session_state.entities:
            entity_counts[ent["pred"]] = entity_counts.get(ent["pred"], 0) + 1
        
        for entity_type, count in entity_counts.items():
            st.markdown(f"- {entity_type}: {count}")
