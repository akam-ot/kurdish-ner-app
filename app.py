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
st.set_page_config(page_title="Kurdish NER", layout="centered", page_icon="🧠")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .entity-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .entity-word {
        background: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        color: #1976d2;
    }
    
    .entity-label {
        background: #f3e5f5;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
        color: #7b1fa2;
    }
    
    .sample-box {
        background: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        height: 2.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Kurdish NER</h1>
    <p>Advanced Named Entity Recognition for Kurmanji Kurdish</p>
</div>
""", unsafe_allow_html=True)

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
# 4) Sample sentences
# ─────────────────────────────────────────────
SAMPLE_SENTENCES = [
    "Navê min Hejar e û ez li Hewlêr dijîm.",
    "Serok Obama û Merkel li Washington hevdîtin kirin.",
    "Zanîngeha Dihokê li Kurdistanê ye.",
    "Rojnamevana BBC li Londonê kar dike.",
    "Wezîrê Pêşmerge yê Herêma Kurdistanê axivî.",
]

# ─────────────────────────────────────────────
# 5) Initialize session state
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
        "confidence": float(confidence),
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
                        "score": float(ent["score"]), 
                    })
        except Exception as e:
            add_feedback_message("error", f"Error processing sentence: {str(e)}")
    
    return entities

# ─────────────────────────────────────────────
# 6) Main Interface
# ─────────────────────────────────────────────

# Display any feedback messages
display_feedback_messages()

# Sample sentences section
st.markdown("### 📝 Try These Sample Sentences")
st.markdown('<div class="sample-box">', unsafe_allow_html=True)

# Create columns for sample sentences
cols = st.columns(2)
for i, sample in enumerate(SAMPLE_SENTENCES):
    with cols[i % 2]:
        if st.button(f"📄 {sample[:30]}{'...' if len(sample) > 30 else ''}", 
                    key=f"sample_{i}",
                    help=sample):
            st.session_state.selected_sample = sample
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Input text area
default_text = st.session_state.get('selected_sample', '')
text = st.text_area(
    "✍️ Enter a Kurmanji Kurdish paragraph or sentences (Latin alphabet):",
    height=120,
    placeholder="Navê min Hejar e û ez li Hewlêr dijîm.",
    key="input_text",
    value=default_text
)

# Clear the selected sample after it's been used
if 'selected_sample' in st.session_state:
    del st.session_state.selected_sample

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
    st.markdown("### 🔍 Detected Entities")
    
    for idx, ent in enumerate(st.session_state.entities):
        st.markdown(f'<div class="entity-card">', unsafe_allow_html=True)
        
        # Entity information
        st.markdown(f"**📄 Sentence:** {ent['sentence']}")
        st.markdown(f"**🏷️ Entity:** <span class='entity-word'>{ent['word']}</span> → "
                   f"<span class='entity-label'>{ent['pred']}</span> "
                   f"(confidence: {ent['score']:.2f})", unsafe_allow_html=True)
        
        # Correction interface with better alignment
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            correction_key = f"correction_{idx}"
            corrected_label = st.selectbox(
                "🔧 Correct label:",
                options=["PER", "LOC", "ORG", "O"],
                index=["PER", "LOC", "ORG", "O"].index(ent["pred"]) if ent["pred"] in ["PER", "LOC", "ORG"] else 3,
                key=correction_key,
                help="Select the correct entity type"
            )
        
        with col2:
            # Status indicator
            if corrected_label != ent["pred"]:
                st.markdown("**🔄 Status:** Changed")
            else:
                st.markdown("**✅ Status:** Same")
        
        with col3:
            # Aligned save button
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("💾 Save", key=f"save_{idx}", help="Save correction to database"):
                clear_feedback_messages()
                
                if save_correction(
                    ent["sentence"], 
                    ent["word"], 
                    ent["pred"], 
                    corrected_label, 
                    ent["score"]
                ):
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8) Sidebar information
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ Information")
    
    # Entity types info
    st.markdown("""
    **🏷️ Entity Types:**
    - **PER**: Person names (e.g., Obama, Merkel)
    - **LOC**: Locations (e.g., Hewlêr, Washington)
    - **ORG**: Organizations (e.g., BBC, UN)
    - **O**: Not an entity
    """)
    
    # Instructions
    st.markdown("""
    **📋 How to use:**
    1. 📄 Try a sample sentence or enter your own
    2. 🔍 Click "Analyze Text"
    3. 👀 Review detected entities
    4. 🔧 Correct wrong predictions
    5. 💾 Save corrections to help improve the model
    """)
    
    # Current analysis stats
    if st.session_state.entities:
        st.markdown("### 📊 Current Analysis")
        
        # Create metric cards
        total_entities = len(st.session_state.entities)
        st.markdown(f'<div class="metric-card"><h3>{total_entities}</h3><p>Entities Found</p></div>', 
                   unsafe_allow_html=True)
        
        # Count by type
        entity_counts = {}
        for ent in st.session_state.entities:
            entity_counts[ent["pred"]] = entity_counts.get(ent["pred"], 0) + 1
        
        st.markdown("**📈 By Type:**")
        for entity_type, count in entity_counts.items():
            percentage = (count / total_entities) * 100
            st.markdown(f"- **{entity_type}**: {count} ({percentage:.1f}%)")
        
        # Average confidence
        avg_conf = sum(ent["score"] for ent in st.session_state.entities) / total_entities
        st.markdown(f"**🎯 Avg Confidence**: {avg_conf:.2f}")
    
    # Additional info
    st.markdown("---")
    st.markdown("""
    **🔬 About the Model:**
    - Based on XLM-RoBERTa
    - Fine-tuned for Kurdish
    - Supports Latin script
    """)
    
    st.markdown("""
    **💡 Tips:**
    - Use clear, well-formed sentences
    - Check for proper names and locations
    - Your corrections help improve the model
    """)
