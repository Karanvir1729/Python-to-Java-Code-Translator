import streamlit as st
import pandas as pd
import os
from translate import load_model, translate_sample, DATA_DIR, PAIRS_FILE
from run_evaluation import evaluate_translations

# Page Config
st.set_page_config(page_title="AVATAR Translation UI", layout="wide")

st.title("🐍 Python to ☕ Java Translator (AVATAR)")

# Sidebar
st.sidebar.header("Configuration")
num_samples = st.sidebar.slider("Number of Samples", min_value=1, max_value=20, value=5)
model_name = st.sidebar.text_input("Model Name", value="uclanlp/plbart-base")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv(PAIRS_FILE)

try:
    df = load_data()
    st.sidebar.success(f"Loaded {len(df)} pairs from dataset.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Load Model
@st.cache_resource
def get_model():
    return load_model()

# Main Area
tab1, tab2 = st.tabs(["Translation", "Evaluation"])

with tab1:
    st.header("Run Translation")
    
    if st.button("Start Translation"):
        with st.spinner("Loading model..."):
            tokenizer, model = get_model()
        
        progress_bar = st.progress(0)
        
        for i in range(num_samples):
            st.subheader(f"Sample {i+1}")
            
            # Translate
            python_code, java_target, java_pred = translate_sample(i, tokenizer, model, df)
            
            # Display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Python Source**")
                st.code(python_code, language="python")
            
            with col2:
                st.markdown("**Java Ground Truth**")
                st.code(java_target, language="java")
                
            with col3:
                st.markdown("**Java Prediction**")
                st.code(java_pred, language="java")
            
            # Save prediction for evaluation (mimicking the script behavior)
            output_dir = "translated_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, f"sample_{i+1}_translated.java"), 'w', encoding='utf-8') as f:
                f.write(java_pred)

            progress_bar.progress((i + 1) / num_samples)
        
        st.success("Translation complete!")

with tab2:
    st.header("Evaluation Metrics")
    
    if st.button("Run Evaluation"):
        with st.spinner("Calculating BLEU score..."):
            results = evaluate_translations(num_samples)
        
        if results:
            st.metric(label="BLEU Score", value=round(results['score'], 2))
            
            st.subheader("Detailed Results")
            st.json(results)
        else:
            st.warning("No predictions found. Please run translation first.")
