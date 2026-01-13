import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page Setup 
st.set_page_config(page_title="Infox| Fake News Detector", page_icon="⚖️", layout="wide")

# Custom CSS For Branding 
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stTextArea textarea { font-size: 1.1rem !important; border-radius: 10px !important; border: 1px solid #d1d1d1 !important; }
    .prediction-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .real { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .fake { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)

#  Asset Loading 
@st.cache_resource
def load_data():
    nltk.download('stopwords')
    model = pickle.load(open('model.pkl', 'rb'))
    vector = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vector

model, vector = load_data()
ps = PorterStemmer()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2540/2540832.png", width=100)
    st.title("Infox")
    st.info("This AI analyzes linguistics patterns to determine if a news story is likely fabricated or factual.")
    st.divider()
    st.markdown("### How to use:")
    st.write("1. Copy a news article text.\n2. Paste it in the main box.\n3. Click on 'Analyze'.")
    st.caption("v1.0.4 | Powered by Scikit-Learn")

# Main UI
st.title(" 🔍 Infox : Fake News Detection")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("📰 Article Content", height=350, placeholder="Paste article text here...")
    predict_btn = st.button("Analyze Authenticity", use_container_width=True)

with col2:
    st.subheader("Analysis Metrics")
    if user_input:
        words = len(user_input.split())
        chars = len(user_input)
        st.metric("Word Count", words)
        st.metric("Character Count", chars)
    else:
        st.write("Waiting for input...")

# Prediction Logic
if predict_btn:
    if user_input.strip():
        with st.spinner('Performing Forensic Analysis...'):
            # 1. Preprocessing 
            text = re.sub('[^a-zA-Z]', ' ', user_input).lower().split()
            processed = [ps.stem(w) for w in text if w not in set(stopwords.words('english'))]
            processed_str = ' '.join(processed)
            
            # 2. Vectorization
            input_data = vector.transform([processed_str])
            
            # 3. Prediction
            prediction = model.predict(input_data)
            if prediction[0] == 1: 
                st.markdown('<div class="prediction-card real">✅ RELIABLE SOURCE</div>', unsafe_allow_html=True)
                st.write("### Analysis")
                st.write("The linguistic structure aligns with verified factual reporting patterns.")
            else:
                st.markdown('<div class="prediction-card fake">🚨 UNRELIABLE / FAKE</div>', unsafe_allow_html=True)
                st.write("### Red Flags Detected")
                st.write("Our model detected patterns common in disinformation or hyper-partisan text.")
    else:
        st.warning("Please provide text to analyze.")