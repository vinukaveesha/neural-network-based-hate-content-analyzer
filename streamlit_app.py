import streamlit as st
import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
import demoji
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Hate Content Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    # Download demoji codes if not present
    try:
        demoji.download_codes()
    except:
        pass

# Initialize NLTK components
@st.cache_resource
def initialize_nltk():
    download_nltk_data()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

# Text preprocessing function
def preprocess_text_lstm(text, stop_words, lemmatizer):
    """Enhanced preprocessing optimized for LSTM model"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text)
    
    # Remove emojis
    text = demoji.replace(text, '')
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Clean and normalize
    text = re.sub(r'\s+', ' ', text.strip().lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    
    if not text.strip():
        return ''
    
    # Tokenization and lemmatization
    tokens = word_tokenize(text)
    processed_tokens = []
    
    for token in tokens:
        if token and len(token) > 1 and token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)
    
    return " ".join(processed_tokens) if processed_tokens else ''

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        model = load_model('models/best_lstm_model.h5')
        with open('models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Hate Speech Detector Class
class StreamlitHateSpeechDetector:
    def __init__(self, model, tokenizer, threshold=0.5, max_len=100):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_len = max_len
        
        # Initialize NLTK components
        self.stop_words, self.lemmatizer = initialize_nltk()
        
        # Initialize Gemini for category classification
        self.gemini = self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini API"""
        try:
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                return genai.GenerativeModel('gemini-1.5-flash')
            else:
                return None
        except Exception:
            return None
    
    def preprocess(self, text):
        """Apply same preprocessing as training"""
        return preprocess_text_lstm(text, self.stop_words, self.lemmatizer)
    
    def predict_with_confidence(self, text):
        """Predict with confidence estimation"""
        cleaned_text = self.preprocess(text)
        
        if not cleaned_text:
            return False, 0.0, "Low", cleaned_text
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # Get prediction
        prob = self.model.predict(padded_sequence, verbose=0)[0][0]
        is_hate = prob > self.threshold
        
        # Calculate confidence based on distance from threshold
        distance_from_threshold = abs(prob - self.threshold)
        if distance_from_threshold > 0.3:
            confidence = "High"
        elif distance_from_threshold > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return is_hate, float(prob), confidence, cleaned_text
    
    def classify_hate_category(self, text):
        """Classify hate speech category using Gemini"""
        if not self.gemini:
            return {
                "category": "Classification not available",
                "confidence": 0.0,
                "explanation": "Gemini API not configured"
            }
        
        candidate_labels = [
            "Sexual harassment",
            "Religious hate",
            "Political hate", 
            "Racial discrimination",
            "Gender-based hate",
            "Other hate speech"
        ]
        
        prompt = f"""Classify the following text into exactly one of these categories:
        {", ".join(candidate_labels)}.
        Text: "{text}"
        Return your response in JSON format with these keys:
        - "category": the most appropriate category name
        - "confidence": your confidence score between 0-1
        - "explanation": brief explanation (1 sentence)
        """
        
        try:
            response = self.gemini.generate_content(prompt)
            json_str = response.text.replace('```json', '').replace('```', '').strip()
            result = eval(json_str)
            return result
        except Exception as e:
            return {
                "category": "Classification failed",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}"
            }

# Main Streamlit App
def main():
    # Header
    st.title("Hate Content Detection System")
    st.markdown("### Advanced AI-powered hate content detection with category classification")
    st.markdown("---")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("❌ Failed to load model or tokenizer. Please check if the model files exist.")
        return
    
    # Initialize detector
    detector = StreamlitHateSpeechDetector(model, tokenizer)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.01)
    detector.threshold = threshold
    
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    This system uses an LSTM neural network to detect hate speech in text.
    
    **Features**
    - Real-time hate speech detection
    - Confidence scoring
    - Category classification
    - Text preprocessing analysis
    """)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Text")
        
        # Text input options
        input_method = st.radio("Choose input method:", ["Single Text", "Multiple Texts"])
        
        if input_method == "Single Text":
            user_input = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your text here...",
                help="Enter any text you want to analyze for hate speech"
            )
            
            if st.button("Analyze Text", type="primary"):
                if user_input.strip():
                    with st.spinner("Analyzing text..."):
                        analyze_single_text(detector, user_input)
                else:
                    st.warning("Please enter some text to analyze.")
        
        else:  # Multiple Texts
            st.subheader("Batch Analysis")
            sample_texts = st.text_area(
                "Enter multiple texts (one per line):",
                height=200,
                placeholder="Enter each text on a new line...",
                help="Enter multiple texts separated by new lines"
            )
            
            if st.button("Analyze All Texts", type="primary"):
                if sample_texts.strip():
                    texts = [text.strip() for text in sample_texts.split('\n') if text.strip()]
                    if texts:
                        with st.spinner("Analyzing texts..."):
                            analyze_multiple_texts(detector, texts)
                    else:
                        st.warning("Please enter valid texts to analyze.")
                else:
                    st.warning("Please enter some texts to analyze.")
    
    with col2:
        st.subheader("Quick Stats")
        
        # Sample texts for quick testing
        st.markdown("**Quick Test Examples**")
        sample_examples = [
            "I love this beautiful day!",
            "You're such an idiot!",
            "This food is terrible",
            "I hate when people don't clean up",
            "You people are disgusting"
        ]
        
        for i, example in enumerate(sample_examples):
            if st.button(f"Test: {example[:30]}...", key=f"example_{i}"):
                with st.spinner("Analyzing..."):
                    analyze_single_text(detector, example)

def analyze_single_text(detector, text):
    """Analyze a single text input"""
    # Get prediction
    is_hate, prob, confidence, cleaned_text = detector.predict_with_confidence(text)
    
    # Display results
    st.subheader("Analysis Results")
    
    # Create columns for results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if is_hate:
            st.error("🚨 **HATE SPEECH DETECTED**")
        else:
            st.success("✅ **NO HATE SPEECH**")
    
    with col2:
        st.metric("Probability", f"{prob:.3f}")
    
    with col3:
        st.metric("Confidence", confidence)
    
    
    # Category classification for hate speech
    if is_hate:
        st.subheader("🏷️ Category Classification")
        with st.spinner("Classifying category..."):
            category_result = detector.classify_hate_category(text)
        
        if category_result["category"] != "Classification not available":
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Category:** {category_result['category']}")
            with col2:
                st.info(f"**Confidence:** {category_result['confidence']:.3f}")
            
            st.markdown(f"**Explanation:** {category_result['explanation']}")
    
    # Text preprocessing details
    with st.expander("🔍 Text Preprocessing Details"):
        st.write("**Original Text:**")
        st.code(text)
        st.write("**Cleaned Text:**")
        st.code(cleaned_text if cleaned_text else "Empty after preprocessing")
        st.write("**Preprocessing Steps:**")
        st.markdown("""
        1. Remove emojis
        2. Remove URLs, mentions, hashtags
        3. Convert to lowercase
        4. Remove punctuation
        5. Tokenization
        6. Remove stopwords
        7. Lemmatization
        """)

def analyze_multiple_texts(detector, texts):
    """Analyze multiple texts"""
    st.subheader("📊 Batch Analysis Results")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        is_hate, prob, confidence, cleaned_text = detector.predict_with_confidence(text)
        
        result = {
            "Text": text[:50] + "..." if len(text) > 50 else text,
            "Is Hate Speech": "Yes" if is_hate else "No",
            "Probability": f"{prob:.3f}",
            "Confidence": confidence,
            "Full Text": text
        }
        results.append(result)
        progress_bar.progress((i + 1) / len(texts))
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    hate_count = len([r for r in results if r["Is Hate Speech"] == "Yes"])
    total_count = len(results)
    
    with col1:
        st.metric("Total Texts", total_count)
    with col2:
        st.metric("Hate Speech Detected", hate_count)
    with col3:
        st.metric("Clean Texts", total_count - hate_count)
    with col4:
        st.metric("Hate Speech Rate", f"{(hate_count/total_count)*100:.1f}%")
    
    # Display results table
    st.subheader("Detailed Results")
    
    # Add filtering options
    filter_option = st.selectbox(
        "Filter results:",
        ["All", "Hate Speech Only", "Clean Text Only"]
    )
    
    if filter_option == "Hate Speech Only":
        filtered_df = df[df["Is Hate Speech"] == "Yes"]
    elif filter_option == "Clean Text Only":
        filtered_df = df[df["Is Hate Speech"] == "No"]
    else:
        filtered_df = df
    
    # Display filtered results
    st.dataframe(
        filtered_df.drop("Full Text", axis=1), 
        use_container_width=True,
        hide_index=True
    )
    

if __name__ == "__main__":
    main()
