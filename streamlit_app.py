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
import demoji
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Define hate class mappings from the categorization notebook
HATE_CLASS_MAPPING = {
    'sex_hate': 0,
    'other_hate': 1,
    'sports_hate': 2,
    'politics_hate': 3,
    'religious_hate': 4
}

REVERSE_HATE_CLASS_MAPPING = {v: k for k, v in HATE_CLASS_MAPPING.items()}

def simple_preprocess(text):
    """Simple preprocessing for categorization model"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    
    # Keep only letters, numbers and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def preprocess_text_cnn(text):
    """Enhanced preprocessing optimized for CNN model from notebook"""
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
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    processed_tokens = []

    for token in tokens:
        if token and len(token) > 1 and token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)

    return " ".join(processed_tokens) if processed_tokens else ''

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
    return True

# Text preprocessing function (removed as we're using CNN preprocessing)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained CNN model and tokenizer from new_models folder, plus categorization models"""
    try:
        # Load CNN model for hate detection
        model = load_model('new_models/best_cnn_model.h5')
        with open('new_models/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load categorization CNN model
        categorizer_model = load_model('categorizing models/best_hate_classifier_cnn.h5')
        with open('categorizing models/hate_classifier_tokenizer.pickle', 'rb') as f:
            categorizer_tokenizer = pickle.load(f)
        with open('categorizing models/hate_classifier_labels.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
            
        return model, tokenizer, categorizer_model, categorizer_tokenizer, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# CNN Hate Speech Detector Class with Categorization
class CNNHateSpeechDetector:
    def __init__(self, model, tokenizer, categorizer_model, categorizer_tokenizer, label_encoder, threshold=0.5, max_len=100):
        self.model = model
        self.tokenizer = tokenizer
        self.categorizer_model = categorizer_model
        self.categorizer_tokenizer = categorizer_tokenizer
        self.label_encoder = label_encoder
        self.threshold = threshold
        self.max_len = max_len
        
        # Initialize NLTK components
        initialize_nltk()
        
        # Get class names for categorizer
        self.class_names = list(self.label_encoder.classes_)
    
    def preprocess_cnn(self, text):
        """Apply CNN preprocessing for hate detection"""
        return preprocess_text_cnn(text)
    
    def preprocess_simple(self, text):
        """Apply simple preprocessing for categorization"""
        return simple_preprocess(text)
    
    def predict_with_confidence(self, text):
        """Predict hate speech with confidence estimation"""
        # Preprocess for CNN model
        cleaned_text = self.preprocess_cnn(text)
        
        if not cleaned_text:
            return False, 0.0, "Low", cleaned_text, "No content after preprocessing", None
        
        # Convert to sequence for CNN
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # Get CNN prediction
        cnn_prob = self.model.predict(padded_sequence, verbose=0)[0][0]
        is_hate = cnn_prob > self.threshold
        
        # Calculate confidence based on probability
        if cnn_prob > 0.8 or cnn_prob < 0.2:
            confidence = "High"
        elif cnn_prob > 0.6 or cnn_prob < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        detection_method = "CNN model"
        category_result = None
        
        # If it's hate speech, classify the category
        if is_hate:
            category_result = self.classify_hate_category(text)
        
        return is_hate, float(cnn_prob), confidence, cleaned_text, detection_method, category_result
    
    def classify_hate_category(self, text):
        """Classify hate speech category using the CNN categorizer model"""
        try:
            # Preprocess for categorizer
            cleaned_text = self.preprocess_simple(text)
            
            if not cleaned_text:
                return {
                    "category": "other_hate",
                    "confidence": 0.0,
                    "explanation": "No content after preprocessing",
                    "all_probabilities": {}
                }
            
            # Convert to sequence for categorizer
            sequence = self.categorizer_tokenizer.texts_to_sequences([cleaned_text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_len, padding='post')
            
            # Predict category
            prob_dist = self.categorizer_model.predict(padded_sequence, verbose=0)[0]
            predicted_idx = np.argmax(prob_dist)
            predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
            confidence_score = float(prob_dist[predicted_idx])
            
            # Create probability dictionary
            prob_dict = {self.class_names[i]: float(prob_dist[i]) 
                        for i in range(len(self.class_names))}
            
            # Convert category names to more readable format
            category_display_names = {
                'sex_hate': 'Gender-based Hate',
                'other_hate': 'General Hate Speech',
                'sports_hate': 'Sports-related Hate',
                'politics_hate': 'Political Hate',
                'religious_hate': 'Religious Hate'
            }
            
            display_category = category_display_names.get(predicted_class, predicted_class)
            
            return {
                "category": display_category,
                "raw_category": predicted_class,
                "confidence": confidence_score,
                "explanation": f"Classified as {display_category} with {confidence_score:.1%} confidence",
                "all_probabilities": prob_dict,
                "api_failed": False
            }
            
        except Exception as e:
            return {
                "category": "Classification Error",
                "confidence": 0.0,
                "explanation": f"Error in classification: {str(e)}",
                "all_probabilities": {},
                "api_failed": True
            }

# Main Streamlit App
def main():
    # Header
    st.title("Hate Content Detection System")
    st.markdown("### AI-powered hate content detection using CNN model with categorization")
    st.markdown("""
    **Models Used:**
    - **CNN Model**: Convolutional Neural Network for hate speech detection
    - **CNN Categorizer**: Hate speech category classification (5 categories)
    """)
    st.markdown("---")
    
    # Load models and tokenizers
    model, tokenizer, categorizer_model, categorizer_tokenizer, label_encoder = load_model_and_tokenizer()
    
    if any(x is None for x in [model, tokenizer, categorizer_model, categorizer_tokenizer, label_encoder]):
        st.error("Failed to load models or tokenizers. Please check if the model files exist in the new_models and categorizing models folders.")
        return
    
    # Initialize detector
    detector = CNNHateSpeechDetector(model, tokenizer, categorizer_model, categorizer_tokenizer, label_encoder)
   
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
            "Women belong in the kitchen",
            "All politicians are corrupt thieves",
            "Those religious fanatics are crazy"
        ]
        
        for i, example in enumerate(sample_examples):
            if st.button(f"Test: {example[:30]}...", key=f"example_{i}"):
                with st.spinner("Analyzing..."):
                    analyze_single_text(detector, example)

def analyze_single_text(detector, text):
    """Analyze a single text input"""
    # Get prediction
    is_hate, prob, confidence, cleaned_text, detection_method, category_result = detector.predict_with_confidence(text)
    
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
    if is_hate and category_result:
        st.subheader("Category Classification")
        
        if category_result.get("api_failed", False):
            # Classification failed
            st.warning("**Category Classification Failed**")
            st.markdown(f"**Error:** {category_result['explanation']}")
        else:
            # Classification successful
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Category:** {category_result['category']}")
            with col2:
                st.info(f"**Confidence:** {category_result['confidence']:.3f}")
            
            st.markdown(f"**Explanation:** {category_result['explanation']}")
            
            # Show all probabilities if available
            if category_result.get('all_probabilities'):
                st.subheader("All Category Probabilities")
                prob_df = pd.DataFrame([
                    {"Category": k, "Probability": f"{v:.3f}"} 
                    for k, v in sorted(category_result['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
    
    # Text preprocessing details
    with st.expander("Text Preprocessing Details"):
        st.write("**Original Text:**")
        st.code(text)
        st.write("**Cleaned Text:**")
        st.code(cleaned_text if cleaned_text else "Empty after preprocessing")

def analyze_multiple_texts(detector, texts):
    """Analyze multiple texts"""
    st.subheader("Batch Analysis Results")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        is_hate, prob, confidence, cleaned_text, detection_method, category_result = detector.predict_with_confidence(text)
        
        # Get category if it's hate speech
        category = "N/A"
        category_confidence = 0.0
        if is_hate and category_result and not category_result.get("api_failed", False):
            category = category_result.get("category", "Unknown")
            category_confidence = category_result.get("confidence", 0.0)
        
        result = {
            "Text": text[:50] + "..." if len(text) > 50 else text,
            "Is Hate Speech": "Yes" if is_hate else "No",
            "Probability": f"{prob:.3f}",
            "Confidence": confidence,
            "Category": category,
            "Category Confidence": f"{category_confidence:.3f}" if category_confidence > 0 else "N/A",
            "Detection Method": detection_method,
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
    categorized_count = len([r for r in results if r["Category"] != "N/A"])
    
    with col1:
        st.metric("Total Texts", total_count)
    with col2:
        st.metric("Hate Speech Detected", hate_count)
    with col3:
        st.metric("Clean Texts", total_count - hate_count)
    with col4:
        st.metric("Successfully Categorized", categorized_count)
    
    # Additional stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hate Speech Rate", f"{(hate_count/total_count)*100:.1f}%")
    with col2:
        st.metric("Categorization Rate", f"{(categorized_count/max(hate_count, 1))*100:.1f}%")
    with col3:
        accuracy_metric = f"{((total_count - hate_count + categorized_count)/total_count)*100:.1f}%"
        st.metric("Overall Success", accuracy_metric)
    
    # Display results table
    st.subheader("Detailed Results")
    
    # Add filtering options
    filter_option = st.selectbox(
        "Filter results:",
        ["All", "Hate Speech Only", "Clean Text Only", "Categorized Only"]
    )
    
    if filter_option == "Hate Speech Only":
        filtered_df = df[df["Is Hate Speech"] == "Yes"]
    elif filter_option == "Clean Text Only":
        filtered_df = df[df["Is Hate Speech"] == "No"]
    elif filter_option == "Categorized Only":
        filtered_df = df[df["Category"] != "N/A"]
    else:
        filtered_df = df
    
    # Display filtered results
    st.dataframe(
        filtered_df.drop("Full Text", axis=1), 
        use_container_width=True,
        hide_index=True
    )
    
    # Show category breakdown if there are hate speech results
    if hate_count > 0:
        st.subheader("Category Breakdown")
        category_counts = {}
        for result in results:
            if result["Is Hate Speech"] == "Yes" and result["Category"] != "N/A":
                category = result["Category"]
                category_counts[category] = category_counts.get(category, 0) + 1
        
        if category_counts:
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / hate_count) * 100
                st.write(f"**{category}:** {count} texts ({percentage:.1f}% of hate speech)")
        else:
            st.write("No successful categorizations available.")
    

if __name__ == "__main__":
    main()
