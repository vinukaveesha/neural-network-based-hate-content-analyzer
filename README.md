# Neural Network-Based Hate Content Analyzer

An advanced AI-powered hate speech detection system that uses LSTM and CNN neural networks for real-time text analysis and category classification. The system operates entirely offline without requiring external API dependencies.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Model Files](#model-files)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Neural Network-Based Hate Content Analyzer is a comprehensive solution for detecting and classifying hate speech in text content. It combines the power of LSTM neural networks for hate detection with CNN networks for category classification, providing accurate and fast analysis completely offline.

### Key Capabilities:
- **Real-time Analysis**: Instant hate speech detection
- **High Accuracy**: LSTM-based deep learning model for detection
- **Category Classification**: CNN-based automatic categorization of hate speech types
- **Confidence Scoring**: Probability scores with confidence levels
- **Batch Processing**: Analyze multiple texts simultaneously
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Export Features**: Download results as CSV files
- **Offline Processing**: No external API dependencies required

## Features

### Core Features
- **Single Text Analysis** - Analyze individual comments or posts
- **Batch Processing** - Process multiple texts at once
- **Probability Scoring** - Visual display showing hate speech probability
- **CNN Category Classification** - Automatic categorization using trained CNN model
- **Confidence Levels** - High/Medium/Low confidence indicators
- **Export Results** - Download analysis results as CSV
- **Multi-Model Architecture** - LSTM for detection, CNN for categorization

### Advanced Features
- **Dual Text Preprocessing** - Optimized preprocessing for each model type
- **Responsive Design** - Works on desktop and mobile devices
- **Real-time Processing** - Optimized for speed and performance
- **Privacy-Focused** - All processing done locally
- **Interactive Visualizations** - Detailed probability breakdowns

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** (Python package installer)
- **Git** (for cloning the repository)

**Note**: No external API keys are required. All processing is done locally using trained neural network models.

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/vinukaveesha/neural-network-based-hate-content-analyzer.git
cd neural-network-based-hate-content-analyzer
```

### Step 2: Create Virtual Environment
Creating a virtual environment is **highly recommended** to avoid package conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Requirements
Install all required dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
The application will automatically download required NLTK data on first run, but you can also download it manually:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Step 5: Verify Model Files
Ensure all required model files are present by running the test script:

```bash
python test_models.py
```

This will verify that all model files are correctly loaded and ready for use.

## Usage

### Running the Application

#### Method 1: Direct Command
```bash
# Ensure virtual environment is activated
streamlit run streamlit_app.py
```

#### Method 2: Custom Port
```bash
# Run on a specific port
streamlit run streamlit_app.py --server.port 8502
```

### Accessing the Application
1. Open your web browser
2. Navigate to `http://localhost:8501` (or the port shown in terminal)
3. Start analyzing text for hate speech!

### Using the Interface

#### Single Text Analysis
1. Select "Single Text" mode
2. Enter text in the input area
3. Click "Analyze Text"
4. View results including:
   - Hate speech detection (Yes/No)
   - Probability score and confidence level
   - Category classification (if hate speech detected)
   - All category probabilities breakdown

#### Batch Analysis
1. Select "Multiple Texts" mode
2. Enter multiple texts (one per line)
3. Click "Analyze All Texts"
4. View comprehensive results including:
   - Summary statistics
   - Category breakdown
   - Detailed results table
   - Export options for CSV download

#### Quick Test Examples
Use the built-in quick test examples in the sidebar to test the system with various text types.

## Model Architecture

### Dual-Model System

#### 1. LSTM Neural Network (Primary Detection)
- **Purpose**: Binary hate speech detection (hate vs. non-hate)
- **Architecture**: Bidirectional LSTM with attention mechanisms
- **Input**: Preprocessed text sequences (NLTK-based preprocessing)
- **Output**: Binary classification with probability score
- **File**: `models/best_lstm_model.h5` (12.6MB)
- **Vocabulary**: 17,079 unique tokens

#### 2. CNN Neural Network (Category Classification)  
- **Purpose**: Classify detected hate speech into specific categories
- **Architecture**: Multi-layer CNN with BatchNormalization and GlobalMaxPool1D
- **Input**: Simple preprocessed text (regex-based preprocessing)
- **Output**: 5-category classification with probability distribution
- **File**: `categorizing models/best_hate_classifier_cnn.h5` (12.6MB)
- **Vocabulary**: 18,080 unique tokens

### Category Classification System

The CNN model classifies hate speech into **5 distinct categories**:

1. **🔴 Gender-based Hate** (`sex_hate`)
   - Sexual harassment, gender discrimination
   
2. **🟠 General Hate Speech** (`other_hate`)  
   - General offensive content, personal attacks
   
3. **🟡 Sports-related Hate** (`sports_hate`)
   - Sports fan rivalry, team-based hatred
   
4. **🟢 Political Hate** (`politics_hate`)
   - Political party hatred, election-related toxicity
   
5. **🔵 Religious Hate** (`religious_hate`)
   - Religious discrimination, faith-based hatred

### Text Preprocessing Pipeline

#### LSTM Preprocessing (Enhanced):
1. **Emoji Removal**: Clean emojis using demoji library
2. **URL/Mention Cleaning**: Remove URLs, mentions, and hashtags
3. **Normalization**: Lowercase conversion and punctuation removal
4. **Tokenization**: Word-level tokenization with NLTK
5. **Stopword Removal**: Remove common English stopwords
6. **Lemmatization**: Word lemmatization for better accuracy

#### CNN Preprocessing (Simple):
1. **Lowercase Conversion**: Convert all text to lowercase
2. **URL/Mention Removal**: Remove URLs and mentions
3. **Character Filtering**: Keep only letters, numbers, and spaces
4. **Space Normalization**: Remove extra whitespace

### Training Details
- **Datasets**: Trained on curated hate speech datasets with category labels
- **LSTM Accuracy**: High precision for hate detection
- **CNN Accuracy**: Effective multi-class classification
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, BatchNormalization, and L2 regularization

## Model Files


All required model files are included in the repository and should be automatically loaded:

### LSTM Model Files (`models/` folder):
- `best_lstm_model.h5` (12.61 MB) - Trained LSTM model for hate detection
- `tokenizer.pickle` (0.64 MB) - LSTM tokenizer with 17,079 vocabulary

### CNN Model Files (`categorizing models/` folder):
- `best_hate_classifier_cnn.h5` (12.63 MB) - Trained CNN model for categorization
- `hate_classifier_tokenizer.pickle` (0.68 MB) - CNN tokenizer with 18,080 vocabulary  
- `hate_classifier_labels.pickle` (0.00 MB) - Label encoder for 5 categories

### Model Verification
Run the verification script to ensure all models are properly loaded:

```bash
python test_models.py
```

Expected output:
```
============================================================
HATE CONTENT DETECTION SYSTEM - MODEL TEST
============================================================
Testing model file existence...
✅ models/best_lstm_model.h5 (12.61 MB)
✅ models/tokenizer.pickle (0.64 MB)
✅ categorizing models/best_hate_classifier_cnn.h5 (12.63 MB)
✅ categorizing models/hate_classifier_tokenizer.pickle (0.68 MB)
✅ categorizing models/hate_classifier_labels.pickle (0.00 MB)

Testing pickle file loading...
✅ models/tokenizer.pickle - Tokenizer loaded (vocab size: 17079)
✅ categorizing models/hate_classifier_tokenizer.pickle - Tokenizer loaded (vocab size: 18080)
✅ categorizing models/hate_classifier_labels.pickle - Label encoder loaded (classes: ['other_hate', 'politics_hate', 'religious_hate', 'sex_hate', 'sports_hate'])

============================================================
✅ ALL TESTS PASSED!
✅ Models are ready for use in the Streamlit application.
============================================================
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Error
```
Error: Cannot load model file
```
**Solution**: Ensure all model files exist in the correct directories. Run `python test_models.py` to verify.

#### 2. NLTK Data Error
```
LookupError: Resource punkt not found
```
**Solution**: Run the NLTK download commands or restart the application to auto-download.

#### 3. TensorFlow Warnings
```
WARNING: TensorFlow optimizations enabled
```
**Solution**: These are informational warnings and don't affect functionality.

#### 4. Virtual Environment Issues
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Ensure virtual environment is activated and requirements are installed.

#### 5. Scikit-learn Version Warning
```
InconsistentVersionWarning: Trying to unpickle estimator
```
**Solution**: This warning is safe to ignore; the models will work correctly.

---
