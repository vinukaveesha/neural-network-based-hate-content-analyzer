# 🛡️ Neural Network-Based Hate Content Analyzer

An advanced AI-powered hate speech detection system that uses LSTM neural networks and Google Gemini AI for real-time text analysis and category classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [API Integration](#api-integration)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The Neural Network-Based Hate Content Analyzer is a comprehensive solution for detecting and classifying hate speech in text content. It combines the power of LSTM neural networks with Google's Gemini AI to provide accurate detection with detailed category classification.

### Key Capabilities:
- **Real-time Analysis**: Instant hate speech detection
- **High Accuracy**: LSTM-based deep learning model
- **Category Classification**: Automatic categorization of hate speech types
- **Confidence Scoring**: Probability scores with confidence levels
- **Batch Processing**: Analyze multiple texts simultaneously
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Export Features**: Download results as CSV files

## ✨ Features

### Core Features
- 🔍 **Single Text Analysis** - Analyze individual comments or posts
- 📊 **Batch Processing** - Process multiple texts at once
- 🎯 **Probability Scoring** - Visual gauge showing hate speech probability
- 🏷️ **AI Category Classification** - Automatic categorization using Gemini AI
- 📈 **Confidence Levels** - High/Medium/Low confidence indicators
- 💾 **Export Results** - Download analysis results as CSV
- ⚙️ **Adjustable Threshold** - Customize detection sensitivity

### Advanced Features
- 🔧 **Text Preprocessing** - Advanced text cleaning and normalization
- 📱 **Responsive Design** - Works on desktop and mobile devices
- 🚀 **Real-time Processing** - Optimized for speed and performance
- 🔒 **Secure Configuration** - Environment variable-based API key management
- 📊 **Interactive Visualizations** - Plotly-based charts and gauges

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **pip** (Python package installer)
- **Git** (for cloning the repository)
- **Google Gemini API Key** (for category classification)

## 📦 Installation

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

## 🔑 Configuration

### Step 1: Get Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for the next step

### Step 2: Create Environment File
Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env  # On Windows: type nul > .env
```

### Step 3: Add API Key to .env
Open the `.env` file and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Important**: Never commit your API key to version control. The `.env` file is already included in `.gitignore`.

## 🚀 Usage

### Running the Application

#### Method 1: Direct Command
```bash
# Ensure virtual environment is activated
python -m streamlit run streamlit_app.py
```

#### Method 2: Using the Launcher Script (Windows)
```bash
# Make sure the script is executable
run_app.bat
```

#### Method 3: Custom Port
```bash
# Run on a specific port
python -m streamlit run streamlit_app.py --server.port 8502
```

### Accessing the Application
1. Open your web browser
2. Navigate to `http://localhost:8501` (or the port shown in terminal)
3. Start analyzing text for hate speech!

### Using the Interface

#### Single Text Analysis
1. Select "Single Text" mode
2. Enter text in the input area
3. Click "🔍 Analyze Text"
4. View results including probability, confidence, and category

#### Batch Analysis
1. Select "Multiple Texts" mode
2. Enter multiple texts (one per line)
3. Click "🔍 Analyze All Texts"
4. View summary statistics and detailed results
5. Download results as CSV if needed

#### Settings
- **Detection Threshold**: Adjust sensitivity using the sidebar slider
- **Quick Examples**: Test with pre-built sample texts
- **Export Options**: Download batch results for further analysis

## 📁 Project Structure

```
neural-network-based-hate-content-analyzer/
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── streamlit_app.py             # Main Streamlit application
├── sixth-kaggle.ipynb           # Jupyter notebook with model training
├── README.md                    # This file
├── models/                      # Model files directory
│   ├── best_lstm_model.h5       # Trained LSTM model
│   └── tokenizer.pickle         # Text tokenizer
└── venv/                        # Virtual environment (created by you)
```

## 🧠 Model Architecture

### LSTM Neural Network
- **Architecture**: Bidirectional LSTM with attention mechanisms
- **Input**: Preprocessed text sequences
- **Output**: Binary classification (hate speech vs. clean text)
- **Features**: 
  - Text preprocessing with NLTK
  - Tokenization and sequence padding
  - Confidence estimation based on prediction probability

### Text Preprocessing Pipeline
1. **Emoji Removal**: Clean emojis using demoji library
2. **URL/Mention Cleaning**: Remove URLs, mentions, and hashtags
3. **Normalization**: Lowercase conversion and punctuation removal
4. **Tokenization**: Word-level tokenization
5. **Stopword Removal**: Remove common English stopwords
6. **Lemmatization**: Word lemmatization for better accuracy

### Training Details
- **Dataset**: Trained on curated hate speech datasets
- **Accuracy**: High precision and recall scores
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout and L2 regularization to prevent overfitting

## 🤖 API Integration

### Google Gemini AI
- **Purpose**: Category classification of detected hate speech
- **Categories**:
  - Sexual harassment
  - Religious hate
  - Political hate
  - Racial discrimination
  - Gender-based hate
  - Other hate speech
- **Features**: Confidence scoring and explanation generation

### Rate Limiting
- The system handles API rate limits gracefully
- Automatic retry mechanisms for failed requests
- Fallback options when API is unavailable

## 📊 Screenshots

### Main Interface
![Main Interface](https://via.placeholder.com/800x400?text=Main+Interface+Screenshot)

### Analysis Results
![Analysis Results](https://via.placeholder.com/800x400?text=Analysis+Results+Screenshot)

### Batch Processing
![Batch Processing](https://via.placeholder.com/800x400?text=Batch+Processing+Screenshot)

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Error
```
Error: Cannot load model file
```
**Solution**: Ensure model files exist in the `models/` directory.

#### 2. NLTK Data Error
```
LookupError: Resource punkt not found
```
**Solution**: Run the NLTK download commands or restart the application.

#### 3. Gemini API Error
```
Error: API key not configured
```
**Solution**: Check your `.env` file and ensure the API key is correct.

#### 4. Virtual Environment Issues
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Ensure virtual environment is activated and requirements are installed.

### Performance Tips
- Use batch processing for multiple texts
- Adjust detection threshold based on your use case
- Close other applications to free up memory for large batches

## 🛡️ Security Considerations

- **API Key Security**: Never commit API keys to version control
- **Input Validation**: All user inputs are sanitized
- **Rate Limiting**: Respect API rate limits to avoid blocking
- **Data Privacy**: No user data is stored permanently

## 📈 Future Enhancements

- [ ] Multi-language support
- [ ] Real-time streaming analysis
- [ ] Custom model training interface
- [ ] Integration with social media APIs
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] API endpoint for integration

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/neural-network-based-hate-content-analyzer.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow/Keras**: For the deep learning framework
- **Streamlit**: For the web application framework
- **Google Gemini AI**: For advanced text classification
- **NLTK**: For natural language processing
- **Plotly**: For interactive visualizations
- **Open Source Community**: For various tools and libraries

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/vinukaveesha/neural-network-based-hate-content-analyzer/issues) page
2. Create a new issue with detailed description
3. Join our [Discussions](https://github.com/vinukaveesha/neural-network-based-hate-content-analyzer/discussions)

## 📊 Project Status

- ✅ **Core Features**: Complete
- ✅ **Web Interface**: Complete
- ✅ **API Integration**: Complete
- ✅ **Documentation**: Complete
- 🔄 **Testing**: In Progress
- 🔄 **Deployment**: In Progress

---

**Made with ❤️ by [vinukaveesha](https://github.com/vinukaveesha)**

⭐ If you find this project useful, please consider giving it a star!
