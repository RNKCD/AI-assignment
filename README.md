# Emotion Detection & Support System

A complete 3-agent AI pipeline for emotion detection and supportive suggestion generation, built with Python and Streamlit.

## 🏗️ System Architecture

This project implements a 3-agent pipeline:

1. **NLP Agent** - Converts text to embeddings using Voyage AI API (`voyage-lite-02-instruct`)
2. **Emotion Detection Agent** - RoBERTa-based emotion classifier:
   - j-hartmann/emotion-english-distilroberta-base
   - Pre-trained model for emotion classification (6 basic emotions)
3. **Suggestion Agent** - Generates supportive advice using Together API or Hugging Face as backup

## 📁 Project Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── nlp_agent.py          # Embedding generator (Voyage AI)
│   ├── emotion_agent.py      # Emotion classifier (RoBERTa)
│   └── suggestion_agent.py  # Suggestion generator (Llama3 API)
├── app.py                    # Main Streamlit application
├── config.py                 # API keys configuration
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🚀 Quick Deploy to Streamlit Cloud (Easiest!)

1. **Push your code to GitHub** (create a new repository and push all files)

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and sign in with GitHub

3. **Click "New app"** and:
   - Select your repository
   - Set **Main file path** to: `app.py`
   - Click "Deploy!"

4. **Add API keys** (in Streamlit Cloud dashboard):
   - Go to your app → Settings → Secrets
   - Add your API keys as secrets:
     ```
     TOGETHER_API_KEY=your_key_here
     VOYAGE_API_KEY=your_key_here
     ```
   - Or use the sidebar in the app to enter keys

5. **Done!** Your app will be live at `https://your-app-name.streamlit.app`

---

## 🚀 Local Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys (choose one method):**
   
   **Option A: Set in config.py (easiest)**
   ```python
   # Edit config.py and set your keys:
   VOYAGE_API_KEY = "your_voyage_api_key_here"
   OPENROUTER_API_KEY = "your_openrouter_api_key_here"  # FREE! (Recommended for better responses)
   ```
   
   **Option B: Set environment variables**
   ```bash
   export VOYAGE_API_KEY="your_voyage_api_key"
   export OPENROUTER_API_KEY="your_openrouter_api_key"  # FREE! (Recommended)
   ```
   
   **Option C: Enter API keys in the Streamlit app sidebar**
   
   Get your API keys:
   - Voyage AI: https://www.voyageai.com/
   - OpenRouter: https://openrouter.ai/keys (FREE tier available! Recommended for better contextual responses)

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

The app will open in your default web browser at `http://localhost:8501`

## 💻 Usage

1. Enter your thoughts or feelings in the text input box
2. Click "Analyze Emotion" button
3. View the detected emotion, confidence score, and supportive suggestions

## 🔧 Technical Details

### NLP Agent
- Model: `voyage-lite-02-instruct` via Voyage AI API
- Output: High-quality embedding vectors
- Purpose: Convert natural language text to numerical representations
- API: Requires Voyage AI API key

### Emotion Detection Agent
- Model: Pre-trained RoBERTa-based emotion classifier
  - `j-hartmann/emotion-english-distilroberta-base`: 6 basic emotions (joy, sadness, anger, fear, surprise, disgust)
- Input: Raw text (no embeddings needed)
- Output: 6 emotion classes (happiness, sadness, anger, anxiety, frustration, depression)
- Method: Maps 6 basic emotions to standard emotion set
- Framework: HuggingFace Transformers (runs locally)

### Suggestion Agent
- Model: `Mixtral-8x7B-Instruct-v0.1' via TOGETHER API (or Mistral-7B via Hugging Face as backup)
- Purpose: Generate contextual, supportive, non-medical advice sentences
- Framework: API-based
- Note: Uses user's actual message for context-aware responses. OpenRouter provides better quality responses with free tier

## 📝 Notes

- **API Keys Required**: 
  - Voyage AI API key for embeddings
  - OpenRouter API key for suggestions (FREE tier available! Recommended for better responses)
  - Hugging Face API key as backup option
  - Keys can be set in config.py, via environment variables, or in the app sidebar
- **Emotion Model**: Runs locally (no API needed)
- **First Run**: Emotion model will be downloaded automatically (~300MB)
- **API Costs**: 
  - Voyage AI: Check their pricing
  - OpenRouter: FREE tier available! Get key at https://openrouter.ai/keys (recommended for better contextual responses)
  - Hugging Face: FREE! Available as backup option
- **Internet Required**: API calls require internet connection
- **Context-Aware Responses**: The system now uses your actual message for more relevant, contextual suggestions

## 🎯 Features

- Clean, user-friendly Streamlit interface
- Real-time emotion detection
- Confidence scores for predictions
- Emotion-specific supportive suggestions
- Emoji indicators for emotions
- Fully modular, well-commented code

## ⚠️ Important Disclaimer

This system is for demonstration purposes only. It provides non-medical, supportive suggestions and should not be used as a substitute for professional mental health care. For serious emotional or mental health concerns, please consult with qualified healthcare providers.

## 📦 Dependencies

- `streamlit`: Web application framework
- `sentence-transformers`: Text embedding generation
- `torch`: Deep learning framework (PyTorch)
- `transformers`: HuggingFace transformers library
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning utilities

## 🔄 Future Enhancements

- Train the emotion detection model on a real emotion dataset
- Add model persistence and checkpoint saving
- Implement batch processing for multiple texts
- Add emotion history tracking
- Include more detailed probability distributions

## 📄 License

This project is provided as-is for educational and demonstration purposes.


