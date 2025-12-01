# AI-Powered Mental Health Support Chat System

A complete 3-agent AI pipeline for emotion detection and supportive suggestion generation, built with Python and Streamlit. This system provides real-time emotion analysis and contextual, empathetic responses through a conversational chat interface.

## 🏗️ System Architecture

This project implements a 3-agent pipeline:

1. **NLP Agent** - Converts text to embeddings using Voyage AI API (`voyage-lite-02-instruct`)
2. **Emotion Detection Agent** - RoBERTa-based emotion classifier:
   - `j-hartmann/emotion-english-distilroberta-base`
   - Pre-trained model for emotion classification (6 basic emotions)
   - Runs locally (no API needed)
3. **Suggestion Agent** - Generates supportive advice using Together AI API (Mixtral-8x7B - FREE tier!) with enhanced fallback system

## 📁 Project Structure

```
.
├── agents/
│   ├── __init__.py
│   ├── nlp_agent.py          # Embedding generator (Voyage AI)
│   ├── emotion_agent.py      # Emotion classifier (RoBERTa - Local)
│   └── suggestion_agent.py   # Response generator (Together AI + Fallback)
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
   TOGETHER_API_KEY = "your_together_api_key_here"  # FREE! ($25 free credits, no credit card)
   ```
   
   **Option B: Set environment variables**
   ```bash
   export VOYAGE_API_KEY="your_voyage_api_key"
   export TOGETHER_API_KEY="your_together_api_key"  # FREE!
   ```
   
   **Option C: Enter API keys in the Streamlit app sidebar**
   
   Get your API keys:
   - Voyage AI: https://www.voyageai.com/
   - Together AI: https://api.together.xyz/settings/api-keys (FREE tier: $25 free credits, no credit card needed!)

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

The app will open in your default web browser at `http://localhost:8501`

## 💻 Usage

1. Enter your thoughts or feelings in the chat input box
2. Click "Send" or press Enter
3. View the detected emotion, confidence score, and supportive suggestions
4. Continue the conversation - the system maintains context across messages

## 🔧 Technical Details

### NLP Agent
- **Model**: `voyage-lite-02-instruct` via Voyage AI API
- **Output**: High-quality embedding vectors (1024 dimensions)
- **Purpose**: Convert natural language text to numerical representations
- **API**: Requires Voyage AI API key

### Emotion Detection Agent
- **Model**: Pre-trained RoBERTa-based emotion classifier
  - `j-hartmann/emotion-english-distilroberta-base`: 6 basic emotions (joy, sadness, anger, fear, surprise, disgust)
- **Input**: Raw text (no preprocessing needed)
- **Output**: 
  - Primary emotion label (1 of 6 emotions)
  - Confidence score (0-100%)
  - Top 3 emotions with probabilities
- **Emotions Detected**:
  1. Happiness (😊)
  2. Sadness (😢)
  3. Anger (😠)
  4. Anxiety (😰)
  5. Frustration (😤)
  6. Depression (😔)
- **Framework**: HuggingFace Transformers (runs locally)
- **Special Features**:
  - Context-aware keyword analysis
  - Handles sarcasm and context
  - Robust error handling
  - Runs on CPU or GPU (auto-detects)

### Suggestion Agent
- **Model**: `mistralai/Mixtral-8x7B-Instruct-v0.1` via Together AI API (primary)
- **Architecture**: Mixtral-8x7B (Mixture of Experts, 8x7B parameters)
- **Purpose**: Generate contextual, supportive, non-medical advice
- **Framework**: API-based (Together AI - FREE tier available!)
- **Features**:
  - Context-aware responses (references user's actual message)
  - Conversation history integration (last 4 messages)
  - Enhanced fallback system (works without API)
  - Message alternation validation (for API compatibility)
- **Fallback Options**:
  - Together AI (primary) - FREE tier with $25 credits
  - OpenRouter (optional backup)
  - Enhanced rule-based fallback (works offline)

## 📝 Notes

- **API Keys Required**: 
  - Voyage AI API key for embeddings
  - Together AI API key for suggestions (FREE tier available! $25 free credits, no credit card needed)
  - Keys can be set in config.py, via environment variables, or in the app sidebar
- **Emotion Model**: Runs locally (no API needed)
- **First Run**: Emotion model will be downloaded automatically (~300MB)
- **API Costs**: 
  - Voyage AI: Check their pricing
  - Together AI: FREE tier available! Get key at https://api.together.xyz/settings/api-keys ($25 free credits)
- **Internet Required**: API calls require internet connection (but fallback works offline)
- **Context-Aware Responses**: The system uses your actual message and conversation history for more relevant, contextual suggestions
- **Chat Interface**: Full conversational experience with message history and session statistics

## 🎯 Features

- **Chat-Based Interface**: Natural conversation flow using Streamlit's chat components
- **Real-Time Emotion Detection**: Instant analysis as user types
- **Visual Emotion Badges**: Color-coded emotion indicators with emojis
- **Confidence Scores**: Transparent emotion prediction confidence (0-100%)
- **Top 3 Emotions Display**: Shows primary + 2 other detected emotions
- **Session Statistics**: Tracks messages and emotion distribution
- **Clear Chat History**: Button to reset conversation
- **Responsive Design**: Works on desktop and mobile browsers
- **Conversation Memory**: Maintains context across messages
- **Enhanced Fallback**: Works even without API keys (rule-based responses)

## ⚠️ Important Disclaimer

This system is for demonstration purposes only. It provides non-medical, supportive suggestions and should not be used as a substitute for professional mental health care. For serious emotional or mental health concerns, please consult with qualified healthcare providers.

## 📦 Dependencies

- `streamlit>=1.28.0`: Web application framework
- `torch>=2.0.0`: Deep learning framework (PyTorch)
- `transformers>=4.35.0`: HuggingFace transformers library
- `numpy>=1.24.0`: Numerical computing
- `scikit-learn>=1.3.0`: Machine learning utilities
- `accelerate>=0.24.0`: Model optimization
- `requests>=2.31.0`: HTTP requests for API calls

## 🔄 Future Enhancements

- Train the emotion detection model on a real emotion dataset
- Add model persistence and checkpoint saving
- Implement batch processing for multiple texts
- Add emotion history tracking across sessions
- Include more detailed probability distributions
- User authentication and conversation history persistence
- Crisis detection and intervention protocols
- Multi-language support

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Built with ❤️ using Python, Streamlit, and state-of-the-art AI models**
