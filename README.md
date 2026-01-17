# 🏥 HealthAI - AI-Powered Patient Diagnostics System

An intelligent medical diagnostics system that leverages machine learning and AI to provide symptom-based disease predictions with an interactive AI chatbot for patient consultation.

## ✨ Features

- **🔬 Smart Diagnosis**: Advanced neural network trained on medical symptom-disease datasets with 95% accuracy
- **💬 AI Chatbot**: Interactive AI doctor powered by Google Gemini API (with intelligent fallback responses)
- **⚡ Fast Results**: Real-time predictions with confidence scores
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **🔍 Symptom Search**: Search from 131+ symptoms with intelligent filtering
- **📊 Disease Predictions**: Get top 5 disease predictions with confidence levels
- **💾 Patient Information**: Optional patient data collection (name, age, gender)
- **📋 Medical Recommendations**: AI-generated medical recommendations for top predictions

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React 18)                     │
│              http://localhost:3000                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   HomePage  │  │ SymptomsPage │  │  ResultsPage    │   │
│  │  (Memoized) │  │  (Memoized)  │  │   (Memoized)    │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
│                          ↓                                 │
│                  Axios HTTP Requests                       │
│                          ↓                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                  Backend (FastAPI)                         │
│              http://localhost:8000                         │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │  API Endpoints:                                    │   │
│  │  • GET  /api/symptoms          (Fetch symptom list)   │   │
│  │  • GET  /api/diseases          (Fetch disease list)   │   │
│  │  • POST /api/predict           (Disease prediction)   │   │
│  │  • POST /api/chat              (AI chat response)     │   │
│  │  • GET  /api/symptom-search    (Search symptoms)      │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Model Manager:                                    │   │
│  │  • DiseasePredictor (PyTorch Neural Network)       │   │
│  │  • Model: 131 → 128 → 64 → 41 neurons             │   │
│  │  • Accuracy: 95%                                  │   │
│  └────────────────────────────────────────────────────┘   │
│                          ↓                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │  ML Artifacts:                                     │   │
│  │  • disease_prediction_model.pth (PyTorch weights) │   │
│  │  • ml_artifacts.pkl (symptoms/diseases mapping)   │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │  AI Integration:                                   │   │
│  │  • Google Gemini API (optional)                    │   │
│  │  • Fallback intelligent responses                 │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Frontend
- **React 18.2.0** - UI library
- **Axios 1.6.0** - HTTP client
- **CSS3** - Responsive styling with variables and animations
- **Node.js & npm** - Package management

### Backend
- **Python 3.11** - Programming language
- **FastAPI 0.104.1** - Web framework
- **Uvicorn 0.24.0** - ASGI server
- **PyTorch 2.1.1** - Deep learning framework
- **scikit-learn 1.3.2** - Machine learning utilities
- **Google Generative AI** - AI chat integration (optional)
- **Pydantic 2.5.0** - Data validation

### Database & Storage
- **Pickle** - Model serialization
- **CSV** - Dataset files

## 📋 Requirements

### System Requirements
- **Python 3.8+**
- **Node.js 14+** and npm
- **4GB RAM** (minimum)
- **500MB** disk space

### Python Dependencies
```
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.1
numpy==1.24.3
scikit-learn==1.3.2
google-generativeai==0.3.0
python-dotenv==1.0.0
pydantic==2.5.0
```

### Node.js Dependencies
```
react==18.2.0
react-dom==18.2.0
axios==1.6.0
react-scripts==5.0.1
```

## 🚀 Installation & Setup

### Prerequisites
1. Clone the repository:
```bash
git clone https://github.com/yourusername/HealthAI.git
cd HealthAI
```

2. Create Python virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Install Node.js dependencies:
```bash
cd ../frontend
npm install
```

### Configuration

1. **Backend Configuration** - Create `.env` file in `backend/` directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=True
```

Note: The GEMINI_API_KEY is optional. If not provided, the system will use intelligent fallback responses.

## 💻 Running the Application

### Start Backend Server
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

### Start Frontend Server
In a new terminal:
```bash
cd frontend
npm start
```

Frontend will open automatically at: `http://localhost:3000`

### Alternative: Run Backend with Batch Script (Windows)
```bash
backend\run_backend.bat
```

## 🎯 Usage

### Web Application
1. Open http://localhost:3000 in your browser
2. Click **"Start Diagnosis →"** button
3. **Enter Patient Information** (Optional):
   - Full Name
   - Age
   - Gender
4. **Search and Select Symptoms**:
   - Type symptom names in the search box
   - Click to add symptoms
   - Remove symptoms by clicking the × button
5. **Get Diagnosis**:
   - Click "Get Diagnosis →" button
   - View top 5 disease predictions with confidence scores
   - Read medical recommendation
6. **Chat with AI Doctor**:
   - Ask questions about your diagnosis
   - Get intelligent responses (with or without Gemini API)
   - View conversation history

### Streamlit Alternative
```bash
cd backend
streamlit run app.py
```
Access at: `http://localhost:8501`

## 📊 Dataset

The system uses medical datasets containing:
- **131 Symptoms**: Comprehensive list of medical symptoms
- **41 Diseases**: Common diseases for prediction
- **4,920 Training Samples**: With 95% model accuracy
- **Null Value Handling**: 53% of raw data processed through smart preprocessing

### Dataset Files
- `dataset/dataset.csv` - Main training dataset
- `dataset/symptom_Description.csv` - Symptom descriptions
- `dataset/symptom_precaution.csv` - Precautions for symptoms
- `dataset/Symptom-severity.csv` - Symptom severity ratings

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```
Response: `{"status": "ok"}`

### Get Symptoms
```
GET /api/symptoms
```
Response:
```json
{
  "symptoms": ["fever", "cough", "headache", ...]
}
```

### Get Diseases
```
GET /api/diseases
```
Response:
```json
{
  "diseases": ["Common Cold", "Flu", "COVID-19", ...]
}
```

### Predict Disease
```
POST /api/predict
Content-Type: application/json

{
  "symptoms": ["fever", "cough"],
  "patient_name": "John Doe",
  "patient_age": 30,
  "patient_gender": "Male"
}
```

Response:
```json
{
  "top_diseases": [
    {
      "disease": "Common Cold",
      "probability": 0.45,
      "confidence": "45.0%"
    },
    ...
  ],
  "confidence": 0.45,
  "recommendation": "Based on your symptoms...",
  "timestamp": "2024-01-17T15:30:00"
}
```

### AI Chat
```
POST /api/chat
Content-Type: application/json

{
  "message": "What should I do about my symptoms?",
  "symptoms": ["fever", "cough"],
  "conversation_history": []
}
```

Response:
```json
{
  "response": "Based on your symptoms and medical analysis...",
  "insights": "Based on symptoms: fever, cough"
}
```

### Search Symptoms
```
GET /api/symptom-search?query=fever
```

Response:
```json
{
  "query": "fever",
  "matches": ["fever", "high fever", "intermittent fever"],
  "count": 3
}
```

## 📁 Project Structure

```
HealthAI/
├── backend/
│   ├── main.py                      # FastAPI application
│   ├── app.py                       # Streamlit alternative
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Environment variables
│   ├── run_backend.bat              # Windows startup script
│   └── venv/                        # Python virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main React component
│   │   ├── App.css                  # Styling
│   │   ├── index.js                 # Entry point
│   │   ├── index.css                # Global styles
│   │   └── index.html               # HTML template
│   ├── package.json                 # Node dependencies
│   └── node_modules/                # Installed packages
│
├── ml/
│   ├── disease_prediction_model.pth # Trained PyTorch model
│   ├── ml_artifacts.pkl             # Model artifacts
│   └── model_training.ipynb         # Training notebook
│
├── dataset/
│   ├── dataset.csv                  # Main training data
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
│
└── README.md                        # This file
```

## 🔍 Key Features Explained

### Model Architecture
- **Input Layer**: 131 neurons (one for each symptom)
- **Hidden Layer 1**: 128 neurons with ReLU activation and Batch Normalization
- **Hidden Layer 2**: 64 neurons with ReLU activation and Batch Normalization
- **Output Layer**: 41 neurons (softmax for disease classification)
- **Dropout**: 30% regularization
- **Optimizer**: Adam with learning rate reduction
- **Accuracy**: 95% on test set

### Frontend Optimization
- **React.memo**: Prevents unnecessary re-renders
- **useCallback**: Memoized event handlers
- **Functional State Updates**: Prevents stale closures
- **Responsive Design**: Mobile-first approach with breakpoints at 480px, 768px, 1200px
- **No Page Refresh**: Smooth UX with persistent input focus

### Backend Optimization
- **Lenient Model Loading**: Handles state dict mismatches
- **CORS Configuration**: Allows cross-origin requests from frontend
- **Error Handling**: Graceful fallbacks for missing API keys
- **Async Operations**: Non-blocking API calls
- **Model Caching**: Single model instance reused for all predictions

## ⚙️ Configuration

### Environment Variables (.env)
```env
# Google Gemini API Key (optional)
GEMINI_API_KEY=your_api_key_here

# Debug Mode
DEBUG=True
```

To get a free Gemini API key:
1. Visit https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key and paste in .env file
4. Restart backend

## ⚠️ Disclaimer

**IMPORTANT**: This system is for **educational and informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment.

- Always consult with a qualified healthcare professional for medical concerns
- Do not use this system as your sole source of medical guidance
- Results are AI-generated predictions, not medical diagnoses
- Seek immediate medical attention for severe symptoms
- This tool should only supplement, not replace, medical consultation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- [ ] User authentication and profile management
- [ ] Historical diagnosis tracking
- [ ] Integration with real medical databases
- [ ] Multi-language support
- [ ] Mobile app (React Native/Flutter)
- [ ] Advanced analytics dashboard
- [ ] Doctor review system
- [ ] Telemedicine integration
- [ ] Appointment scheduling
- [ ] Medical record storage

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Dataset sources: Medical symptom-disease mapping datasets
- Google Gemini API for AI chat capabilities
- PyTorch for deep learning framework
- FastAPI for backend framework
- React for frontend framework

---

**Made with ❤️ for healthcare innovation**

Last Updated: January 17, 2026
