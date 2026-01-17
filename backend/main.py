"""
FastAPI Backend for HealthAI - AI-Powered Patient Diagnostics System
Integrates PyTorch model with Google Gemini for intelligent diagnostics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import torch
import numpy as np
import pickle
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HealthAI Diagnostics API",
    description="AI-powered patient diagnostics system with Google Gemini integration",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Configuration =====================
# Load environment variables from .env file
load_dotenv()

MODEL_PATH = Path(__file__).parent.parent / "ml" / "disease_prediction_model.pth"
ARTIFACTS_PATH = Path(__file__).parent.parent / "ml" / "ml_artifacts.pkl"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("⚠️ GEMINI_API_KEY not set. Chat feature will be unavailable.")

# ===================== Models & Classes =====================

class DiseasePredictor(torch.nn.Module):
    """PyTorch model for disease prediction"""
    def __init__(self, input_dim, output_dim):
        super(DiseasePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.batch_norm1 = torch.nn.BatchNorm1d(128)
        self.batch_norm2 = torch.nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PredictionRequest(BaseModel):
    """Request model for symptom-based diagnosis"""
    symptoms: List[str]
    patient_name: Optional[str] = "Patient"
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for diagnosis results"""
    top_diseases: List[Dict[str, Any]]
    confidence: float
    recommendation: str
    timestamp: str


class DiagnosticChatRequest(BaseModel):
    """Request model for diagnostic chat"""
    message: str
    symptoms: List[str]
    conversation_history: Optional[List[Dict[str, str]]] = None


class DiagnosticChatResponse(BaseModel):
    """Response model for diagnostic chat"""
    response: str
    insights: Optional[str] = None


# ===================== Global State =====================

class ModelManager:
    """Manages model loading and inference"""
    def __init__(self):
        self.model = None
        self.artifacts = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model and artifacts"""
        try:
            # Load artifacts
            with open(ARTIFACTS_PATH, 'rb') as f:
                self.artifacts = pickle.load(f)
            
            # Ensure symptom_map exists for backward compatibility
            if 'symptom_map' not in self.artifacts:
                symptom_list = self.artifacts.get('symptoms', [])
                self.artifacts['symptom_map'] = {sym.lower().strip(): idx for idx, sym in enumerate(symptom_list)}
                logger.info("✓ Created symptom_map from symptoms list")
            
            # Load model - handle different artifact structures
            if 'model_config' in self.artifacts:
                model_config = self.artifacts['model_config']
                input_dim = model_config.get('input_dim', model_config.get('input_size', 131))
                output_dim = model_config.get('output_dim', model_config.get('output_size', 41))
            else:
                # Fallback dimensions
                input_dim = len(self.artifacts.get('symptoms', []))
                output_dim = len(self.artifacts.get('diseases', []))
            
            self.model = DiseasePredictor(input_dim, output_dim)
            
            # Load state dict with tolerance for missing keys
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            try:
                self.model.load_state_dict(checkpoint)
            except RuntimeError as e:
                # If there's a mismatch, try loading without strict mode
                logger.warning(f"⚠️ Model state dict mismatch, attempting lenient load: {str(e)}")
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Model loaded successfully on {self.device}")
            logger.info(f"  Input dimensions: {input_dim}")
            logger.info(f"  Output dimensions: {output_dim}")
            
            if 'model_config' in self.artifacts:
                accuracy = self.artifacts['model_config'].get('test_accuracy', 'N/A')
                logger.info(f"  Model accuracy: {accuracy}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            logger.warning("⚠️ Model will not be available, but server will continue running")
            # Don't raise - allow server to start without model
            self.model = None
    
    def predict(self, symptoms: List[str], top_k: int = 5) -> Dict:
        """
        Predict diseases based on symptoms
        
        Args:
            symptoms: List of symptom names
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions and confidence
        """
        if not self.model or not self.artifacts:
            raise RuntimeError("Model not loaded")
        
        # Normalize symptom names
        symptoms = [s.lower().strip() for s in symptoms]
        
        # Create feature vector
        symptom_list = self.artifacts['symptoms']
        feature_vector = [0] * len(symptom_list)
        
        matched_symptoms = []
        for symptom in symptoms:
            if symptom in self.artifacts['symptom_map']:
                idx = self.artifacts['symptom_map'][symptom]
                feature_vector[idx] = 1
                matched_symptoms.append(symptom)
        
        # Convert to tensor
        X = torch.FloatTensor([feature_vector]).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(X)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        diseases = self.artifacts['diseases']
        
        predictions = [
            {
                "disease": diseases[idx],
                "probability": float(probabilities[idx]),
                "confidence": f"{probabilities[idx]*100:.1f}%"
            }
            for idx in top_indices
        ]
        
        return {
            "predictions": predictions,
            "matched_symptoms": matched_symptoms,
            "unmatched_symptoms": [s for s in symptoms if s not in matched_symptoms],
            "overall_confidence": float(np.max(probabilities))
        }


# Initialize model manager
try:
    model_manager = ModelManager()
except Exception as e:
    logger.error(f"Failed to initialize model manager: {e}")
    model_manager = None


# ===================== API Endpoints =====================

@app.get("/", tags=["Health Check"])
async def root():
    """Root endpoint - health check"""
    return {
        "status": "✓ HealthAI API is running",
        "version": "1.0.0",
        "features": [
            "Symptom-based disease prediction",
            "AI-powered diagnostic chat with Google Gemini",
            "Patient consultation support"
        ]
    }


@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check endpoint"""
    model_status = "✓ Loaded" if model_manager and model_manager.model else "✗ Not loaded"
    return {
        "status": "healthy",
        "model": model_status,
        "device": str(model_manager.device) if model_manager else "N/A",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/symptoms", tags=["Diagnostics"])
async def get_available_symptoms():
    """Get list of all recognized symptoms"""
    if not model_manager or not model_manager.artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "total_symptoms": len(model_manager.artifacts['symptoms']),
        "symptoms": sorted(model_manager.artifacts['symptoms']),
        "note": "Use these exact symptom names for predictions"
    }


@app.get("/api/diseases", tags=["Diagnostics"])
async def get_available_diseases():
    """Get list of all recognized diseases"""
    if not model_manager or not model_manager.artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "total_diseases": len(model_manager.artifacts['diseases']),
        "diseases": sorted(model_manager.artifacts['diseases'])
    }


@app.post("/api/predict", response_model=PredictionResponse, tags=["Diagnostics"])
async def predict_disease(request: PredictionRequest):
    """
    Predict disease based on symptoms
    
    Example request:
    {
        "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
        "patient_name": "John Doe",
        "patient_age": 35,
        "patient_gender": "Male"
    }
    """
    if not model_manager or not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get predictions
        result = model_manager.predict(request.symptoms, top_k=5)
        predictions = result['predictions']
        
        # Generate recommendation
        top_disease = predictions[0]['disease'] if predictions else "Unknown"
        confidence = result['overall_confidence']
        
        recommendation = generate_medical_recommendation(
            top_disease, 
            request.symptoms,
            confidence
        )
        
        return PredictionResponse(
            top_diseases=predictions,
            confidence=confidence,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/api/chat", response_model=DiagnosticChatResponse, tags=["AI Chat"])
async def diagnostic_chat(request: DiagnosticChatRequest):
    """
    Chat with AI doctor about diagnosis
    Uses Google Gemini for intelligent responses (optional)
    Falls back to rule-based responses if API key not configured
    """
    try:
        # Get disease predictions
        result = model_manager.predict(request.symptoms, top_k=3)
        predictions = result['predictions']
        
        # Try to use Gemini API if available
        if GEMINI_API_KEY:
            try:
                # Build context for Gemini
                context = f"""
You are a medical assistant AI doctor helping patients understand their potential diagnoses.

Based on symptoms: {', '.join(request.symptoms)}
Predicted conditions (from ML model):
"""
                for pred in predictions:
                    context += f"\n- {pred['disease']} ({pred['confidence']})"
                
                context += f"""

Patient message: {request.message}

Guidelines:
- Be empathetic and clear
- Always recommend seeing a doctor for professional diagnosis
- Do not provide definitive medical diagnosis
- Focus on explaining symptoms and conditions
- Suggest lifestyle changes where appropriate
- Ask clarifying questions if needed
"""
                
                # Call Gemini API
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(context)
                
                return DiagnosticChatResponse(
                    response=response.text,
                    insights=f"Based on symptoms: {', '.join(request.symptoms)}"
                )
            except Exception as e:
                logger.warning(f"Gemini API error: {str(e)}, falling back to default response")
        
        # Fallback response if Gemini not available
        top_disease = predictions[0]['disease'] if predictions else "Unknown condition"
        confidence = predictions[0]['confidence'] if predictions else "N/A"
        
        fallback_response = f"""Thank you for your question about your diagnosis.

Based on the symptom analysis, the most likely condition appears to be **{top_disease}** (confidence: {confidence}).

However, I must emphasize that:
- This is an AI-assisted analysis, not a professional medical diagnosis
- You should consult with a qualified healthcare professional for definitive diagnosis
- Treatment options vary based on individual factors

Regarding your question: {request.message}

I recommend documenting your symptoms and discussing them with a doctor who can perform a proper physical examination and potentially order tests if needed.

In the meantime, ensure adequate rest, stay hydrated, and seek immediate medical attention if symptoms worsen significantly."""
        
        return DiagnosticChatResponse(
            response=fallback_response,
            insights=f"Based on symptoms: {', '.join(request.symptoms)}"
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Chat failed: {str(e)}")


@app.post("/api/symptom-search", tags=["Diagnostics"])
async def search_symptoms(query: str):
    """Search for symptoms matching a query"""
    if not model_manager or not model_manager.artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    query = query.lower().strip()
    symptoms = model_manager.artifacts['symptoms']
    
    # Find matching symptoms
    matches = [s for s in symptoms if query in s]
    
    return {
        "query": query,
        "matches": matches,
        "count": len(matches)
    }


# ===================== Helper Functions =====================

def generate_medical_recommendation(disease: str, symptoms: List[str], confidence: float) -> str:
    """Generate basic medical recommendation"""
    confidence_level = "High" if confidence > 0.7 else "Moderate" if confidence > 0.5 else "Low"
    
    recommendation = f"""
⚠️ IMPORTANT DISCLAIMER: This is an AI-assisted diagnosis support tool, NOT a professional medical diagnosis.
Always consult with a qualified healthcare professional for proper diagnosis and treatment.

🔍 Analysis Results:
- Predicted Condition: {disease}
- Confidence Level: {confidence_level} ({confidence*100:.1f}%)
- Symptoms Provided: {len(symptoms)}

📋 Recommendations:
1. Schedule an appointment with your doctor immediately
2. Keep a detailed symptom diary
3. Bring this assessment to your doctor's visit
4. Do not self-medicate without professional guidance
5. If symptoms worsen, seek emergency care

💊 Next Steps:
- Consult a medical professional for confirmation
- Get prescribed treatment if needed
- Follow medical advice strictly
- Monitor symptom progression
"""
    return recommendation


# ===================== Startup/Shutdown =====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 60)
    logger.info("HealthAI API Starting Up...")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Device: {model_manager.device if model_manager else 'N/A'}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("HealthAI API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
