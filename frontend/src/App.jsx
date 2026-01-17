import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ==================== Extracted Components (Outside App) ====================
const HomePage = React.memo(({ onStartDiagnosis }) => (
  <div className="page home-page">
    <div className="home-container">
      <div className="hero">
        <h1>🏥 HealthAI Diagnostics</h1>
        <p className="subtitle">AI-Powered Patient Diagnostics System</p>
        <p className="description">
          Get instant diagnostic insights based on your symptoms with AI technology
        </p>
      </div>

      <div className="features">
        <div className="feature-card">
          <span className="feature-icon">🔬</span>
          <h3>Smart Diagnosis</h3>
          <p>Advanced AI model trained on medical data</p>
        </div>
        <div className="feature-card">
          <span className="feature-icon">💬</span>
          <h3>AI Chatbot</h3>
          <p>Chat with our AI doctor for more insights</p>
        </div>
        <div className="feature-card">
          <span className="feature-icon">⚡</span>
          <h3>Fast Results</h3>
          <p>Get instant predictions and recommendations</p>
        </div>
      </div>

      <button
        className="btn btn-primary btn-lg"
        onClick={onStartDiagnosis}
      >
        Start Diagnosis →
      </button>

      <div className="disclaimer">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for informational purposes only. Always consult with a qualified healthcare professional for medical diagnosis and treatment.</p>
      </div>
    </div>
  </div>
));

const SymptomsPage = React.memo(({
  patientInfo,
  onPatientNameChange,
  onPatientAgeChange,
  onPatientGenderChange,
  symptomQuery,
  onSymptomSearch,
  filteredSymptoms,
  onAddSymptom,
  symptoms,
  onRemoveSymptom,
  loading,
  error,
  onPredict,
  onBack
}) => (
  <div className="page symptoms-page">
    <div className="page-header">
      <button className="btn-back" onClick={onBack}>← Back</button>
      <h1>Symptom Selection</h1>
    </div>

    <div className="form-section">
      <h2>Patient Information (Optional)</h2>
      <div className="form-row">
        <input
          type="text"
          placeholder="Full Name"
          value={patientInfo.name}
          onChange={onPatientNameChange}
          className="form-input"
        />
        <input
          type="number"
          placeholder="Age"
          value={patientInfo.age}
          onChange={onPatientAgeChange}
          className="form-input"
        />
        <select
          value={patientInfo.gender}
          onChange={onPatientGenderChange}
          className="form-input"
        >
          <option value="">Select Gender</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
          <option value="Other">Other</option>
        </select>
      </div>
    </div>

    <div className="form-section">
      <h2>Select Your Symptoms</h2>
      <div className="symptom-search">
        <input
          type="text"
          placeholder="Search symptoms (e.g., 'itching', 'fever')..."
          value={symptomQuery}
          onChange={(e) => onSymptomSearch(e.target.value)}
          className="form-input"
        />
        {filteredSymptoms.length > 0 && (
          <div className="suggestions">
            {filteredSymptoms.map(symptom => (
              <div
                key={symptom}
                className="suggestion-item"
                onClick={() => onAddSymptom(symptom)}
              >
                + {symptom}
              </div>
            ))}
          </div>
        )}
      </div>

      {symptoms.length > 0 && (
        <div className="selected-symptoms">
          <h3>Selected Symptoms ({symptoms.length})</h3>
          <div className="symptom-tags">
            {symptoms.map(symptom => (
              <div key={symptom} className="symptom-tag">
                {symptom}
                <button
                  className="tag-remove"
                  onClick={() => onRemoveSymptom(symptom)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>

    {error && <div className="error-message">{error}</div>}

    <button
      className="btn btn-primary btn-lg"
      onClick={onPredict}
      disabled={symptoms.length === 0 || loading}
    >
      {loading ? 'Analyzing...' : 'Get Diagnosis →'}
    </button>
  </div>
));

const ResultsPage = React.memo(({
  predictions,
  chatHistory,
  chatMessage,
  onChatMessageChange,
  onSendChat,
  loading,
  chatEndRef,
  onNewDiagnosis,
  error
}) => {
  if (!predictions) return null;

  return (
    <div className="page results-page">
      <div className="page-header">
        <button className="btn-back" onClick={onNewDiagnosis}>← New Diagnosis</button>
        <h1>Diagnostic Results</h1>
      </div>

      <div className="results-container">
        <div className="predictions-section">
          <h2>🔍 Top Predictions</h2>
          <div className="predictions-list">
            {predictions.top_diseases.map((disease, index) => (
              <div key={index} className="prediction-card">
                <div className="prediction-rank">#{index + 1}</div>
                <div className="prediction-info">
                  <h3>{disease.disease}</h3>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${disease.probability * 100}%` }}
                    />
                  </div>
                  <p className="confidence-text">
                    Confidence: <strong>{disease.confidence}</strong>
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="recommendation-section">
          <h2>📋 Medical Recommendation</h2>
          <div className="recommendation-box">
            <pre>{predictions.recommendation}</pre>
          </div>
        </div>

        <div className="chat-section">
          <h2>💬 Ask AI Doctor</h2>
          <div className="chat-history">
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`chat-message ${msg.role}`}>
                <p>{msg.content}</p>
              </div>
            ))}
            {loading && (
              <div className="chat-message assistant">
                <p>🤖 <span className="typing-indicator">●●●</span></p>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {error && <div className="error-message">{error}</div>}

          <div className="chat-input-section">
            <input
              type="text"
              placeholder="Ask me anything about your diagnosis..."
              value={chatMessage}
              onChange={(e) => onChatMessageChange(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && onSendChat()}
              className="form-input"
            />
            <button
              className="btn btn-primary"
              onClick={onSendChat}
              disabled={loading || chatMessage.trim() === ''}
            >
              Send
            </button>
          </div>
        </div>

        <div className="disclaimer">
          <p>⚠️ <strong>Disclaimer:</strong> This is an AI-assisted diagnosis tool, NOT professional medical advice. Always consult a qualified healthcare professional.</p>
        </div>
      </div>
    </div>
  );
});

// ==================== Main App Component ====================
function App() {
  // ==================== State Management ====================
  const [currentPage, setCurrentPage] = useState('home');
  const [symptoms, setSymptoms] = useState([]);
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [symptomQuery, setSymptomQuery] = useState('');
  const [filteredSymptoms, setFilteredSymptoms] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatMessage, setChatMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    age: '',
    gender: ''
  });
  const chatEndRef = useRef(null);

  // ==================== Lifecycle ====================
  useEffect(() => {
    fetchAvailableSymptoms();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // ==================== API Calls ====================
  const fetchAvailableSymptoms = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/symptoms`);
      setAvailableSymptoms(response.data.symptoms);
    } catch (err) {
      setError('Failed to load symptoms list');
      console.error(err);
    }
  };

  // ==================== Handlers (Memoized) ====================
  const handleSymptomSearch = useCallback((query) => {
    setSymptomQuery(query);
    if (query.length > 0) {
      const filtered = availableSymptoms.filter(s =>
        s.toLowerCase().includes(query.toLowerCase())
      );
      setFilteredSymptoms(filtered.slice(0, 10));
    } else {
      setFilteredSymptoms([]);
    }
  }, [availableSymptoms]);

  const handlePatientNameChange = useCallback((e) => {
    setPatientInfo(prev => ({ ...prev, name: e.target.value }));
  }, []);

  const handlePatientAgeChange = useCallback((e) => {
    setPatientInfo(prev => ({ ...prev, age: e.target.value }));
  }, []);

  const handlePatientGenderChange = useCallback((e) => {
    setPatientInfo(prev => ({ ...prev, gender: e.target.value }));
  }, []);

  const addSymptom = useCallback((symptom) => {
    setSymptoms(prev => {
      if (!prev.includes(symptom)) {
        return [...prev, symptom];
      }
      return prev;
    });
    setSymptomQuery('');
    setFilteredSymptoms([]);
  }, []);

  const removeSymptom = useCallback((symptom) => {
    setSymptoms(prev => prev.filter(s => s !== symptom));
  }, []);

  const predictDisease = useCallback(async () => {
    if (symptoms.length === 0) {
      setError('Please select at least one symptom');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/predict`, {
        symptoms: symptoms,
        patient_name: patientInfo.name || 'Patient',
        patient_age: patientInfo.age ? parseInt(patientInfo.age) : null,
        patient_gender: patientInfo.gender || null
      });

      setPredictions(response.data);
      setChatHistory([
        {
          role: 'assistant',
          content: `I've analyzed your symptoms: ${symptoms.join(', ')}. The most likely condition is ${response.data.top_diseases[0].disease} with ${response.data.top_diseases[0].confidence} confidence. How can I help you understand this better?`
        }
      ]);
      setCurrentPage('results');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [symptoms, patientInfo]);

  const sendChatMessage = useCallback(async () => {
    if (chatMessage.trim() === '') return;

    const userMessage = chatMessage;
    setChatMessage('');
    setChatHistory(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: userMessage,
        symptoms: symptoms,
        conversation_history: chatHistory
      });

      setChatHistory(prev => [
        ...prev,
        { role: 'assistant', content: response.data.response }
      ]);
    } catch (err) {
      setError('Failed to get chat response');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [chatMessage, symptoms, chatHistory]);

  const handleNewDiagnosis = useCallback(() => {
    setCurrentPage('symptoms');
    setPredictions(null);
    setChatHistory([]);
  }, []);

  const handleStartDiagnosis = useCallback(() => {
    setCurrentPage('symptoms');
  }, []);

  const handleBackToHome = useCallback(() => {
    setCurrentPage('home');
  }, []);

  const handleChatMessageChange = useCallback((msg) => {
    setChatMessage(msg);
  }, []);

  // ==================== Main Render ====================
  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>HealthAI</h1>
          <p>Intelligent Patient Diagnostics System</p>
        </div>
      </header>

      <main className="app-main">
        {currentPage === 'home' && <HomePage onStartDiagnosis={handleStartDiagnosis} />}
        {currentPage === 'symptoms' && (
          <SymptomsPage
            patientInfo={patientInfo}
            onPatientNameChange={handlePatientNameChange}
            onPatientAgeChange={handlePatientAgeChange}
            onPatientGenderChange={handlePatientGenderChange}
            symptomQuery={symptomQuery}
            onSymptomSearch={handleSymptomSearch}
            filteredSymptoms={filteredSymptoms}
            onAddSymptom={addSymptom}
            symptoms={symptoms}
            onRemoveSymptom={removeSymptom}
            loading={loading}
            error={error}
            onPredict={predictDisease}
            onBack={handleBackToHome}
          />
        )}
        {currentPage === 'results' && (
          <ResultsPage
            predictions={predictions}
            chatHistory={chatHistory}
            chatMessage={chatMessage}
            onChatMessageChange={handleChatMessageChange}
            onSendChat={sendChatMessage}
            loading={loading}
            chatEndRef={chatEndRef}
            onNewDiagnosis={handleNewDiagnosis}
            error={error}
          />
        )}
      </main>

      <footer className="app-footer">
        <p>&copy; HealthAI. All rights reserved. | For educational purposes only.</p>
      </footer>
    </div>
  );
}

export default App;
