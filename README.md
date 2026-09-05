# 🛡️ VoiceShield — Voice Impersonation Protection System

> AI-powered voice impersonation, fraud detection and risk assessment system.

## 🚨 Problem

AI voice cloning and deepfake technology can be misused to impersonate individuals and perform financial or social-engineering fraud.

Traditional voice verification systems may not be sufficient when attackers use AI-generated or manipulated speech.

VoiceShield addresses this problem by combining:

- AI-based voice spoof detection
- Speech-to-text analysis
- Fraud indicator detection
- Composite risk scoring
- Tamper-evident evidence logging

---

## 💡 Solution

VoiceShield analyzes an uploaded voice recording through a multi-stage security pipeline:

Audio Upload
→ Audio Preprocessing
→ AI Voice Detection
→ Speech-to-Text
→ Fraud Detection
→ Risk Engine
→ Evidence Ledger

The system produces an explainable risk assessment instead of only returning a simple
"real/fake" prediction.

---

## ✨ Key Features

### 🎙️ AI Voice Detection
Detects whether a recording is likely:

- BONAFIDE
- SPOOF / AI-GENERATED

using a custom CNN-based voice spoof detector.

### 📝 Speech-to-Text

OpenAI Whisper is used to:

- transcribe the recording
- automatically detect the spoken language

### 🛡️ Fraud Detection

The transcript is scanned for suspicious indicators such as:

- OTP
- PIN
- Password
- Bank
- Account
- Transfer
- Money
- Payment
- UPI
- Verification
- CVV
- Card
- Urgent

### 🚨 Risk Engine

Combines voice authenticity risk and fraud risk into a single overall risk score.

Risk levels:

| Score | Level |
|---|---|
| 0–39 | LOW |
| 40–69 | MEDIUM |
| 70–84 | HIGH |
| 85–100 | CRITICAL |

### 🔗 Evidence Ledger

Each analysis is stored as a hash-linked evidence block.

Each block contains:

- Block index
- Timestamp
- Voice classification
- Confidence
- Fraud score
- Overall risk
- Previous block hash
- Current block hash
- Transcript

The chain can be independently verified through the application.

---

# 🧠 AI Model

## CloneDetectorCNN

The system uses a custom CNN for voice spoof detection.

### Input Features

Audio is converted into Mel-spectrogram features.

Configuration:

- Sample rate: 16 kHz
- Target duration: 4 seconds
- Mel bins: 64
- FFT size: 1024
- Hop length: 256
- Feature shape: 64 × 251

### Model

The CNN learns discriminative acoustic patterns from the Mel-spectrogram.

The final layer produces a spoof probability between 0 and 1.

Classification:

```text
Probability >= 0.5 → SPOOF
Probability < 0.5  → BONAFIDE

📊 Model Performance

Evaluation was performed on the 1,024-sample held-out test set.

Metric	Result
Accuracy	80.96%
Precision	87.27%
Recall	73.42%
F1 Score	79.75%
ROC-AUC	89.50%
EER

Confusion Matrix:

                 Predicted
              Bonafide  Spoof

Actual
Bonafide         445      56
Spoof            139     384

🏗️ System Architecture
                 ┌─────────────────┐
                 │   Audio Upload  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Audio Processing│
                 │ Validation      │
                 │ Mel Features    │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ CloneDetectorCNN│
                 │ Voice Detection │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │     Whisper     │
                 │ Speech-to-Text  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Fraud Detection │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │   Risk Engine   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Evidence Ledger │
                 │ Hash-linked     │
                 └─────────────────┘

                 🛠️ Tech Stack
Frontend
React
Vite
JavaScript
Lucide React
Backend
Python
FastAPI
Uvicorn
AI / ML
PyTorch
OpenAI Whisper
Librosa / audio processing tools
Security
Fraud keyword analysis
Composite risk engine
SHA-256 hash-linked evidence chain
Audio

Supported formats:

WAV
MP3
M4A
FLAC
OGG
AAC
🔐 Security

The application includes:

File extension validation
25 MB upload limit
Chunk-based file handling
Empty file validation
Safe filename handling
FFmpeg timeout protection
Temporary file cleanup
Controlled API error messages
CORS configuration
🚀 Running the Project
Backend
cd D:\SIH104\voice-impersonation-protection
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
Frontend
cd frontend
npm run dev

Frontend:

http://localhost:5173

Backend:

http://127.0.0.1:8000
🎯 Future Scope
Real-time call analysis
Speaker verification
Advanced deepfake detection
Multilingual fraud intent detection
Cloud deployment
Distributed blockchain integration
Privacy-preserving evidence storage
Mobile application
Real-time telecom integration