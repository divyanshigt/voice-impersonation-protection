// 
import { useEffect, useState } from 'react'
import {
  FileAudio,
  UploadCloud,
  AlertTriangle,
  Brain,
  MessageSquareText,
  ShieldAlert,
  Activity,
  Link as LinkIcon,
  ShieldCheck,
  XCircle,
  History,
  RefreshCw,
} from 'lucide-react'
import './AudioUploader.css'

// Maps a risk/severity string to a visual tone. Centralized here so every
// card (fraud, overall risk) reads severity the same way.
function toneForLevel(level) {
  if (level === 'CRITICAL' || level === 'HIGH') return 'danger'
  if (level === 'MEDIUM') return 'warn'
  return 'safe'
}

function safe(value, fallback = 'N/A') {
  return value === undefined || value === null || value === '' ? fallback : value
}

function pct(value) {
  return value === undefined || value === null ? 'N/A' : `${(value * 100).toFixed(2)}%`
}

const SCANNER_STAGES = [
  'Uploading audio...',
  'Extracting acoustic features...',
  'Analyzing voice authenticity...',
  'Transcribing speech...',
  'Scanning fraud indicators...',
  'Calculating risk...',
  'Securing evidence...',
]

function AudioUploader() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [blockchainStatus, setBlockchainStatus] = useState(null)
  const [blockchainHistory, setBlockchainHistory] = useState(null)
  const [verifying, setVerifying] = useState(false)
  const [loadingHistory, setLoadingHistory] = useState(false)

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0]
    if (!selectedFile) return

    setFile(selectedFile)
    setResult(null)
    setError('')
    setBlockchainStatus(null)
    setBlockchainHistory(null)
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an audio file first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setBlockchainStatus(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/api/audio/process', {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Something went wrong.')
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // ----------------------------------------
  // VERIFY BLOCKCHAIN
  // ----------------------------------------
  const handleVerifyBlockchain = async () => {
    setVerifying(true)
    setBlockchainStatus(null)
    setError('')

    try {
      const response = await fetch(
        'http://127.0.0.1:8000/api/audio/blockchain/verify'
      )
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Blockchain verification failed.')
      }

      setBlockchainStatus(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setVerifying(false)
    }
  }

  // ----------------------------------------
  // LOAD BLOCKCHAIN HISTORY
  // ----------------------------------------
  const handleBlockchainHistory = async () => {
    setLoadingHistory(true)
    setError('')

    try {
      const response = await fetch('http://127.0.0.1:8000/api/audio/blockchain')
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Could not load blockchain history.')
      }

      setBlockchainHistory(data.blockchain)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoadingHistory(false)
    }
  }

  return (
    <div>
      {/* -------------------------------- */}
      {/* UPLOAD PANEL */}
      {/* -------------------------------- */}
      <div className="uploader-panel">
        <label className={`dropzone ${file ? 'has-file' : ''}`}>
          <div className="dropzone-icon">
            <UploadCloud size={24} />
          </div>
          <div className="dropzone-title">Drop an audio file, or click to browse</div>
          <div className="dropzone-formats">WAV • MP3 • M4A • FLAC • OGG • AAC</div>

          {file && (
            <div className="dropzone-filename">
              Selected: <strong>{file.name}</strong>
            </div>
          )}

          <input
            type="file"
            accept=".wav,.mp3,.m4a,.flac,.ogg,.aac"
            onChange={handleFileChange}
          />
        </label>

        <div style={{ marginTop: 18, textAlign: 'center' }}>
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="btn btn-primary"
          >
            <FileAudio size={16} />
            {loading ? 'Analyzing...' : 'Analyze Audio'}
          </button>
        </div>

        {loading && <LoadingScanner />}

        {error && (
          <div className="error-banner">
            <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 1 }} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* -------------------------------- */}
      {/* RESULTS */}
      {/* -------------------------------- */}
      {result && (
        <div className="results">
          <div className="summary-banner">
            <ShieldCheck size={16} />
            Voice analysis complete — {safe(result.filename)}
          </div>

          <RiskAnalysisCard riskAnalysis={result.risk_analysis} />

          <div className="card-grid">
            <VoiceDetectionCard prediction={result.prediction} model={result.model} />
            <TranscriptionCard speechToText={result.speech_to_text} />
            <FraudDetectionCard fraud={result.fraud_detection} />
          </div>

          <EvidenceIntegrity
            blockchain={result.blockchain}
            blockchainStatus={blockchainStatus}
            verifying={verifying}
            loadingHistory={loadingHistory}
            onVerify={handleVerifyBlockchain}
            onLoadHistory={handleBlockchainHistory}
          />

          {blockchainHistory && <BlockchainHistory blocks={blockchainHistory} />}
        </div>
      )}
    </div>
  )
}

/* ============================================================
   LOADING SCANNER
   ============================================================ */

function LoadingScanner() {
  const [activeIndex, setActiveIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveIndex((i) => (i + 1 < SCANNER_STAGES.length ? i + 1 : i))
    }, 900)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="scanner">
      {SCANNER_STAGES.map((stage, i) => (
        <div
          key={stage}
          className={`scanner-stage ${
            i === activeIndex ? 'is-active' : i < activeIndex ? 'is-done' : ''
          }`}
        >
          <span className="scanner-dot" />
          {stage}
        </div>
      ))}
    </div>
  )
}

/* ============================================================
   RISK RING (SVG circular progress)
   ============================================================ */

function RiskRing({ score, tone }) {
  const radius = 64
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(100, score ?? 0))
  const offset = circumference * (1 - clamped / 100)

  const strokeColor =
    tone === 'danger' ? 'var(--danger)' : tone === 'warn' ? 'var(--warn)' : 'var(--safe)'

  return (
    <div className="risk-ring-wrap">
      <svg width="160" height="160" viewBox="0 0 160 160">
        <circle className="risk-ring-track" cx="80" cy="80" r={radius} />
        <circle
          className="risk-ring-value"
          cx="80"
          cy="80"
          r={radius}
          stroke={strokeColor}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="risk-ring-center">
        <span className="risk-ring-number">{score ?? 'N/A'}</span>
        <span className="risk-ring-max">/ 100</span>
      </div>
    </div>
  )
}

/* ============================================================
   OVERALL RISK ANALYSIS (the one bold element)
   ============================================================ */

function RiskAnalysisCard({ riskAnalysis }) {
  if (!riskAnalysis) return null

  const tone = toneForLevel(riskAnalysis.risk_level)
  const reasons = riskAnalysis.reasons || []

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-icon">
          <Activity size={16} />
        </div>
        <h3>Overall Risk Analysis</h3>
      </div>

      <div className="risk-summary">
        <RiskRing score={riskAnalysis.overall_score} tone={tone} />

        <div className="risk-summary-details">
          <span className={`risk-level-badge tone-${tone}`}>
            {safe(riskAnalysis.risk_level)}
          </span>

          <div className="risk-breakdown">
            <div className="risk-breakdown-item">
              <span className="risk-breakdown-label">Voice Risk</span>
              <span className="risk-breakdown-value">
                {safe(riskAnalysis.voice_risk)}/100
              </span>
            </div>
            <div className="risk-breakdown-item">
              <span className="risk-breakdown-label">Fraud Risk</span>
              <span className="risk-breakdown-value">
                {safe(riskAnalysis.fraud_risk)}/100
              </span>
            </div>
          </div>

          {reasons.length > 0 && (
            <ul className="reason-list">
              {reasons.map((reason, i) => (
                <li key={i}>{reason}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {riskAnalysis.recommended_action && (
        <div
          className="action-panel"
          style={{
            borderColor:
              tone === 'danger'
                ? 'rgba(239,68,68,0.35)'
                : tone === 'warn'
                ? 'rgba(245,158,11,0.35)'
                : 'rgba(34,197,94,0.3)',
            background:
              tone === 'danger'
                ? 'var(--danger-soft)'
                : tone === 'warn'
                ? 'var(--warn-soft)'
                : 'var(--safe-soft)',
          }}
        >
          <div className={`action-panel-title tone-${tone}`} style={{ background: 'none' }}>
            Recommended Action
          </div>
          <div className="action-panel-body">{riskAnalysis.recommended_action}</div>
        </div>
      )}
    </div>
  )
}

/* ============================================================
   AI VOICE DETECTION
   ============================================================ */

function VoiceDetectionCard({ prediction, model }) {
  if (!prediction) return null

  const isSpoof = prediction.label === 'spoof'
  const tone = isSpoof ? 'danger' : 'safe'

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-icon">
          <Brain size={16} />
        </div>
        <h3>AI Voice Detection</h3>
      </div>

      <span className={`risk-level-badge tone-${tone}`} style={{ marginBottom: 14 }}>
        {isSpoof ? 'SPOOF' : 'BONAFIDE'}
      </span>

      <div className="field-row">
        <span className="field-label">Confidence</span>
        <span className="field-value">{pct(prediction.confidence)}</span>
      </div>
      <div className="field-row">
        <span className="field-label">Spoof Probability</span>
        <span className="field-value">{pct(prediction.spoof_probability)}</span>
      </div>
      <div className="field-row">
        <span className="field-label">Model</span>
        <span className="field-value">{safe(model?.name)}</span>
      </div>
    </div>
  )
}

/* ============================================================
   SPEECH-TO-TEXT
   ============================================================ */

function TranscriptionCard({ speechToText }) {
  if (!speechToText) return null

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-icon">
          <MessageSquareText size={16} />
        </div>
        <h3>Speech-to-Text</h3>
      </div>

      <div className="lang-badge">{safe(speechToText.language, 'Unknown')}</div>

      <div className="transcript-box">
        {speechToText.transcript || 'No speech detected.'}
      </div>
    </div>
  )
}

/* ============================================================
   FRAUD DETECTION
   ============================================================ */

function FraudDetectionCard({ fraud }) {
  if (!fraud) return null

  const tone = toneForLevel(fraud.risk_level)
  const keywords = fraud.detected_keywords || []
  const reasons = fraud.reasons || []

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-icon">
          <ShieldAlert size={16} />
        </div>
        <h3>Fraud Detection</h3>
      </div>

      <span className={`risk-level-badge tone-${tone}`} style={{ marginBottom: 12 }}>
        {safe(fraud.risk_level)} · {safe(fraud.risk_score)}/100
      </span>

      <div style={{ marginTop: 6 }}>
        <div className="field-label" style={{ fontSize: 12, marginBottom: 6 }}>
          Detected Keywords
        </div>
        {keywords.length > 0 ? (
          <div className="chip-row">
            {keywords.map((kw, i) => (
              <span className="chip" key={i}>
                {kw}
              </span>
            ))}
          </div>
        ) : (
          <div className="transcript-box" style={{ marginBottom: 12 }}>
            No suspicious keywords detected.
          </div>
        )}
      </div>

      {reasons.length > 0 && (
        <ul className="reason-list">
          {reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ============================================================
   EVIDENCE INTEGRITY (blockchain)
   ============================================================ */

function EvidenceIntegrity({
  blockchain,
  blockchainStatus,
  verifying,
  loadingHistory,
  onVerify,
  onLoadHistory,
}) {
  if (!blockchain) return null

  const blockIndex = blockchain.block_index

  return (
    <div className="card evidence-card">
      <div className="card-header">
        <div className="card-header-icon">
          <LinkIcon size={16} />
        </div>
        <h3>Evidence Integrity</h3>
      </div>

      <div className="field-row">
        <span className="field-label">Block</span>
        <span className="field-value">#{safe(blockIndex)}</span>
      </div>
      <div className="field-row">
        <span className="field-label">Timestamp</span>
        <span className="field-value">{safe(blockchain.timestamp)}</span>
      </div>

      {typeof blockIndex === 'number' && (
        <div className="evidence-chain">
          {Array.from({ length: blockIndex + 1 }, (_, i) => i)
            .slice(-5)
            .map((i, idx, arr) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span
                  className={`evidence-chain-block ${i === blockIndex ? 'is-current' : ''}`}
                >
                  Block #{i}
                </span>
                {idx < arr.length - 1 && <span className="evidence-chain-arrow">→</span>}
              </div>
            ))}
        </div>
      )}

      <div className="evidence-hash-row">
        <div className="evidence-hash-label">Block Hash</div>
        <div className="evidence-hash-value">{safe(blockchain.block_hash)}</div>
      </div>
      <div className="evidence-hash-row">
        <div className="evidence-hash-label">Previous Hash</div>
        <div className="evidence-hash-value">{safe(blockchain.previous_hash)}</div>
      </div>

      <div className="evidence-actions">
        <button onClick={onVerify} disabled={verifying} className="btn btn-secondary">
          <ShieldCheck size={15} />
          {verifying ? 'Verifying...' : 'Verify Blockchain'}
        </button>

        <button
          onClick={onLoadHistory}
          disabled={loadingHistory}
          className="btn btn-secondary"
        >
          <History size={15} />
          {loadingHistory ? 'Loading...' : 'View Blockchain History'}
        </button>
      </div>

      {blockchainStatus && (
        <div
          className="verify-result"
          style={{
            borderColor: blockchainStatus.valid
              ? 'rgba(34,197,94,0.3)'
              : 'rgba(239,68,68,0.35)',
            background: blockchainStatus.valid ? 'var(--safe-soft)' : 'var(--danger-soft)',
            color: blockchainStatus.valid ? '#86efac' : '#fca5a5',
          }}
        >
          <strong style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {blockchainStatus.valid ? (
              <ShieldCheck size={15} />
            ) : (
              <XCircle size={15} />
            )}
            {blockchainStatus.valid
              ? 'Blockchain Integrity Verified'
              : 'Blockchain Integrity Failed'}
          </strong>
          <p style={{ margin: '6px 0 0' }}>{blockchainStatus.message}</p>
        </div>
      )}
    </div>
  )
}

/* ============================================================
   BLOCKCHAIN HISTORY
   ============================================================ */

function BlockchainHistory({ blocks }) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-header-icon">
          <RefreshCw size={16} />
        </div>
        <h3>Blockchain Evidence History</h3>
      </div>

      <div className="history-list">
        {blocks.map((block, index) => (
          <div key={block.index ?? index} className="history-block">
            <div className="history-block-head">
              <span className="history-block-title">
                {block.type === 'GENESIS' ? 'Genesis Block' : `Block #${block.index}`}
              </span>
              <span className="history-block-time">{safe(block.timestamp)}</span>
            </div>

            {block.evidence && block.type === 'VOICE_ANALYSIS' && (
              <div className="history-block-grid">
                <div className="history-block-field">
                  <span className="field-label">Voice</span>
                  {safe(block.evidence.voice_label)}
                </div>
                <div className="history-block-field">
                  <span className="field-label">Voice Confidence</span>
                  {pct(block.evidence.voice_confidence)}
                </div>
                <div className="history-block-field">
                  <span className="field-label">Fraud Score</span>
                  {safe(block.evidence.fraud_score)}/100
                </div>
                <div className="history-block-field">
                  <span className="field-label">Overall Risk</span>
                  {safe(block.evidence.overall_risk_level)}
                </div>
              </div>
            )}

            {block.evidence && block.type !== 'VOICE_ANALYSIS' && (
              <div className="transcript-box" style={{ marginBottom: 10 }}>
                {block.evidence.message}
              </div>
            )}

            {block.evidence?.transcript && (
              <div className="transcript-box" style={{ marginBottom: 10 }}>
                {block.evidence.transcript}
              </div>
            )}

            <div className="evidence-hash-row" style={{ marginTop: 0 }}>
              <div className="evidence-hash-label">Hash</div>
              <div className="history-block-hash">{safe(block.hash)}</div>
            </div>
            <div className="evidence-hash-row">
              <div className="evidence-hash-label">Previous Hash</div>
              <div className="history-block-hash">{safe(block.previous_hash)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default AudioUploader