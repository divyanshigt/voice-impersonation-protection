// import { useEffect, useState } from 'react'
// import AudioUploader from './components/AudioUploader'

// function App() {
//   const [status, setStatus] = useState('Checking...')

//   useEffect(() => {
//     fetch('http://127.0.0.1:8000/api/health')
//       .then((res) => res.json())
//       .then((data) => {
//         if (data.status === 'ok') {
//           setStatus('Connected')
//         } else {
//           setStatus('Unknown')
//         }
//       })
//       .catch(() => {
//         setStatus('Not Connected')
//       })
//   }, [])

//   const isConnected = status === 'Connected'

//   return (
//     <div style={styles.app}>

//       {/* HEADER */}
//       <header style={styles.header}>

//         <div>
//           <div style={styles.logo}>
//             🛡️ VoiceShield
//           </div>

//           <div style={styles.subtitle}>
//             AI-Powered Voice Impersonation Protection
//           </div>
//         </div>

//         <div
//           style={{
//             ...styles.statusBadge,
//             ...(isConnected
//               ? styles.connected
//               : styles.disconnected),
//           }}
//         >
//           <span style={styles.statusDot}></span>
//           {status}
//         </div>

//       </header>

//       {/* HERO SECTION */}
//       <section style={styles.hero}>

//         <div style={styles.heroIcon}>
//           🎙️
//         </div>

//         <div>
//           <h1 style={styles.title}>
//             Voice Impersonation
//             <br />
//             Protection System
//           </h1>

//           <p style={styles.description}>
//             Detect AI-generated voices, analyze suspicious
//             conversations, calculate risk, and securely
//             record evidence using a blockchain-style ledger.
//           </p>
//         </div>

//       </section>

//       {/* PIPELINE */}
//       <section style={styles.pipeline}>

//         <PipelineStep
//           icon="🎵"
//           title="Audio"
//           text="Upload"
//         />

//         <Arrow />

//         <PipelineStep
//           icon="🤖"
//           title="AI Voice"
//           text="Detection"
//         />

//         <Arrow />

//         <PipelineStep
//           icon="📝"
//           title="Speech"
//           text="Analysis"
//         />

//         <Arrow />

//         <PipelineStep
//           icon="🛡️"
//           title="Fraud"
//           text="Detection"
//         />

//         <Arrow />

//         <PipelineStep
//           icon="🚨"
//           title="Risk"
//           text="Engine"
//         />

//         <Arrow />

//         <PipelineStep
//           icon="🔗"
//           title="Evidence"
//           text="Blockchain"
//         />

//       </section>

//       {/* MAIN ANALYZER */}
//       <main style={styles.main}>

//         <div style={styles.sectionHeader}>
//           <h2 style={styles.sectionTitle}>
//             🔍 Analyze Voice
//           </h2>

//           <p style={styles.sectionText}>
//             Upload an audio recording to perform complete
//             voice impersonation and fraud analysis.
//           </p>
//         </div>

//         <AudioUploader />

//       </main>

//       {/* FOOTER */}
//       <footer style={styles.footer}>
//         <p>
//           VoiceShield • AI Voice Security & Evidence Protection
//         </p>

//         <p style={styles.footerSmall}>
//           Built for Smart India Hackathon
//         </p>
//       </footer>

//     </div>
//   )
// }


// /* -------------------------------- */
// /* PIPELINE COMPONENT */
// /* -------------------------------- */

// function PipelineStep({ icon, title, text }) {
//   return (
//     <div style={styles.pipelineStep}>

//       <div style={styles.pipelineIcon}>
//         {icon}
//       </div>

//       <strong>{title}</strong>

//       <span style={styles.pipelineText}>
//         {text}
//       </span>

//     </div>
//   )
// }


// /* -------------------------------- */
// /* ARROW */
// /* -------------------------------- */

// function Arrow() {
//   return (
//     <div style={styles.arrow}>
//       →
//     </div>
//   )
// }


// /* -------------------------------- */
// /* STYLES */
// /* -------------------------------- */

// const styles = {

//   app: {
//     minHeight: '100vh',
//     background:
//       'linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%)',
//     color: '#111827',
//     fontFamily:
//       'Inter, Arial, Helvetica, sans-serif',
//   },

//   /* HEADER */

//   header: {
//     display: 'flex',
//     justifyContent: 'space-between',
//     alignItems: 'center',
//     padding: '22px 50px',
//     background: '#ffffff',
//     borderBottom: '1px solid #e5e7eb',
//     boxShadow: '0 2px 10px rgba(0,0,0,0.04)',
//   },

//   logo: {
//     fontSize: '24px',
//     fontWeight: '800',
//     color: '#111827',
//   },

//   subtitle: {
//     marginTop: '4px',
//     color: '#6b7280',
//     fontSize: '14px',
//   },

//   statusBadge: {
//     display: 'flex',
//     alignItems: 'center',
//     gap: '8px',
//     padding: '9px 16px',
//     borderRadius: '999px',
//     fontWeight: '600',
//     fontSize: '14px',
//   },

//   connected: {
//     background: '#ecfdf5',
//     color: '#047857',
//     border: '1px solid #a7f3d0',
//   },

//   disconnected: {
//     background: '#fef2f2',
//     color: '#b91c1c',
//     border: '1px solid #fecaca',
//   },

//   statusDot: {
//     width: '9px',
//     height: '9px',
//     borderRadius: '50%',
//     background: 'currentColor',
//   },

//   /* HERO */

//   hero: {
//     maxWidth: '1100px',
//     margin: '45px auto 30px',
//     padding: '0 25px',
//     display: 'flex',
//     alignItems: 'center',
//     gap: '25px',
//   },

//   heroIcon: {
//     width: '85px',
//     height: '85px',
//     borderRadius: '22px',
//     background: '#111827',
//     display: 'flex',
//     alignItems: 'center',
//     justifyContent: 'center',
//     fontSize: '42px',
//     boxShadow:
//       '0 10px 30px rgba(0,0,0,0.15)',
//     flexShrink: 0,
//   },

//   title: {
//     margin: 0,
//     fontSize: '42px',
//     lineHeight: '1.15',
//     fontWeight: '800',
//     letterSpacing: '-1px',
//   },

//   description: {
//     maxWidth: '750px',
//     color: '#6b7280',
//     fontSize: '16px',
//     lineHeight: '1.7',
//     marginTop: '14px',
//   },

//   /* PIPELINE */

//   pipeline: {
//     maxWidth: '1100px',
//     margin: '30px auto 45px',
//     padding: '18px 20px',
//     display: 'flex',
//     alignItems: 'center',
//     justifyContent: 'space-between',
//     background: '#ffffff',
//     border: '1px solid #e5e7eb',
//     borderRadius: '16px',
//     boxShadow:
//       '0 5px 20px rgba(0,0,0,0.05)',
//   },

//   pipelineStep: {
//     display: 'flex',
//     flexDirection: 'column',
//     alignItems: 'center',
//     gap: '4px',
//     minWidth: '90px',
//     fontSize: '13px',
//   },

//   pipelineIcon: {
//     width: '45px',
//     height: '45px',
//     borderRadius: '12px',
//     background: '#f3f4f6',
//     display: 'flex',
//     alignItems: 'center',
//     justifyContent: 'center',
//     fontSize: '21px',
//     marginBottom: '3px',
//   },

//   pipelineText: {
//     color: '#6b7280',
//     fontSize: '11px',
//   },

//   arrow: {
//     color: '#9ca3af',
//     fontSize: '24px',
//     fontWeight: 'bold',
//   },

//   /* MAIN */

//   main: {
//     maxWidth: '900px',
//     margin: '0 auto',
//     padding: '0 20px 50px',
//   },

//   sectionHeader: {
//     marginBottom: '10px',
//   },

//   sectionTitle: {
//     fontSize: '28px',
//     marginBottom: '6px',
//   },

//   sectionText: {
//     color: '#6b7280',
//     marginTop: 0,
//   },

//   /* FOOTER */

//   footer: {
//     textAlign: 'center',
//     padding: '30px 20px',
//     borderTop: '1px solid #e5e7eb',
//     background: '#ffffff',
//     color: '#6b7280',
//     fontSize: '14px',
//   },

//   footerSmall: {
//     marginTop: '5px',
//     fontSize: '12px',
//     color: '#9ca3af',
//   },
// }

// export default App
import { useEffect, useState } from 'react'
import {
  Shield,
  AudioWaveform,
  Brain,
  MessageSquareText,
  ShieldAlert,
  Link as LinkIcon,
} from 'lucide-react'
import AudioUploader from './components/AudioUploader'
import './App.css'

function App() {
  const [status, setStatus] = useState('Checking...')

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/health')
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'ok') {
          setStatus('Connected')
        } else {
          setStatus('Unknown')
        }
      })
      .catch(() => {
        setStatus('Not Connected')
      })
  }, [])

  const isConnected = status === 'Connected'

  return (
    <div className="app">
      {/* HEADER */}
      <header className="header">
        <div className="header-brand">
          <div className="header-brand-icon">
            <Shield size={18} strokeWidth={2.2} />
          </div>
          <div>
            <div className="header-brand-name">VoiceShield</div>
            <div className="header-brand-tag">AI Voice Security</div>
          </div>
        </div>

        <div className="header-right">
          <div className="status-pill is-ok">
            <span className="status-dot" />
            System Operational
          </div>

          <div className={`status-pill ${isConnected ? 'is-ok' : 'is-bad'}`}>
            <span className="status-dot" />
            {isConnected ? 'Backend Connected' : status}
          </div>
        </div>
      </header>

      {/* HERO */}
      <section className="hero">
        <div>
          <div className="hero-label">AI-POWERED VOICE SECURITY</div>
          <h1 className="hero-title">
            Voice Impersonation
            <br />
            Protection System
          </h1>
          <p className="hero-description">
            Detect AI-generated voices, identify fraud indicators, assess
            caller risk, and preserve tamper-evident analysis evidence.
          </p>
        </div>

        <div className="hero-visual" aria-hidden="true">
          <div className="hero-waveform">
            {[18, 34, 52, 70, 96, 70, 52, 80, 46, 30, 60, 40, 22].map(
              (h, i) => (
                <span
                  key={i}
                  className="hero-waveform-bar"
                  style={{ height: `${h}%`, animationDelay: `${i * 0.09}s` }}
                />
              )
            )}
          </div>
        </div>
      </section>

      {/* SECURITY PIPELINE */}
      <section className="pipeline">
        <div className="pipeline-track">
          <PipelineStage
            icon={<AudioWaveform size={16} />}
            name="Audio"
            desc="Upload & preprocess"
            delay={0}
          />
          <PipelineStage
            icon={<Brain size={16} />}
            name="AI Voice Detection"
            desc="Clone / deepfake analysis"
            delay={0.08}
          />
          <PipelineStage
            icon={<MessageSquareText size={16} />}
            name="Speech-to-Text"
            desc="Transcription"
            delay={0.16}
          />
          <PipelineStage
            icon={<ShieldAlert size={16} />}
            name="Fraud Detection"
            desc="Keyword & intent scan"
            delay={0.24}
          />
          <PipelineStage
            icon={<Shield size={16} />}
            name="Risk Engine"
            desc="Composite scoring"
            delay={0.32}
          />
          <PipelineStage
            icon={<LinkIcon size={16} />}
            name="Evidence Ledger"
            desc="Tamper-evident record"
            delay={0.4}
          />
        </div>
      </section>

      {/* MAIN ANALYZER */}
      <main className="main">
        <div className="section-heading">
          <h2>Analyze a Voice Recording</h2>
          <p>
            Upload an audio recording to perform AI voice detection,
            transcription, fraud analysis and risk assessment.
          </p>
        </div>

        <AudioUploader />
      </main>

      {/* FOOTER */}
      <footer className="footer">
        <p>
          <strong>VoiceShield</strong> — AI Voice Security & Evidence
          Protection
        </p>
        <p style={{ marginTop: 4 }}>Built for Smart India Hackathon</p>
      </footer>
    </div>
  )
}

function PipelineStage({ icon, name, desc, delay }) {
  return (
    <div className="pipeline-stage" style={{ animationDelay: `${delay}s` }}>
      <div className="pipeline-stage-icon">{icon}</div>
      <div className="pipeline-stage-name">{name}</div>
      <div className="pipeline-stage-desc">{desc}</div>
    </div>
  )
}

export default App