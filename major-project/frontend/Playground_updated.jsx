import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Zap, Clock, DollarSign, Leaf, AlertCircle, MessageSquare, Cpu, FlaskConical } from 'lucide-react'
import { api } from '../lib/api'
import './Playground.css'

const SAMPLES = [
  { text: 'What is 2 + 2?',                                     label: 'Trivial' },
  { text: 'How many days are in a week?',                        label: 'Simple' },
  { text: 'Explain supply and demand economics.',                 label: 'Medium' },
  { text: 'Describe the water cycle in detail.',                  label: 'Medium' },
  { text: 'Discuss Gödel\'s incompleteness theorems.',            label: 'Complex' },
  { text: 'Write a Python quicksort implementation with tests.',  label: 'Complex' },
]

const STAGES = [
  { id: 'scoring',  label: 'Complexity Scorer', sub: 'spaCy · textstat · 5 NLP features', color: 'var(--green-400)' },
  { id: 'fuzzy',    label: 'Fuzzy Controller',  sub: 'scikit-fuzzy · centroid defuzz',     color: 'var(--indigo-400)' },
  { id: 'routellm', label: 'RouteLLM Bridge',   sub: 'MF router · MID-zone bypass',        color: 'var(--indigo-400)' },
  { id: 'cascade',  label: 'Cascade Inference', sub: 'FrugalGPT · LoRA adapters',          color: 'var(--amber-400)' },
  { id: 'energy',   label: 'Energy Tracker',    sub: 'CodeCarbon · joules / inference',    color: 'var(--green-400)' },
]

const STAGE_ORDER = ['scoring', 'fuzzy', 'routellm', 'cascade', 'energy']

const TIER_META = {
  '4bit':  { label: '4-bit',  color: 'var(--green-400)',  bg: 'rgba(34,197,94,0.1)',   border: 'rgba(34,197,94,0.25)',   gear: 'Gear 1 · Simple' },
  '8bit':  { label: '8-bit',  color: 'var(--indigo-400)', bg: 'rgba(129,140,248,0.1)', border: 'rgba(129,140,248,0.25)', gear: 'Gear 2 · Medium' },
  '16bit': { label: '16-bit', color: 'var(--amber-400)',  bg: 'rgba(251,191,36,0.1)',  border: 'rgba(251,191,36,0.25)',  gear: 'Gear 3 · Complex' },
}

function FeatureGauge({ label, value }) {
  const pct = Math.min(Math.max(value, 0), 1) * 100
  const color = value > 0.66 ? 'var(--amber-400)' : value > 0.33 ? 'var(--indigo-400)' : 'var(--green-400)'
  return (
    <div className="feature-gauge">
      <div className="feature-gauge-header">
        <span className="feature-gauge-label">{label.replace(/_/g, ' ')}</span>
        <span className="feature-gauge-value mono">{value.toFixed(3)}</span>
      </div>
      <div className="feature-gauge-track">
        <motion.div
          className="feature-gauge-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          style={{ background: color }}
        />
      </div>
    </div>
  )
}

function PipelineStage({ stage, activeStage }) {
  const idx = STAGE_ORDER.indexOf(stage.id)
  const activeIdx = STAGE_ORDER.indexOf(activeStage)
  const isDone = activeStage === 'done' || (activeIdx > idx && activeIdx !== -1)
  const isActive = activeStage === stage.id

  return (
    <div className={`pipeline-stage ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}>
      <div className="pipeline-stage-indicator" style={{ '--stage-color': stage.color }}>
        {isDone ? (
          <motion.svg initial={{ scale: 0 }} animate={{ scale: 1 }} width="12" height="12" viewBox="0 0 12 12">
            <path d="M2 6l3 3 5-5" stroke={stage.color} strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </motion.svg>
        ) : isActive ? (
          <div className="pipeline-stage-pulse" style={{ background: stage.color }} />
        ) : (
          <div className="pipeline-stage-dot" />
        )}
      </div>
      <div className="pipeline-stage-text">
        <span className="pipeline-stage-name">{stage.label}</span>
        <span className="pipeline-stage-sub">{stage.sub}</span>
      </div>
    </div>
  )
}

function StatChip({ icon: Icon, label, value, color }) {
  return (
    <div className="stat-chip">
      <Icon size={13} color={color} strokeWidth={2} />
      <div>
        <div className="stat-chip-value mono" style={{ color }}>{value}</div>
        <div className="stat-chip-label">{label}</div>
      </div>
    </div>
  )
}

export function Playground({ online, gpuReady }) {
  const [prompt, setPrompt] = useState('')
  const [activeStage, setActiveStage] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const textRef = useRef(null)

  async function run() {
    if (!prompt.trim() || !online) return
    setError(null)
    setResult(null)

    try {
      setActiveStage('scoring')
      await delay(300)
      setActiveStage('fuzzy')
      await delay(250)
      setActiveStage('routellm')

      // Always request a response (routing_only: false)
      // Backend decides: real GPU inference or mock
      const data = await api.infer({ prompt, routing_only: false })

      await delay(200)
      setActiveStage('cascade')
      await delay(350)
      setActiveStage('energy')
      await delay(200)
      setActiveStage('done')

      setResult(data)
      setHistory(h => [{ ...data, ts: Date.now() }, ...h].slice(0, 30))
    } catch (e) {
      setError(e.message)
      setActiveStage(null)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run()
  }

  const tier = result?.final_tier
  const tm = tier ? TIER_META[tier] : null
  const isRunning = activeStage && activeStage !== 'done'

  return (
    <div className="playground">
      {/* Left column */}
      <div className="playground-left">

        {/* GPU status banner */}
        {online && (
          <div className={`inference-mode-banner ${gpuReady ? 'gpu' : 'mock'}`}>
            {gpuReady
              ? <><Cpu size={13} strokeWidth={2} /> GPU ready — real inference via model pool</>
              : <><FlaskConical size={13} strokeWidth={2} /> No GPU detected — mock responses will be shown</>
            }
          </div>
        )}

        {/* Sample prompts */}
        <div className="samples-row">
          {SAMPLES.map(s => (
            <button
              key={s.text}
              className={`sample-btn complexity-${s.label.toLowerCase()}`}
              onClick={() => { setPrompt(s.text); textRef.current?.focus() }}
            >
              <span className="sample-complexity">{s.label}</span>
              {s.text.length > 38 ? s.text.slice(0, 38) + '…' : s.text}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="input-card">
          <div className="input-card-header">
            <span className="input-card-title">Prompt</span>
            <span className="input-card-hint mono">POST /api/infer</span>
          </div>
          <textarea
            ref={textRef}
            className="prompt-textarea"
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Enter a prompt to route through the Green-Weight pipeline…"
            rows={5}
          />
          <div className="input-card-footer">
            <span className="input-hint">{online ? '⌘↵ to run' : '⚠ backend offline — start api.py first'}</span>
            <button
              className={`run-btn ${isRunning || !online || !prompt.trim() ? 'disabled' : ''}`}
              onClick={run}
              disabled={isRunning || !online || !prompt.trim()}
            >
              {isRunning ? (
                <><div className="run-spinner" />Running…</>
              ) : (
                <><Send size={14} strokeWidth={2} />Run Pipeline</>
              )}
            </button>
          </div>
        </div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div className="error-banner" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <AlertCircle size={14} />
              <span>{error}</span>
              <button onClick={() => setError(null)}>✕</button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Result */}
        <AnimatePresence>
          {result && activeStage === 'done' && (
            <motion.div
              className="result-card"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
            >
              {/* Tier + badges */}
              <div className="result-header">
                <div className="result-tier-badge" style={{ background: tm.bg, border: `1px solid ${tm.border}`, color: tm.color }}>
                  <span className="result-tier-label mono">{tm.label}</span>
                  <span className="result-tier-gear">{tm.gear}</span>
                </div>
                {tier !== '16bit' && (
                  <div className="green-badge">
                    <Leaf size={11} strokeWidth={2.5} />
                    Green inference
                  </div>
                )}
                {result.is_mock && (
                  <div className="mock-badge">
                    <FlaskConical size={10} strokeWidth={2} />
                    Mock · no GPU
                  </div>
                )}
                <div className="result-routing mono">
                  fuzzy: <span style={{ color: TIER_META[result.fuzzy_tier]?.color }}>{result.fuzzy_tier}</span>
                  &nbsp;→&nbsp;final: <span style={{ color: tm.color }}>{tier}</span>
                  &nbsp;·&nbsp;p={result.win_probability?.toFixed(3)}
                </div>
              </div>

              {/* ── RESPONSE SECTION ── */}
              <div className="response-section">
                <div className="response-section-header">
                  <MessageSquare size={13} color={tm.color} strokeWidth={2} />
                  <span className="result-section-title">
                    Response
                    {result.is_mock && <span className="response-mock-note"> (mock — connect GPU for real inference)</span>}
                  </span>
                </div>
                <div className="response-body" style={{ borderColor: tm.border }}>
                  {result.response
                    ? result.response
                    : <span className="response-empty">No response returned — routing-only mode</span>
                  }
                </div>
              </div>

              {/* Features */}
              <div className="result-section-title" style={{ marginTop: 20 }}>Complexity Features</div>
              <div className="features-grid">
                {result.features && Object.entries(result.features).map(([k, v]) => (
                  <FeatureGauge key={k} label={k} value={v} />
                ))}
              </div>

              {/* Stats */}
              <div className="stats-row">
                <StatChip icon={Zap}        label="Energy"  value={`${result.energy_joules}J`}       color="var(--green-400)"  />
                <StatChip icon={Clock}      label="Latency" value={`${result.latency_ms}ms`}          color="var(--indigo-400)" />
                <StatChip icon={DollarSign} label="Cost"    value={`$${result.cost_usd?.toFixed(4)}`} color="var(--amber-400)"  />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* History */}
        {history.length > 0 && (
          <div className="history-card">
            <div className="history-title">
              Recent Inferences
              <span className="history-count">{history.length}</span>
            </div>
            <div className="history-list">
              {history.slice(0, 8).map((h, i) => (
                <div key={i} className="history-item">
                  <div className="history-item-tier" style={{ background: TIER_META[h.final_tier]?.bg, color: TIER_META[h.final_tier]?.color }}>
                    {h.final_tier}
                  </div>
                  <span className="history-item-prompt">{h.prompt.length > 55 ? h.prompt.slice(0, 55) + '…' : h.prompt}</span>
                  <span className="history-item-response mono">
                    {h.response ? h.response.slice(0, 30) + '…' : '—'}
                  </span>
                  <span className="history-item-energy mono">{h.energy_joules}J</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Right column: pipeline viz */}
      <div className="playground-right">
        <div className="pipeline-card">
          <div className="pipeline-header">
            <span className="pipeline-title">Live Pipeline</span>
            {activeStage && activeStage !== 'done' && <span className="pipeline-running mono">running</span>}
            {activeStage === 'done' && <span className="pipeline-done mono">complete</span>}
          </div>

          {result && (
            <div className="winprob-section">
              <div className="winprob-header">
                <span className="winprob-label">Win Probability</span>
                <span className="winprob-val mono">{(result.win_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="winprob-track">
                <motion.div
                  className="winprob-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.win_probability * 100}%` }}
                  transition={{ duration: 0.7, ease: 'easeOut' }}
                  style={{ background: tm?.color }}
                />
              </div>
              <div className="winprob-hint mono">
                {result.win_probability < 0.33 ? '→ 4-bit direct' : result.win_probability > 0.66 ? '→ 16-bit direct' : '→ MID zone → 8-bit bypass'}
              </div>
            </div>
          )}

          <div className="pipeline-stages">
            {STAGES.map((stage, idx) => (
              <div key={stage.id}>
                <PipelineStage stage={stage} activeStage={activeStage} />
                {idx < STAGES.length - 1 && <div className="pipeline-connector" />}
              </div>
            ))}
          </div>

          {activeStage && (
            <div className="tier-selector">
              <div className="tier-selector-label">Routed to</div>
              <div className="tier-selector-options">
                {['4bit', '8bit', '16bit'].map(t => {
                  const active = (result?.final_tier || '') === t
                  const tm2 = TIER_META[t]
                  return (
                    <motion.div
                      key={t}
                      className={`tier-option ${active ? 'active' : ''}`}
                      style={active ? { background: tm2.bg, border: `1px solid ${tm2.border}`, color: tm2.color } : {}}
                      animate={{ scale: active ? 1.03 : 1 }}
                    >
                      <span className="mono">{t}</span>
                    </motion.div>
                  )
                })}
              </div>
            </div>
          )}

          {isRunning && (
            <div className="scan-line-container">
              <div className="scan-line" />
            </div>
          )}
        </div>

        <div className="config-peek">
          <div className="config-peek-title">Active config.yaml</div>
          <pre className="config-peek-pre mono">{`router:
  routellm:
    router_type: mf
    mid_zone: [0.33, 0.66]
cascade:
  judger:
    threshold: 0.5
model:
  base: llama-2-7b-hf`}</pre>
        </div>
      </div>
    </div>
  )
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)) }
