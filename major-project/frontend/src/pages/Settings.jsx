import { useState } from 'react'
import { motion } from 'framer-motion'
import { Leaf, RotateCcw, Target, Activity } from 'lucide-react'
import { useAnalytics } from '../hooks/useBackend'
import './Settings.css'

const TIER_ORDER = { '4bit': 0, '8bit': 1, '16bit': 2 }

const DEFAULTS = { accuracy: 70, energy: 50, judger: 50 }

function Slider({ label, desc, value, onChange, left, right, color, unit = '' }) {
  return (
    <div className="slider-group">
      <div className="slider-header">
        <div>
          <div className="slider-label">{label}</div>
          <div className="slider-desc">{desc}</div>
        </div>
        <div className="slider-val mono" style={{ color }}>{value}{unit}</div>
      </div>
      <div className="slider-track-wrap">
        <input
          type="range" min={0} max={100} value={value}
          onChange={e => onChange(+e.target.value)}
          className="slider-input"
          style={{ '--slider-color': color, '--slider-pct': `${value}%` }}
        />
        <div className="slider-ends">
          <span>{left}</span>
          <span>{right}</span>
        </div>
      </div>
    </div>
  )
}

function TierBar({ label, pct, color }) {
  return (
    <div className="tier-bar-row">
      <div className="tier-bar-header">
        <span className="tier-bar-label">{label}</span>
        <span className="tier-bar-pct mono" style={{ color }}>{pct}%</span>
      </div>
      <div className="tier-bar-track">
        <motion.div
          className="tier-bar-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{ background: color }}
        />
      </div>
    </div>
  )
}


/* ------------------------------------------------------------------ *
 * Measured metrics.  Everything here is derived from real trace rows
 * or from the accuracy evaluation's own output — nothing is modelled,
 * and a metric with no data shows an empty state rather than a number.
 * ------------------------------------------------------------------ */

function Metric({ label, value, unit, hint, accent }) {
  return (
    <div className="metric">
      <div className="metric-value mono" style={accent ? { color: accent } : undefined}>
        {value ?? '—'}<span className="metric-unit">{value != null ? unit : ''}</span>
      </div>
      <div className="metric-label">{label}</div>
      <div className="metric-hint">{hint}</div>
    </div>
  )
}

function pct(n, d) { return d ? Math.round((n / d) * 100) : null }

function RoutingQuality({ traces }) {
  const n = traces.length
  const scored = traces.filter(t => t.fuzzy_tier && (t.final_tier || t.tier))
  const agree = scored.filter(t => t.fuzzy_tier === (t.final_tier || t.tier)).length
  const escalated = scored.filter(
    t => TIER_ORDER[t.final_tier || t.tier] > TIER_ORDER[t.fuzzy_tier]
  ).length
  const deescalated = scored.filter(
    t => TIER_ORDER[t.final_tier || t.tier] < TIER_ORDER[t.fuzzy_tier]
  ).length
  const wp = traces.filter(t => typeof t.win_probability === 'number')
  const meanWp = wp.length
    ? (wp.reduce((s, t) => s + t.win_probability, 0) / wp.length)
    : null
  const lat = traces.filter(t => t.latency_ms > 0)
  const meanLat = lat.length
    ? Math.round(lat.reduce((s, t) => s + t.latency_ms, 0) / lat.length)
    : null

  if (!n) {
    return (
      <div className="metrics-empty">
        No questions routed yet. Run a few prompts in the Playground and these
        fill in from the real trace log.
      </div>
    )
  }

  return (
    <>
      <div className="metrics-grid">
        <Metric
          label="Questions routed" value={n} unit=""
          hint="Rows in pipeline_trace.jsonl for this session."
        />
        <Metric
          label="Tier agreement" value={pct(agree, scored.length)} unit="%"
          accent="var(--green-400)"
          hint="How often the fuzzy controller's tier survived the RouteLLM bridge unchanged."
        />
        <Metric
          label="Escalation rate" value={pct(escalated, scored.length)} unit="%"
          accent="var(--amber-400)"
          hint="Questions the bridge moved to a higher precision than the fuzzy score asked for."
        />
        <Metric
          label="De-escalation rate" value={pct(deescalated, scored.length)} unit="%"
          accent="var(--indigo-400)"
          hint="Questions moved down to a cheaper tier."
        />
        <Metric
          label="Mean win probability" value={meanWp != null ? meanWp.toFixed(3) : null} unit=""
          hint="RouteLLM's confidence that the stronger model would win. Above the mid-zone it routes up."
        />
        <Metric
          label="Mean routing latency" value={meanLat} unit=" ms"
          hint="Time to read a question and choose a tier — the overhead routing adds."
        />
      </div>
      <p className="metrics-note">
        Measured from this session's own traces. They describe the router's
        behaviour, not answer quality — that comes from the benchmark below.
      </p>
    </>
  )
}

function BenchmarkAccuracy({ accuracy }) {
  const byCondition = accuracy?.accuracy_by_condition
  const perTier = accuracy && !byCondition
    ? Object.entries(accuracy).filter(([, v]) => v && typeof v === 'object')
    : []

  if (!byCondition && !perTier.length) {
    return (
      <div className="metrics-empty">
        <p>
          No accuracy evaluation has been run yet, so there are no answer-quality
          numbers to show. These appear once
          {' '}<code className="mono">training/scripts/kaggle_accuracy_eval.py</code>{' '}
          has completed and the results are available.
        </p>
        <ul className="metrics-planned">
          <li><strong>Accuracy per tier</strong> — how often each precision level answers correctly, per benchmark task.</li>
          <li><strong>Accuracy retention</strong> — routed accuracy as a share of full-precision accuracy. This is the number that decides whether the energy saving cost anything.</li>
          <li><strong>Per-task breakdown</strong> — arc_easy, gsm8k and hellaswag separately; a router can hold up on one and collapse on another.</li>
          <li><strong>QAT adapter delta</strong> — accuracy with the fine-tuned adapter minus the base model, at the same bit-width.</li>
          <li><strong>Energy per correct answer</strong> — joules divided by correct answers, which is the only fair way to compare tiers that differ in both.</li>
        </ul>
      </div>
    )
  }

  const rows = byCondition
    ? Object.entries(byCondition).map(([name, v]) => ({
        name,
        cells: Object.entries(v).filter(([, x]) => typeof x === 'number'),
      }))
    : perTier.map(([name, tasks]) => ({
        name,
        cells: Object.entries(tasks).flatMap(([task, metrics]) =>
          Object.entries(metrics || {})
            .filter(([m, x]) => typeof x === 'number' && /acc/.test(m))
            .map(([m, x]) => [`${task} ${m}`, x])
        ),
      }))

  return (
    <div className="metrics-table-wrap">
      <table className="metrics-table">
        <thead>
          <tr><th>Condition</th><th>Metric</th><th className="num">Value</th></tr>
        </thead>
        <tbody>
          {rows.flatMap(r => r.cells.map(([m, v], i) => (
            <tr key={r.name + m}>
              <td>{i === 0 ? r.name.replace(/_/g, ' ') : ''}</td>
              <td className="mono">{m}</td>
              <td className="num mono">{v < 1 ? (v * 100).toFixed(1) + '%' : v.toFixed(3)}</td>
            </tr>
          )))}
        </tbody>
      </table>
    </div>
  )
}

export function Settings() {
  const [vals, setVals] = useState(DEFAULTS)
  const { traces, accuracy } = useAnalytics()
  const set = (k) => (v) => setVals(prev => ({ ...prev, [k]: v }))

  const pct4  = Math.round(50 - vals.accuracy * 0.4)
  const pct16 = Math.round(5  + vals.accuracy * 0.35)
  const pct8  = 100 - pct4 - pct16

  const savings = Math.round(75 - vals.accuracy * 0.35)
  const threshold = (0.3 + vals.judger / 100 * 0.4).toFixed(2)
  const upper4 = Math.round(20 + (100 - vals.accuracy) * 0.2)
  const upper8 = Math.round(55 + (100 - vals.accuracy) * 0.2)

  const configYaml = `router:
  fuzzy_controller:
    tier_thresholds:
      4bit_upper: ${upper4}
      8bit_upper: ${upper8}
      16bit_lower: ${upper8 + 1}
  routellm:
    router_type: mf
    mid_zone_lower: 0.33
    mid_zone_upper: 0.66

cascade:
  judger:
    threshold: ${threshold}
  service_order:
    - local/4bit
    - local/8bit
    - local/16bit

model:
  base_model_id: meta-llama/Llama-3.2-1B
  lazy_16bit: ${vals.energy > 60 ? 'false' : 'true'}
  max_new_tokens: 256

benchmark:
  tasks: [mmlu, hellaswag]`

  return (
    <div className="settings">
      <div className="settings-header">
        <h1 className="settings-title display">System Configuration</h1>
        <p className="settings-subtitle mono">Adjust routing thresholds — changes preview config.yaml below</p>
      </div>

      <div className="settings-grid">
        {/* Controls */}
        <div className="settings-panel">
          <div className="panel-title display">Routing Thresholds</div>

          <Slider
            label="Accuracy Requirement"
            desc="Higher = more prompts escalate to 16-bit"
            value={vals.accuracy}
            onChange={set('accuracy')}
            left="Lenient"
            right="Strict (prefer 16-bit)"
            color="var(--amber-400)"
          />

          <Slider
            label="Energy Tolerance"
            desc="Higher = allow more expensive inference"
            value={vals.energy}
            onChange={set('energy')}
            left="Strict (prefer 4-bit)"
            right="Flexible"
            color="var(--green-400)"
          />

          <Slider
            label="Judger Threshold"
            desc="Lower = escalate less often = more energy savings"
            value={vals.judger}
            onChange={set('judger')}
            left="Lenient"
            right="Strict"
            color="var(--indigo-400)"
          />

          <button className="reset-btn" onClick={() => setVals(DEFAULTS)}>
            <RotateCcw size={12} strokeWidth={2} />
            Reset to defaults
          </button>
        </div>

        {/* Predictor */}
        <div className="settings-panel">
          <div className="panel-title display">Live Impact Predictor</div>
          <div className="panel-sub mono">Predicted routing distribution with current settings</div>

          <div className="tier-bars">
            <TierBar label="4-bit (Gear 1 · Simple)"   pct={pct4}  color="var(--green-400)"  />
            <TierBar label="8-bit (Gear 2 · Medium)"   pct={pct8}  color="var(--indigo-400)" />
            <TierBar label="16-bit (Gear 3 · Complex)" pct={pct16} color="var(--amber-400)"  />
          </div>

          <div className="savings-banner">
            <div className="savings-banner-icon">
              <Leaf size={16} color="var(--green-400)" strokeWidth={2} />
            </div>
            <div>
              <div className="savings-banner-title">Predicted Energy Savings</div>
              <div className="savings-banner-val mono">~{savings}% vs <span className="nowrap">always-16-bit</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* Config YAML preview */}
      <motion.div
        className="yaml-preview"
        key={configYaml}
        initial={{ opacity: 0.6 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
      >
        <div className="yaml-header">
          <div>
            <div className="yaml-title display">config.yaml preview</div>
            <div className="yaml-sub mono">Updates live as you move the sliders above</div>
          </div>
          <button
            className="yaml-copy-btn"
            onClick={() => navigator.clipboard.writeText(configYaml)}
          >
            Copy
          </button>
        </div>
        <pre className="yaml-body mono">{configYaml}</pre>
      </motion.div>

      <div className="settings-panel wide">
        <div className="panel-title display">
          <Activity size={15} strokeWidth={2} /> Routing quality
        </div>
        <div className="panel-sub mono">From this session's routing trace</div>
        <RoutingQuality traces={traces} />
      </div>

      <div className="settings-panel wide">
        <div className="panel-title display">
          <Target size={15} strokeWidth={2} /> Precision &amp; accuracy benchmark
        </div>
        <div className="panel-sub mono">From accuracy evaluation results</div>
        <BenchmarkAccuracy accuracy={accuracy} />
      </div>
    </div>
  )
}
