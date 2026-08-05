import { useState } from 'react'
import { motion } from 'framer-motion'
import { Leaf, RotateCcw } from 'lucide-react'
import './Settings.css'

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

export function Settings() {
  const [vals, setVals] = useState(DEFAULTS)
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
  base_model_id: meta-llama/Llama-2-7b-hf
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
              <div className="savings-banner-val mono">~{savings}% vs always-16-bit</div>
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
    </div>
  )
}
