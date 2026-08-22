import { motion } from 'framer-motion'
import { RefreshCw, TrendingDown, Cpu, Target, Database, Zap } from 'lucide-react'
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, BarChart, Bar, LineChart, Line, Cell, Legend
} from 'recharts'
import { useAnalytics } from '../hooks/useBackend'
import './Analytics.css'

const TIER_COLORS = { '4bit': '#4f7942', '8bit': '#b06a45', '16bit': '#b3873a' }

const FALLBACK_SCATTER = [
  { name: 'Always 4-bit',  accuracy: 0.50, energy: 8.5 },
  { name: 'Always 8-bit',  accuracy: 0.63, energy: 28.5 },
  { name: 'Always 16-bit', accuracy: 0.75, energy: 105 },
  { name: 'Routed (ours)', accuracy: 0.73, energy: 36.8 },
]

function KpiCard({ icon: Icon, label, value, sub, accent, delay = 0 }) {
  return (
    <motion.div
      className="kpi-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35 }}
      style={{ '--accent': accent }}
    >
      <div className="kpi-icon">
        <Icon size={16} color={accent} strokeWidth={2} />
      </div>
      <div className="kpi-body">
        <div className="kpi-value mono">{value}</div>
        <div className="kpi-label">{label}</div>
        {sub && <div className="kpi-sub mono">{sub}</div>}
      </div>
      <div className="kpi-glow" />
    </motion.div>
  )
}

function ChartCard({ title, sub, children, delay = 0 }) {
  return (
    <motion.div
      className="chart-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <div className="chart-card-header">
        <div>
          <div className="chart-card-title display">{title}</div>
          <div className="chart-card-sub mono">{sub}</div>
        </div>
      </div>
      {children}
    </motion.div>
  )
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-name">{d.name}</div>
      {d.accuracy !== undefined && <div>Accuracy: <span className="mono">{(d.accuracy * 100).toFixed(1)}%</span></div>}
      {d.energy   !== undefined && <div>Energy:   <span className="mono">{d.energy}J</span></div>}
      {d.count    !== undefined && <div>Queries:  <span className="mono">{d.count}</span></div>}
    </div>
  )
}

export function Analytics() {
  const { traces, accuracy, loading, error, refresh } = useAnalytics()

  // Derive from real data
  const tierCounts = traces.reduce((acc, t) => {
    const tier = t.final_tier || t.tier
    if (tier) acc[tier] = (acc[tier] || 0) + 1
    return acc
  }, {})
  const barData = Object.entries(tierCounts).map(([name, count]) => ({ name, count }))
  const lineData = traces.slice(-24).map((t, i) => ({
    i: i + 1,
    energy: parseFloat(t.energy_joules || 0),
    tier: t.final_tier || t.tier,
  }))
  // Only real, measured energy counts toward the savings KPI — mock
  // placeholder energy (is_mock: true, api.py's fixed {4bit:8.5, 8bit:28.5,
  // 16bit:105.2} demo values) and unmeasured routing-only calls
  // (energy_joules: 0) must not be presented as a real result. Previously
  // this divided by traces.length unconditionally, so an all-routing-only
  // session (every energy_joules === 0) showed a fabricated "100% saved"
  // KPI — found 2026-08-22 auditing the frontend, matches the exact
  // known risk already flagged in verification/checklist.md about these
  // mock placeholder values leaking into anything presented as real.
  const realTraces = traces.filter(t => !t.is_mock && parseFloat(t.energy_joules || 0) > 0)
  const totalJ = realTraces.reduce((s, t) => s + parseFloat(t.energy_joules || 0), 0)
  const savings = realTraces.length > 0
    ? Math.round((1 - totalJ / (realTraces.length * 105.2)) * 100)
    : null

  const accData = accuracy?.accuracy_by_condition
  const scatterData = accData
    ? Object.entries(accData).map(([k, v]) => ({
        name: k.replace(/_/g, ' '),
        accuracy: v.overall,
        energy: k === 'routed' ? 36.8 : k === 'always_4bit' ? 8.5 : k === 'always_8bit' ? 28.5 : 105,
      }))
    : FALLBACK_SCATTER

  const routedAcc = accData?.routed
  const apgr = accuracy?.routellm_metrics?.APGR

  return (
    <div className="analytics">
      <div className="analytics-header">
        <div>
          <h1 className="analytics-title display">Evaluation Metrics</h1>
          <p className="analytics-subtitle mono">
            Live data from pipeline_trace.jsonl · accuracy_results.json · GET /api/*
          </p>
        </div>
        <button className={`refresh-btn ${loading ? 'loading' : ''}`} onClick={refresh} disabled={loading}>
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} strokeWidth={2} />
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="analytics-error mono">{error}</div>
      )}

      {/* KPIs */}
      <div className="kpi-grid">
        <KpiCard icon={TrendingDown} label="Energy Saved vs 16-bit"   value={savings !== null ? `~${savings}%` : '—'}     sub={savings !== null ? `${totalJ.toFixed(1)}J measured · vs assumed 105.2J baseline` : 'no real measurements yet'} accent="var(--green-400)"  delay={0}    />
        <KpiCard icon={Database}     label="Queries Processed"         value={traces.length || '—'}                         sub="GET /api/results"                        accent="var(--indigo-400)" delay={0.05} />
        <KpiCard icon={Target}       label="Routed MMLU Accuracy"      value={routedAcc ? (routedAcc.mmlu * 100).toFixed(1) + '%' : '—'} sub="GET /api/accuracy"          accent="var(--amber-400)"  delay={0.1}  />
        <KpiCard icon={Cpu}          label="APGR"                      value={apgr ?? '—'}                                  sub="avg perf gap recovered"                  accent="var(--cyan-400)"   delay={0.15} />
      </div>

      {/* Charts row 1 */}
      <div className="charts-row-2">
        {/* Note: the energy axis is always the same fixed demo constants
            (36.8/8.5/28.5/105J) regardless of accData — accuracy_results.json
            only carries per-condition accuracy, not energy, and no live
            per-condition energy endpoint exists yet (that needs Session 4's
            routing_conditions_summary.csv). Labeled accordingly instead of
            claiming "live" for a chart that's still half-fabricated —
            found 2026-08-22 auditing the frontend. */}
        <ChartCard title="Energy – Accuracy Trade-off" sub={accData ? "accuracy: live · energy: modeled (Session 4 pending)" : "fallback expected values"} delay={0.2}>
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(37,45,34,0.08)" />
              <XAxis dataKey="accuracy" type="number" domain={[0.4, 0.8]}
                tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                label={{ value: 'Accuracy', position: 'insideBottom', offset: -10, fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis dataKey="energy" type="number"
                tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                label={{ value: 'J', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-muted)' }} />
              <Tooltip content={<CustomTooltip />} />
              <Scatter data={scatterData}
                shape={(props) => {
                  const isRouted = props.payload.name?.toLowerCase().includes('routed')
                  const r = isRouted ? 11 : 7
                  return (
                    <g>
                      {isRouted && <circle cx={props.cx} cy={props.cy} r={r + 6} fill="rgba(79,121,66,0.12)" />}
                      <circle cx={props.cx} cy={props.cy} r={r}
                        fill={isRouted ? 'var(--green-400)' : 'var(--bg-hover)'}
                        stroke={isRouted ? 'var(--green-400)' : 'var(--border-default)'}
                        strokeWidth={isRouted ? 2 : 1} />
                    </g>
                  )
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Routing Distribution" sub={`live · ${traces.length} queries processed`} delay={0.25}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData.length ? barData : [{ name: '4bit', count: 30 }, { name: '8bit', count: 45 }, { name: '16bit', count: 25 }]}
              margin={{ top: 10, right: 10, bottom: 0, left: -10 }} barSize={40}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(37,45,34,0.08)" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {(barData.length ? barData : [{ name: '4bit' }, { name: '8bit' }, { name: '16bit' }]).map((entry) => (
                  <Cell key={entry.name} fill={TIER_COLORS[entry.name] || 'var(--green-400)'} opacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Energy line */}
      <ChartCard title="Energy per Inference" sub="last 24 queries · live · GET /api/results · pipeline_trace.jsonl" delay={0.3}>
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={lineData.length ? lineData : [{ i: 1, energy: 0 }]} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(37,45,34,0.08)" />
            <XAxis dataKey="i" tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} unit="J" />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="energy" stroke="var(--green-400)" strokeWidth={2}
              dot={(props) => {
                const tier = lineData[props.index]?.tier
                return <circle key={props.index} cx={props.cx} cy={props.cy} r={4} fill={TIER_COLORS[tier] || 'var(--green-400)'} stroke="var(--bg-surface)" strokeWidth={1.5} />
              }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Accuracy table */}
      {accData && (
        <ChartCard title="Accuracy by Condition" sub="GET /api/accuracy · accuracy_results.json" delay={0.35}>
          <table className="acc-table">
            <thead>
              <tr>
                {['Condition', 'MMLU', 'HellaSwag', 'Overall', 'vs 16-bit'].map(h => (
                  <th key={h} className="acc-th">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(accData).map(([cond, vals]) => {
                const isRouted = cond === 'routed'
                const baseline = accData.always_16bit?.overall || 0.75
                const delta = ((vals.overall - baseline) / baseline * 100).toFixed(1)
                return (
                  <tr key={cond} className={`acc-row ${isRouted ? 'highlighted' : ''}`}>
                    <td className="acc-td acc-cond">
                      {isRouted && <span className="acc-star">✦</span>}
                      {cond.replace(/_/g, ' ')}
                    </td>
                    <td className="acc-td mono">{(vals.mmlu * 100).toFixed(1)}%</td>
                    <td className="acc-td mono">{(vals.hellaswag * 100).toFixed(1)}%</td>
                    <td className="acc-td mono acc-bold">{(vals.overall * 100).toFixed(1)}%</td>
                    <td className={`acc-td mono ${delta >= 0 ? 'positive' : 'negative'}`}>
                      {delta >= 0 ? '+' : ''}{delta}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </ChartCard>
      )}
    </div>
  )
}
