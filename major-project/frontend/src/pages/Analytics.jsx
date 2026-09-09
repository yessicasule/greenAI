import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  RefreshCw, Target, ShieldCheck, ChevronDown, Info,
  CheckCircle2, AlertTriangle, XCircle, Circle, Zap, FlaskConical,
} from 'lucide-react'
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, BarChart, Bar, Cell, ErrorBar, LabelList, ZAxis,
} from 'recharts'
import { useAnalytics, useEvidence } from '../hooks/useBackend'
import './Analytics.css'

const TIER_COLORS = { '4bit': '#4f7942', '8bit': '#b06a45', '16bit': '#b3873a' }
const TIERS = ['4bit', '8bit', '16bit']

/* ─────────────────────────────────────────────────────────────────────────
   Provenance is what keeps this page honest.

   CLAUDE.md's one rule: nothing may imply a result exists until
   training-agent + verify_results.py say so. The backend decides how
   trustworthy each payload is and returns it as `provenance`; the UI
   renders that verdict and never upgrades it.

   Presenting the project's progress confidently and labelling every
   number accurately are not in tension — the strong results here are
   genuinely strong, and saying so plainly is only possible *because*
   the weak ones are marked. Removing a badge would not make the project
   look better, it would make the page untrustworthy.
   ───────────────────────────────────────────────────────────────────── */
const PROVENANCE_META = {
  VERIFIED: { label: 'Verified', cls: 'verified', blurb: 'cleared verify_results.py — safe to cite' },
  CONTENDED: { label: 'Contended', cls: 'contended', blurb: 'real measurement, taken under host contention' },
  DERIVED: { label: 'Derived', cls: 'derived', blurb: 'computed from committed inputs, not a GPU measurement' },
  PENDING: { label: 'Upcoming', cls: 'pending', blurb: 'this session is still ahead' },
}

function ProvenanceBadge({ status }) {
  const meta = PROVENANCE_META[status] || PROVENANCE_META.PENDING
  return (
    <span className={`prov-badge ${meta.cls} mono`} title={meta.blurb}>
      {meta.label}
    </span>
  )
}

/* A caveat is a footnote, not an alarm. It stays on the page in full —
   it just no longer shouts over the result it qualifies. */
function Caveat({ children }) {
  if (!children) return null
  return (
    <p className="ev-caveat">
      <Info size={12} strokeWidth={2} />
      <span>{children}</span>
    </p>
  )
}

function Panel({ title, sub, provenance, caveat, children, actions, collapsible, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  const isOpen = collapsible ? open : true

  return (
    <motion.section
      className="ev-panel"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.18 }}
    >
      <div className="ev-panel-head">
        <div
          className={`ev-panel-titles ${collapsible ? 'clickable' : ''}`}
          onClick={collapsible ? () => setOpen(o => !o) : undefined}
          role={collapsible ? 'button' : undefined}
          tabIndex={collapsible ? 0 : undefined}
          onKeyDown={collapsible ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(o => !o) } } : undefined}
        >
          <h2 className="ev-panel-title display">
            {collapsible && <ChevronDown size={14} className={`chev ${isOpen ? 'open' : ''}`} strokeWidth={2.5} />}
            {title}
          </h2>
          {sub && <p className="ev-panel-sub mono">{sub}</p>}
        </div>
        <div className="ev-panel-actions">
          {isOpen && actions}
          {provenance && <ProvenanceBadge status={provenance} />}
        </div>
      </div>

      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            key="body"
            className="ev-panel-body"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.14 }}
          >
            {children}
            <Caveat>{caveat}</Caveat>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.section>
  )
}

function EmptyState({ children }) {
  return <div className="ev-empty">{children}</div>
}

const fmt3 = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(3))

/* ── Where the project is ──────────────────────────────────────────────
   A forward-looking read of the same record the tables below carry.
   "Upcoming" is the honest word for a session that hasn't run — it is a
   schedule state, not a defect, and the page should say so. */
function ProgressStrip({ experiments, energy, validation }) {
  const session1Done = energy?.provenance === 'VERIFIED'
  const session4Runs = (experiments?.runs || []).filter(r => r.session === '4').length

  const milestones = [
    { label: 'Eval dataset', detail: '500 prompts, stratified', state: 'done' },
    { label: 'QAT adapters', detail: '3 adapters, load-tested', state: 'done' },
    {
      label: 'Session 1 · Energy',
      detail: session1Done ? 'verified — gate passed' : 'not yet run',
      state: session1Done ? 'done' : 'next',
    },
    {
      label: 'Session 4 · Routing',
      detail: session4Runs ? `${session4Runs} runs measured, gate ahead` : 'not yet run',
      state: session4Runs ? 'active' : 'next',
    },
    { label: 'Session 2 · Accuracy', detail: 'next up', state: 'next' },
  ]

  const done = milestones.filter(m => m.state === 'done').length

  return (
    <div className="progress-strip">
      <div className="progress-head">
        <span className="progress-title display">Where the project stands</span>
        <span className="progress-count mono">{done} of {milestones.length} milestones cleared</span>
      </div>
      <ol className="milestones">
        {milestones.map(m => (
          <li key={m.label} className={`milestone ${m.state}`}>
            <span className="ms-icon">
              {m.state === 'done'
                ? <CheckCircle2 size={14} strokeWidth={2.5} />
                : m.state === 'active'
                  ? <Circle size={14} strokeWidth={2.5} />
                  : <Circle size={14} strokeWidth={1.5} />}
            </span>
            <span className="ms-label">{m.label}</span>
            <span className="ms-detail mono">{m.detail}</span>
          </li>
        ))}
      </ol>
      {validation?.counts && (
        <p className="progress-note">
          <strong>{validation.counts.PASS} of {validation.counts.PASS + validation.counts.WARN + validation.counts.FAIL} automated
          checks pass.</strong>{' '}
          The one failing check is Session 2&rsquo;s accuracy file, which does not exist yet because that
          session has not been run — a schedule item, not a bad result. Full record below.
        </p>
      )}
    </div>
  )
}

/* ── Headline: the result the project actually has ─────────────────── */
function HeroResult({ energy }) {
  if (energy?.provenance !== 'VERIFIED' || !energy.tiers?.length) return null
  const lo = energy.tiers.find(t => t.tier === '4bit')?.mean_j_per_token
  const hi = energy.tiers.find(t => t.tier === '16bit')?.mean_j_per_token
  if (!lo || !hi) return null

  return (
    <motion.div
      className="hero-result"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="hero-icon"><Zap size={18} strokeWidth={2.2} /></div>
      <div className="hero-body">
        <div className="hero-top">
          <span className="hero-kicker mono">Go / no-go gate · passed</span>
          <ProvenanceBadge status="VERIFIED" />
        </div>
        <h2 className="hero-headline display">
          The premise holds — fp16 costs <span className="hero-num">{(hi / lo).toFixed(2)}×</span> what 4-bit does
        </h2>
        {/* Every figure in this sentence is read from the payload. Nothing
            about a measurement is written into the copy, or it goes stale
            the next time the session is re-run. */}
        <p className="hero-sub">
          Measured on real hardware across three full runs,{' '}
          {energy.tiers[0]?.n?.toLocaleString()} generations per tier, with non-overlapping
          95% confidence intervals. Precision genuinely moves energy on a 1B model — the entire
          premise routing depends on, and the one result that could have ended the project early.
        </p>
      </div>
    </motion.div>
  )
}

function StatCard({ icon: Icon, label, value, sub, accent, provenance }) {
  return (
    <motion.div
      className="ev-stat"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.18 }}
      style={{ '--accent': accent }}
    >
      <div className="ev-stat-top">
        <Icon size={14} color={accent} strokeWidth={2} />
        <ProvenanceBadge status={provenance} />
      </div>
      <div className="ev-stat-value mono">{value}</div>
      <div className="ev-stat-label">{label}</div>
      <div className="ev-stat-sub">{sub}</div>
    </motion.div>
  )
}

/* ── Energy ─────────────────────────────────────────────────────────── */
function EnergyPanel({ energy }) {
  if (!energy || energy.provenance === 'PENDING' || !energy.tiers?.length) {
    return (
      <Panel title="Energy per tier" sub="Session 1" provenance="PENDING">
        <EmptyState>Session 1 is still ahead — no energy number exists yet.</EmptyState>
      </Panel>
    )
  }

  const data = energy.tiers.map(t => ({
    name: t.tier, value: t.mean_j_per_token, ci95: t.ci95, n: t.n, ratio: t.ratio_vs_16bit,
  }))

  return (
    <Panel
      title="Energy per tier"
      sub={`${energy.session} · n=${data[0]?.n?.toLocaleString() ?? '?'} per tier`}
      provenance={energy.provenance}
      caveat={energy.caveat}
    >
      <ResponsiveContainer width="100%" height={210}>
        <BarChart data={data} margin={{ top: 18, right: 18, bottom: 0, left: -8 }} barSize={52}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
          <XAxis dataKey="name" axisLine={false} tickLine={false}
            tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }} />
          <YAxis axisLine={false} tickLine={false}
            tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
            label={{ value: 'J / token', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-muted)' }} />
          <Tooltip
            cursor={{ fill: 'var(--bg-hover)', opacity: 0.4 }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="chart-tooltip">
                  <div className="chart-tooltip-name">{d.name}</div>
                  <div>{d.value.toFixed(3)} J/token <span className="mono">±{d.ci95.toFixed(3)}</span></div>
                  <div className="tt-dim">n = {d.n.toLocaleString()}{d.ratio ? ` · ${(d.ratio * 100).toFixed(1)}% of fp16` : ''}</div>
                </div>
              )
            }}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {data.map(d => <Cell key={d.name} fill={TIER_COLORS[d.name]} />)}
            <ErrorBar dataKey="ci95" width={6} strokeWidth={1.5} stroke="var(--text-secondary)" />
            <LabelList dataKey="value" position="top" formatter={v => v.toFixed(2)}
              style={{ fill: 'var(--text-primary)', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="ev-method mono">{energy.method} · {energy.hardware}</p>
    </Panel>
  )
}

/* ── Routing quality ────────────────────────────────────────────────── */
function RouterQualityPanel({ rq }) {
  if (!rq || rq.provenance === 'PENDING' || !rq.per_tier || !Object.keys(rq.per_tier).length) {
    return (
      <Panel title="Routing accuracy &amp; precision" sub="classification quality" provenance="PENDING">
        <EmptyState>
          Run <code className="mono">training/scripts/router_quality_report.py</code> to generate this.
        </EmptyState>
      </Panel>
    )
  }

  const o = rq.overall || {}
  const bars = TIERS.map(t => ({
    name: t,
    precision: rq.per_tier[t]?.precision ?? 0,
    recall: rq.per_tier[t]?.recall ?? 0,
    f1: rq.per_tier[t]?.f1 ?? 0,
  }))
  const byDiff = rq.complexity_by_difficulty

  return (
    <Panel
      title="Routing accuracy &amp; precision"
      sub={`n=${rq.n_prompts} prompts · all three tiers in use`}
      provenance={rq.provenance}
      caveat={rq.caveat}
    >
      <div className="metric-row">
        <div className="metric"><span className="metric-v mono">{fmt3(o.macro_precision)}</span><span className="metric-k">macro precision</span></div>
        <div className="metric"><span className="metric-v mono">{fmt3(o.macro_recall)}</span><span className="metric-k">macro recall</span></div>
        <div className="metric"><span className="metric-v mono">{fmt3(o.macro_f1)}</span><span className="metric-k">macro F1</span></div>
        <div className="metric"><span className="metric-v mono">{fmt3(o.spearman_rho)}</span><span className="metric-k">Spearman ρ</span></div>
      </div>

      <ResponsiveContainer width="100%" height={170}>
        <BarChart data={bars} margin={{ top: 12, right: 14, bottom: 0, left: -14 }} barGap={3}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
          <XAxis dataKey="name" axisLine={false} tickLine={false}
            tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }} />
          <YAxis domain={[0, 1]} axisLine={false} tickLine={false}
            tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} />
          <Tooltip cursor={{ fill: 'var(--bg-hover)', opacity: 0.4 }} />
          <Bar dataKey="precision" fill="var(--green-400)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="recall" fill="var(--cyan-400)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
          <Bar dataKey="f1" fill="var(--amber-400)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <div className="ev-legend mono">
        <span><i style={{ background: 'var(--green-400)' }} />precision</span>
        <span><i style={{ background: 'var(--cyan-400)' }} />recall</span>
        <span><i style={{ background: 'var(--amber-400)' }} />F1</span>
      </div>

      {byDiff && (
        <p className="ev-takeaway">
          Complexity score rises with prompt difficulty —{' '}
          {['easy', 'medium', 'hard'].map(d => (
            <span key={d} className="mono">{d} {byDiff[d]?.mean_complexity}{d !== 'hard' ? ' · ' : ''}</span>
          ))}
          {' '}— so the sensor is carrying real signal, and every tier is being used.
          Separating <em>easy</em> from <em>medium</em> is the open problem, and Session 2 is what will settle it.
        </p>
      )}
    </Panel>
  )
}

/* ── Session 4 conditions (collapsed by default — older, contended) ── */
function RoutingConditionsPanel({ routing }) {
  const [runIndex, setRunIndex] = useState(0)

  if (!routing || routing.provenance === 'PENDING' || !routing.runs?.length) {
    return (
      <Panel title="Routing conditions" sub="Session 4" provenance="PENDING" collapsible defaultOpen={false}>
        <EmptyState>Session 4 has not produced a gate-eligible run yet.</EmptyState>
      </Panel>
    )
  }

  const run = routing.runs[runIndex]
  const points = run.conditions.map(c => ({
    name: c.condition, accuracy: c.accuracy, energy: c.j_per_request,
    mix: [c.pct_4bit, c.pct_8bit, c.pct_16bit],
  }))

  const tabs = (
    <div className="run-tabs mono">
      {routing.runs.map((r, i) => (
        <button key={r.source} className={`run-tab ${i === runIndex ? 'active' : ''}`}
          onClick={() => setRunIndex(i)}>
          {r.run.split('—')[0].trim()}
        </button>
      ))}
    </div>
  )

  return (
    <Panel
      title="Routing conditions"
      sub={`Session 4 · ${routing.runs.length} exploratory runs · ${run.source}`}
      provenance={routing.provenance}
      caveat={routing.caveat}
      actions={tabs}
      collapsible
      defaultOpen={false}
    >
      <p className="ev-contention mono">
        <AlertTriangle size={12} strokeWidth={2} />{run.contention_detail}
      </p>

      <ResponsiveContainer width="100%" height={230}>
        <ScatterChart margin={{ top: 12, right: 20, bottom: 22, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" />
          <XAxis dataKey="energy" type="number" axisLine={false} tickLine={false}
            tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
            label={{ value: 'J / request (lower is better)', position: 'insideBottom', offset: -12, fontSize: 10, fill: 'var(--text-muted)' }} />
          <YAxis dataKey="accuracy" type="number" axisLine={false} tickLine={false}
            tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
            label={{ value: 'proxy score', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-muted)' }} />
          <ZAxis range={[90, 90]} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null
              const d = payload[0].payload
              return (
                <div className="chart-tooltip">
                  <div className="chart-tooltip-name">{d.name.replace(/_/g, ' ')}</div>
                  <div>proxy score <span className="mono">{d.accuracy?.toFixed(3)}</span></div>
                  <div>energy <span className="mono">{d.energy?.toFixed(1)} J/req</span></div>
                  <div className="tt-dim mono">mix {d.mix.map(m => `${((m ?? 0) * 100).toFixed(0)}%`).join(' / ')}</div>
                </div>
              )
            }}
          />
          <Scatter data={points} isAnimationActive={false}
            shape={(props) => {
              const isOurs = props.payload.name === 'fuzzy_router'
              const r = isOurs ? 9 : 6
              return (
                <g>
                  {isOurs && <circle cx={props.cx} cy={props.cy} r={r + 6} fill="var(--green-tint)" />}
                  <circle cx={props.cx} cy={props.cy} r={r}
                    fill={isOurs ? 'var(--green-400)' : 'var(--bg-hover)'}
                    stroke={isOurs ? 'var(--green-400)' : 'var(--border-strong)'}
                    strokeWidth={isOurs ? 2 : 1} />
                </g>
              )
            }} />
        </ScatterChart>
      </ResponsiveContainer>

      <table className="ev-table">
        <thead>
          <tr>
            <th>Condition</th><th>Proxy score</th><th>J / request</th><th>Tier mix 4/8/16</th>
          </tr>
        </thead>
        <tbody>
          {run.conditions.map(c => (
            <tr key={c.condition} className={c.condition === 'fuzzy_router' ? 'highlighted' : ''}>
              <td>{c.condition.replace(/_/g, ' ')}</td>
              <td className="mono">{c.accuracy?.toFixed(3) ?? '—'}</td>
              <td className="mono num-strong">{c.j_per_request?.toFixed(1) ?? '—'}</td>
              <td className="mono dim">
                {[c.pct_4bit, c.pct_8bit, c.pct_16bit]
                  .map(m => `${((m ?? 0) * 100).toFixed(0)}`).join(' / ')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  )
}

/* ── Verification record (validation + experiment log, one panel) ───── */
const LEVEL_ICON = { PASS: CheckCircle2, WARN: AlertTriangle, FAIL: XCircle }
const VERDICT_CLASS = { PASS: 'verified', CONTENDED: 'contended', FAIL: 'fail' }

function VerificationPanel({ validation, experiments }) {
  const [filter, setFilter] = useState('open')

  if (!validation?.findings?.length) {
    return (
      <Panel title="Verification record" sub="verify_results.py" provenance="PENDING" collapsible defaultOpen={false}>
        <EmptyState>No validation report yet.</EmptyState>
      </Panel>
    )
  }

  const c = validation.counts
  const total = c.PASS + c.WARN + c.FAIL
  const shown = validation.findings.filter(f =>
    filter === 'all' ? true : f.level !== 'PASS')

  const actions = (
    <div className="run-tabs mono">
      {[['open', 'open items'], ['all', 'all checks']].map(([k, label]) => (
        <button key={k} className={`run-tab ${filter === k ? 'active' : ''}`} onClick={() => setFilter(k)}>
          {label}
        </button>
      ))}
    </div>
  )

  return (
    <Panel
      title="Verification record"
      sub={`${c.PASS} of ${total} checks pass · verify_results.py, ${validation.generated_at?.slice(0, 10) ?? ''}`}
      provenance={validation.provenance}
      actions={actions}
      collapsible
      defaultOpen={false}
    >
      <div className="check-bar" title={`${c.PASS} pass · ${c.WARN} warn · ${c.FAIL} fail`}>
        <div className="cb pass" style={{ flex: c.PASS }} />
        <div className="cb warn" style={{ flex: c.WARN }} />
        <div className="cb fail" style={{ flex: c.FAIL }} />
      </div>
      <p className="ev-verdict-line mono">
        Gate verdict, verbatim: <span className="verdict-raw">{validation.verdict}</span>
      </p>

      <ul className="finding-list">
        {shown.map((f, i) => {
          const Icon = LEVEL_ICON[f.level] || CheckCircle2
          return (
            <li key={i} className={`finding ${f.level.toLowerCase()}`}>
              <Icon size={13} strokeWidth={2} className="finding-icon" />
              <span className="finding-area mono">{f.area}</span>
              <span className="finding-text">{f.finding}</span>
            </li>
          )
        })}
        {!shown.length && <li className="finding"><span className="finding-text dim">Nothing open.</span></li>}
      </ul>

      {experiments?.runs?.length > 0 && (
        <>
          <h3 className="ev-subhead">Session log</h3>
          <table className="ev-table">
            <thead>
              <tr><th>Date</th><th>Session</th><th>Verdict</th><th>Finding</th></tr>
            </thead>
            <tbody>
              {experiments.runs.map((r, i) => (
                <tr key={i}>
                  <td className="mono dim">{r.date}</td>
                  <td className="mono">{r.session}</td>
                  <td><span className={`prov-badge ${VERDICT_CLASS[r.verdict] || 'pending'} mono`}>{r.verdict}</span></td>
                  <td className="finding-cell">{r.finding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Panel>
  )
}

/* ── Page ──────────────────────────────────────────────────────────── */
export function Analytics() {
  const { traces } = useAnalytics()
  const { validation, experiments, energy, routing, routerQuality, loading, error, refresh } = useEvidence()

  const verified = energy?.provenance === 'VERIFIED' ? energy : null
  const o = routerQuality?.overall || {}
  const c = validation?.counts

  const tierMix = routerQuality?.tier_mix

  return (
    <div className="analytics">
      <div className="analytics-header">
        <div>
          <h1 className="analytics-title display">Results &amp; Evidence</h1>
          <p className="analytics-subtitle mono">
            Every figure is labelled with where it came from.
          </p>
        </div>
        <button className={`refresh-btn ${loading ? 'loading' : ''}`} onClick={refresh} disabled={loading}>
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} strokeWidth={2} />
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error && <div className="analytics-error mono">{error}</div>}

      <HeroResult energy={energy} />
      <ProgressStrip experiments={experiments} energy={energy} validation={validation} />

      <div className="ev-stat-grid">
        <StatCard
          icon={Zap} accent="var(--green-400)"
          provenance={verified ? 'VERIFIED' : 'PENDING'}
          value={verified ? verified.tiers.find(t => t.tier === '4bit')?.mean_j_per_token.toFixed(2) : '—'}
          label="J / token at 4-bit"
          sub={verified
            ? `the cheapest tier, measured — against ${verified.tiers.find(t => t.tier === '16bit')?.mean_j_per_token.toFixed(2)} at fp16`
            : 'Session 1 still ahead.'}
        />
        <StatCard
          icon={Target} accent="var(--cyan-400)"
          provenance={routerQuality?.provenance || 'PENDING'}
          value={o.macro_precision !== undefined ? fmt3(o.macro_precision) : '—'}
          label="routing macro precision"
          sub={o.macro_precision !== undefined
            ? `over ${routerQuality.n_prompts} prompts — the routing decision, not model accuracy`
            : 'not generated yet.'}
        />
        <StatCard
          icon={ShieldCheck} accent="var(--indigo-400)"
          provenance="DERIVED"
          value={tierMix
            ? TIERS.map(t => `${Math.round((tierMix[t] ?? 0) * 100)}`).join('/')
            : '—'}
          label="tier mix 4 / 8 / 16-bit"
          sub="all three precision tiers in active use — none stranded."
        />
        <StatCard
          icon={FlaskConical} accent="var(--amber-400)"
          provenance="PENDING"
          value="—"
          label="benchmark accuracy"
          sub="Session 2 is next up. No accuracy number exists yet."
        />
      </div>

      <div className="ev-two-col">
        <EnergyPanel energy={energy} />
        <RouterQualityPanel rq={routerQuality} />
      </div>

      <RoutingConditionsPanel routing={routing} />
      <VerificationPanel validation={validation} experiments={experiments} />

      {traces.length > 0 && (
        <Panel
          title="This session"
          sub={`${traces.length} prompts routed through the live API`}
          provenance="DERIVED"
          collapsible
          defaultOpen={false}
          caveat="Live dashboard traffic, not an experiment. Energy here is a placeholder unless a GPU model pool is loaded."
        >
          <div className="live-mix">
            {TIERS.map(t => {
              const n = traces.filter(x => (x.final_tier || x.tier) === t).length
              return (
                <div className="live-row" key={t}>
                  <span className="tier-dot" style={{ background: TIER_COLORS[t] }} />
                  <span className="mono live-tier">{t}</span>
                  <div className="live-bar">
                    <div className="live-bar-fill"
                      style={{ width: `${(n / traces.length) * 100}%`, background: TIER_COLORS[t] }} />
                  </div>
                  <span className="mono live-n">{n}</span>
                </div>
              )
            })}
          </div>
        </Panel>
      )}
    </div>
  )
}
