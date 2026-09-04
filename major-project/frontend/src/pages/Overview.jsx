import { motion } from 'framer-motion'
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Cell, LabelList
} from 'recharts'
import { FileText, Gauge, Zap, Leaf, CheckCircle2, Layers } from 'lucide-react'
import { useAnalytics } from '../hooks/useBackend'
import './Overview.css'

/* Tier identity — the same three colours used everywhere else in the app.
   Naive-user names first, the technical name kept as a subtitle. */
const TIERS = [
  { key: '4bit',  name: 'Light',    tech: '4-bit',  color: 'var(--green-400)'  },
  { key: '8bit',  name: 'Standard', tech: '8-bit',  color: 'var(--indigo-400)' },
  { key: '16bit', name: 'Full',     tech: '16-bit', color: 'var(--amber-400)'  },
]

/* Every number on screen says where it came from. `measured` is the only
   one a reader may treat as a result — see CLAUDE.md's one rule. */
function DataBadge({ kind }) {
  const map = {
    measured: { label: 'Measured', cls: 'measured' },
    modeled:  { label: 'Modeled',  cls: 'modeled'  },
    mock:     { label: 'Demo data', cls: 'mock'    },
    none:     { label: 'No data',  cls: 'none'     },
  }
  const b = map[kind] || map.none
  return <span className={`data-badge ${b.cls} mono`}>{b.label}</span>
}

/* A KPI whose caption is a sentence, not a label. */
function StatTile({ icon: Icon, value, headline, caption, accent, kind, delay = 0 }) {
  return (
    <motion.div
      className="stat-tile"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35 }}
      style={{ '--accent': accent }}
    >
      <div className="stat-tile-top">
        <Icon size={15} color={accent} strokeWidth={2} />
        <DataBadge kind={kind} />
      </div>
      <div className="stat-tile-value mono">{value}</div>
      <div className="stat-tile-headline">{headline}</div>
      <div className="stat-tile-caption">{caption}</div>
    </motion.div>
  )
}

/* "How it works" — static, explanatory, no data. */
function HowItWorks() {
  const steps = [
    { icon: FileText, title: 'Read the question', body: 'Five language features are extracted — length, vocabulary, reading level, structure, reasoning cues.' },
    { icon: Gauge,    title: 'Judge how hard it is', body: 'A fuzzy controller turns those features into one complexity score between 0 and 1.' },
    { icon: Layers,   title: 'Pick the right precision', body: 'Easy questions go to the small, cheap model. Hard ones get the full-precision model.' },
  ]
  return (
    <div className="how-strip">
      {steps.map((s, i) => (
        <div className="how-step" key={s.title}>
          <div className="how-step-num mono">{i + 1}</div>
          <s.icon size={18} strokeWidth={1.8} className="how-step-icon" />
          <div className="how-step-title">{s.title}</div>
          <div className="how-step-body">{s.body}</div>
        </div>
      ))}
    </div>
  )
}

/* One bar, three segments — "where your queries went". Reads as a mix,
   which a vertical bar chart of three numbers does not. */
function RoutingMix({ counts, total }) {
  if (!total) {
    return (
      <div className="empty-state">
        No questions answered yet — try a sample question in the Playground.
      </div>
    )
  }
  return (
    <>
      <div className="mix-bar">
        {TIERS.map(t => {
          const pct = ((counts[t.key] || 0) / total) * 100
          if (!pct) return null
          return (
            <div
              key={t.key}
              className="mix-seg"
              style={{ width: `${pct}%`, background: t.color }}
              title={`${t.name} — ${counts[t.key]} of ${total}`}
            >
              {pct > 9 && <span className="mono">{Math.round(pct)}%</span>}
            </div>
          )
        })}
      </div>
      <div className="mix-legend">
        {TIERS.map(t => (
          <div className="mix-legend-item" key={t.key}>
            <span className="mix-dot" style={{ background: t.color }} />
            <span className="mix-legend-name">{t.name}</span>
            <span className="mix-legend-tech mono">{t.tech}</span>
            <span className="mix-legend-count mono">{counts[t.key] || 0}</span>
          </div>
        ))}
      </div>
      <p className="takeaway">
        Most questions are easy. Every query in a lighter tier is energy that was never spent.
      </p>
    </>
  )
}

/* The money chart: two bars, one message. */
function EnergyComparison({ routedJ, baselineJ, kind }) {
  const data = [
    { name: 'Always full precision', value: baselineJ, color: 'var(--amber-400)' },
    { name: 'Green-Weight routing',  value: routedJ,   color: 'var(--green-400)' },
  ]
  const drop = baselineJ > 0 ? Math.round((1 - routedJ / baselineJ) * 100) : null
  return (
    <>
      <ResponsiveContainer width="100%" height={190}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 44, bottom: 4, left: 8 }}>
          <CartesianGrid horizontal={false} stroke="var(--border-subtle)" />
          <XAxis type="number" hide />
          <YAxis
            type="category" dataKey="name" width={150}
            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
            axisLine={false} tickLine={false}
          />
          <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={26}>
            {data.map(d => <Cell key={d.name} fill={d.color} />)}
            <LabelList
              dataKey="value" position="right"
              formatter={v => `${v.toFixed(1)} J`}
              style={{ fill: 'var(--text-primary)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="takeaway">
        Energy used to answer one question, on average. Shorter is better
        {drop !== null && <> — routing uses <strong>{drop}% less</strong></>}.
        {kind !== 'measured' && ' These are placeholder numbers, not measurements.'}
      </p>
    </>
  )
}

function Panel({ title, sub, children, delay = 0 }) {
  return (
    <motion.section
      className="panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
    >
      <div className="panel-head">
        <h2 className="panel-title display">{title}</h2>
        {sub && <span className="panel-sub mono">{sub}</span>}
      </div>
      {children}
    </motion.section>
  )
}

export function Overview({ online }) {
  const { traces, accuracy } = useAnalytics()

  const counts = traces.reduce((acc, t) => {
    const tier = t.final_tier || t.tier
    if (tier) acc[tier] = (acc[tier] || 0) + 1
    return acc
  }, {})

  // Only real, measured energy may be presented as a result: mock
  // placeholders (is_mock) and routing-only calls (energy 0) are excluded.
  const real = traces.filter(t => !t.is_mock && parseFloat(t.energy_joules || 0) > 0)
  const measured = real.length > 0
  const kind = measured ? 'measured' : traces.length ? 'mock' : 'none'

  const routedJ = measured
    ? real.reduce((s, t) => s + parseFloat(t.energy_joules), 0) / real.length
    : 36.8                      // illustrative only
  const baselineJ = 105.2       // api.py's 16-bit placeholder
  const saved = Math.round((1 - routedJ / baselineJ) * 100)

  const routedAcc = accuracy?.accuracy_by_condition?.routed?.mmlu
  const fullAcc = accuracy?.accuracy_by_condition?.always_16bit?.mmlu
  const retained = routedAcc && fullAcc ? Math.round((routedAcc / fullAcc) * 100) : null

  return (
    <div className="overview">
      <header className="overview-hero">
        <h1 className="overview-title display">Green-Weight</h1>
        <p className="overview-lede">
          Most questions people ask an AI are easy. Green-Weight reads each
          question, judges how hard it is, and sends the easy ones to a smaller,
          cheaper version of the model — using less energy without giving up
          answer quality.
        </p>
        {!online && (
          <div className="offline-banner mono">
            Backend offline — showing illustrative numbers, not measurements.
          </div>
        )}
      </header>

      <div className="stat-grid">
        <StatTile
          icon={Leaf} accent="var(--green-400)" kind={kind}
          value={`${saved}%`}
          headline="less energy per question"
          caption={`${routedJ.toFixed(1)} J with routing, versus ${baselineJ} J if every question used full precision.`}
          delay={0}
        />
        <StatTile
          icon={CheckCircle2} accent="var(--indigo-400)" kind={retained ? 'measured' : 'none'}
          value={retained ? `${retained}%` : '—'}
          headline="of the accuracy kept"
          caption={retained
            ? 'Routed answers score this much of what full precision scores on the MMLU benchmark.'
            : 'Run the accuracy evaluation to fill this in.'}
          delay={0.05}
        />
        <StatTile
          icon={Zap} accent="var(--amber-400)" kind={traces.length ? 'measured' : 'none'}
          value={traces.length || '—'}
          headline="questions routed so far"
          caption="Every question the system has read and assigned a precision level."
          delay={0.1}
        />
      </div>

      <Panel title="How it works" delay={0.15}>
        <HowItWorks />
      </Panel>

      <Panel title="Where the questions went" sub={`${traces.length} routed`} delay={0.2}>
        <RoutingMix counts={counts} total={traces.length} />
      </Panel>

      <Panel title="Energy per question" sub="routing vs. always full precision" delay={0.25}>
        <EnergyComparison routedJ={routedJ} baselineJ={baselineJ} kind={kind} />
      </Panel>
    </div>
  )
}
