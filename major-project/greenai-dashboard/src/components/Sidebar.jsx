import { motion } from 'framer-motion'
import { Cpu, FlaskConical, BarChart3, SlidersHorizontal, Activity, Zap } from 'lucide-react'
import './Sidebar.css'

const NAV = [
  { id: 'playground', label: 'Playground',  Icon: FlaskConical,     desc: 'Run inference' },
  { id: 'analytics',  label: 'Analytics',   Icon: BarChart3,        desc: 'Metrics & results' },
  { id: 'settings',   label: 'Config',      Icon: SlidersHorizontal, desc: 'Tune thresholds' },
]

export function Sidebar({ view, setView, online, gpuReady }) {
  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Zap size={18} strokeWidth={2.5} />
        </div>
        <div>
          <div className="sidebar-logo-name">Green‑Weight</div>
          <div className="sidebar-logo-sub">v0.1.0 · SPIT TE(CE)</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {NAV.map(({ id, label, Icon, desc }) => {
          const active = view === id
          return (
            <button
              key={id}
              className={`sidebar-nav-item ${active ? 'active' : ''}`}
              onClick={() => setView(id)}
            >
              {active && (
                <motion.div
                  layoutId="nav-pill"
                  className="sidebar-nav-pill"
                  transition={{ type: 'spring', stiffness: 400, damping: 35 }}
                />
              )}
              <span className="sidebar-nav-icon">
                <Icon size={16} strokeWidth={active ? 2.2 : 1.8} />
              </span>
              <span className="sidebar-nav-text">
                <span className="sidebar-nav-label">{label}</span>
                <span className="sidebar-nav-desc">{desc}</span>
              </span>
            </button>
          )
        })}
      </nav>

      {/* Status */}
      <div className="sidebar-status">
        <div className={`status-row ${online ? 'ok' : 'err'}`}>
          <span className={`status-dot ${online ? 'ok' : 'err'}`} />
          <span className="status-label">{online ? 'API connected' : 'API offline'}</span>
        </div>
        <div className={`status-row ${gpuReady ? 'ok' : 'warn'}`}>
          <Cpu size={11} strokeWidth={1.5} />
          <span className="status-label">{gpuReady ? 'GPU ready' : 'Routing-only mode'}</span>
        </div>
        <div className="status-endpoint">localhost:8000</div>

        <div className="tier-pills">
          <span className="tier-pill green">4-bit</span>
          <span className="tier-pill indigo">8-bit</span>
          <span className="tier-pill amber">16-bit</span>
        </div>
      </div>
    </aside>
  )
}
