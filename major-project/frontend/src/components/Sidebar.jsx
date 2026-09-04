import { Cpu, LayoutDashboard, FlaskConical, BarChart3, SlidersHorizontal, Leaf } from 'lucide-react'
import './Sidebar.css'

const NAV = [
  { id: 'overview',   label: 'Overview',   Icon: LayoutDashboard,   desc: 'What this does' },
  { id: 'playground', label: 'Playground', Icon: FlaskConical,      desc: 'Ask a question' },
  { id: 'analytics',  label: 'Analytics',  Icon: BarChart3,         desc: 'Metrics & results' },
  { id: 'settings',   label: 'Config',     Icon: SlidersHorizontal, desc: 'Tune thresholds' },
]

export function Sidebar({ view, setView, online, gpuReady }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Leaf size={17} strokeWidth={2} />
        </div>
        <div>
          <div className="sidebar-logo-name">Green-Weight</div>
          <div className="sidebar-logo-sub">v0.1.0 · SPIT TE(CE)</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV.map(({ id, label, Icon, desc }) => {
          const active = view === id
          return (
            <button
              key={id}
              className={`sidebar-nav-item ${active ? 'active' : ''}`}
              onClick={() => setView(id)}
              aria-current={active ? 'page' : undefined}
            >
              <span className="sidebar-nav-icon">
                <Icon size={16} strokeWidth={1.9} />
              </span>
              <span className="sidebar-nav-text">
                <span className="sidebar-nav-label">{label}</span>
                <span className="sidebar-nav-desc">{desc}</span>
              </span>
            </button>
          )
        })}
      </nav>

      <div className="sidebar-status">
        <div className={`status-row ${online ? 'ok' : 'err'}`}>
          <span className={`status-dot ${online ? 'ok' : 'err'}`} />
          <span className="status-label">{online ? 'API connected' : 'API offline'}</span>
        </div>
        <div className={`status-row ${gpuReady ? 'ok' : 'warn'}`}>
          <Cpu size={11} strokeWidth={1.6} />
          <span className="status-label">{gpuReady ? 'GPU ready' : 'Routing-only mode'}</span>
        </div>
        <div className="status-endpoint mono">localhost:8000</div>

        <div className="tier-pills">
          <span className="tier-pill green">Light</span>
          <span className="tier-pill indigo">Standard</span>
          <span className="tier-pill amber">Full</span>
        </div>
      </div>
    </aside>
  )
}
