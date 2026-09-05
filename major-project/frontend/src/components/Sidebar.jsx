import { useState } from 'react'
import { Cpu, LayoutDashboard, FlaskConical, BarChart3, SlidersHorizontal, Leaf, Sun, Moon, ChevronDown } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'
import './Sidebar.css'

const NAV = [
  { id: 'overview',   label: 'Overview',   Icon: LayoutDashboard,   desc: 'What this does' },
  { id: 'playground', label: 'Playground', Icon: FlaskConical,      desc: 'Ask a question' },
  { id: 'analytics',  label: 'Analytics',  Icon: BarChart3,         desc: 'Metrics & results' },
  { id: 'settings',   label: 'Config',     Icon: SlidersHorizontal, desc: 'Tune thresholds' },
]

export function Sidebar({ view, setView, online, gpuReady }) {
  const { theme, toggle } = useTheme()
  const [statusExpanded, setStatusExpanded] = useState(false)

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
        {!statusExpanded && (
          <button
            className="status-expand-toggle"
            onClick={() => setStatusExpanded(true)}
            aria-label="Expand settings"
          >
            <ChevronDown size={16} strokeWidth={1.8} />
          </button>
        )}

        {statusExpanded && (
          <>
            <button
              className="theme-toggle"
              onClick={toggle}
              aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark'
                ? <><Sun size={13} strokeWidth={1.8} /> Light mode</>
                : <><Moon size={13} strokeWidth={1.8} /> Dark mode</>}
            </button>

            <button
              className="status-collapse-toggle"
              onClick={() => setStatusExpanded(false)}
              aria-label="Collapse settings"
            >
              <ChevronDown size={16} strokeWidth={1.8} style={{ transform: 'rotate(180deg)' }} />
            </button>
          </>
        )}
      </div>
    </aside>
  )
}
