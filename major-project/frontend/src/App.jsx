import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Sidebar } from './components/Sidebar'
import { Overview } from './pages/Overview'
import { Playground } from './pages/Playground'
import { Analytics }  from './pages/Analytics'
import { Settings }   from './pages/Settings'
import { useHealth }  from './hooks/useBackend'
import './App.css'

export default function App() {
  const [view, setView] = useState('overview')
  const { online, gpuReady } = useHealth()

  const pages = { overview: Overview, playground: Playground, analytics: Analytics, settings: Settings }
  const Page = pages[view]

  return (
    <div className="app">
      <Sidebar view={view} setView={setView} online={online} gpuReady={gpuReady} />
      <main className="main-content">
        {/* A short cross-fade only — no sliding, so switching pages reads as
            a page change, not an effect. */}
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            style={{ minHeight: '100%' }}
          >
            <Page online={online} gpuReady={gpuReady} />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
