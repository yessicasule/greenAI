import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Sidebar } from './components/Sidebar'
import { Playground } from './pages/Playground'
import { Analytics }  from './pages/Analytics'
import { Settings }   from './pages/Settings'
import { useHealth }  from './hooks/useBackend'
import './App.css'

export default function App() {
  const [view, setView] = useState('playground')
  const { online, gpuReady } = useHealth()

  const pages = { playground: Playground, analytics: Analytics, settings: Settings }
  const Page = pages[view]

  return (
    <div className="app">
      <Sidebar view={view} setView={setView} online={online} gpuReady={gpuReady} />
      <main className="main-content">
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.2 }}
            style={{ minHeight: '100%' }}
          >
            <Page online={online} gpuReady={gpuReady} />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  )
}
