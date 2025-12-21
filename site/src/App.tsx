import { Routes, Route, Navigate, NavLink } from 'react-router-dom'
import Home from './pages/Home'
import Methods from './pages/Methods.mdx'
import Results from './pages/Results'
import Notebooks from './pages/Notebooks'
import Data from './pages/Data.mdx'
import Contributors from './pages/Contributors'

function App() {
  return (
    <div className="app-root">
      <header className="navbar">
        <div className="navbar-inner">
          <a className="brand" href="https://github.com/jpatrickb/vol3_housing_project" target="_blank" rel="noreferrer">
            <svg viewBox="0 0 16 16" aria-hidden="true" fill="currentColor" style={{ width: 24, height: 24 }}>
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
            </svg>
            <span>Housing Markets</span>
          </a>
          <nav className="nav-links">
            <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Home</NavLink>
            <NavLink to="/methods" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Methods</NavLink>
            <NavLink to="/results" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Results</NavLink>
            <NavLink to="/notebooks" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Notebooks</NavLink>
            <NavLink to="/data" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Data</NavLink>
            <NavLink to="/contributors" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>Contributors</NavLink>
          </nav>
        </div>
      </header>
      <main className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/methods" element={<Methods />} />
          <Route path="/results" element={<Results />} />
          <Route path="/notebooks" element={<Notebooks />} />
          <Route path="/data" element={<Data />} />
          <Route path="/contributors" element={<Contributors />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer>
        <div className="container">
          <p>© {new Date().getFullYear()} Housing Markets Project</p>
          <p className="footer-note">
            <a href="https://github.com/jpatrickb/vol3_housing_project" target="_blank" rel="noreferrer">View on GitHub</a>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
