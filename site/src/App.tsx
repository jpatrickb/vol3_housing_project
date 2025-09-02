import { Routes, Route, Navigate, NavLink } from 'react-router-dom'
import './App.css'
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
          <a className="brand brand-title" href="https://github.com/jpatrickb/vol3_housing_project" target="_blank" rel="noreferrer">The Cost of Living</a>
          <nav className="nav-links">
            <NavLink to="/" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Home</NavLink>
            <NavLink to="/methods" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Methods</NavLink>
            <NavLink to="/results" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Results</NavLink>
            <NavLink to="/notebooks" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Notebooks</NavLink>
            <NavLink to="/data" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Data</NavLink>
            <NavLink to="/contributors" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Contributors</NavLink>
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
      <footer className="footer container">© {new Date().getFullYear()} Housing Markets Project</footer>
    </div>
  )
}

export default App
