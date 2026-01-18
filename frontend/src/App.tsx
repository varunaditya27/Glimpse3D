import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Landing } from './pages/Landing';
import { Workspace } from './pages/Workspace';
import { CompareView } from './pages/CompareView';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/workspace" element={<Workspace />} />
        <Route path="/compare" element={<CompareView />} />
      </Routes>
    </Router>
  )
}

export default App
