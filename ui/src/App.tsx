import { Routes, Route } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { HomePage } from './pages/HomePage'
import { KnowledgeBasePage } from './pages/KnowledgeBasePage'
import { ChatPage } from './pages/ChatPage'
import { CodeGenerationPage } from './pages/CodeGenerationPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/knowledge-base" element={<KnowledgeBasePage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/code-generation" element={<CodeGenerationPage />} />
      </Routes>
    </Layout>
  )
}

export default App