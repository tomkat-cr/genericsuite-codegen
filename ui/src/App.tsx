import { Route, Routes } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { ChatPage } from './pages/ChatPage'
import { CodeGenerationPage } from './pages/CodeGenerationPage'
import { ConfigValidatorPage } from './pages/ConfigValidatorPage'
import { HomePage } from './pages/HomePage'
import { KnowledgeBasePage } from './pages/KnowledgeBasePage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/knowledge-base" element={<KnowledgeBasePage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/code-generation" element={<CodeGenerationPage />} />
        <Route path="/config-validator" element={<ConfigValidatorPage />} />
      </Routes>
    </Layout>
  )
}

export default App