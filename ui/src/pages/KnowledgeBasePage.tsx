import { useEffect, useState } from 'react'

import { KnowledgeBaseAlerts, alertTimeout } from '@/components/KnowledgeBase/Alerts'
import { DocumentUpload } from '@/components/KnowledgeBase/DocumentUpload'
import { IngestionStatus, KnowledgeBaseStats, UpdateProgress } from '@/components/KnowledgeBase/interfaces'
import { KnowledgeBaseSearch } from '@/components/KnowledgeBase/KnowledgeBaseSearch'
import { KnowledgeBaseStatus } from '@/components/KnowledgeBase/KnowledgeBaseStatus'
import { AlertElement } from '@/components/KnowledgeBase/types'

import { baseUrl, debug } from '@/lib/api'

const remoteRepoUrl: string = process.env.VITE_REMOTE_REPO_URL || ''
const remoteRepoBranch: string = process.env.VITE_REMOTE_REPO_BRANCH || ''

export function KnowledgeBasePage() {
  const [stats, setStats] = useState<KnowledgeBaseStats>({
    documentCount: 0,
    lastUpdated: 'Never',
    repositoryUrl: remoteRepoUrl,
    repositoryBranch: remoteRepoBranch,
    status: 'healthy'
  })

  const [isUpdating, setIsUpdating] = useState(false)
  const [updateProgress, setUpdateProgress] = useState<UpdateProgress | null>(null)
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [alerts, setAlerts] = useState<Array<AlertElement>>([])

  const addAlert = (type: 'success' | 'error' | 'info', message: string) => {
    const id = Date.now().toString()
    if (debug) console.log('addAlert | id', id)
    if (alerts.some(alert => alert.message === message)) return
    setAlerts(prev => [...prev, { id, type, message }])
    setTimeout(() => {
      setAlerts(prev => prev.filter(alert => alert.id !== id))
    }, alertTimeout)
  }

  const getProgress = async () => {
    const response = await fetch(`${baseUrl}/update-knowledge-base/progress`)
    const result = await response.json()
    const resultData = result.data
    if (debug) console.log('getProgress | result', resultData)
    setUpdateProgress({
      stage: IngestionStatus[resultData.status as keyof typeof IngestionStatus],
      progress: resultData.total_steps > 0 ? resultData.completed_steps / resultData.total_steps * 100 : 0,
      message: resultData.current_step
    })
    setStats(prev => ({
      ...prev,
      documentCount: resultData.total_files ?? prev.documentCount,
      lastUpdated: new Date().toLocaleString(),
      status: resultData.status === 'completed' ? 'healthy' : resultData.status === 'failed' ? 'error' : ['not_started', 'scheduled'].indexOf(resultData.status) !== -1 ? 'scheduled' : 'updating'
    }))
    if (stats.status === 'updating' && !isUpdating) {
      setIsUpdating(true)
    }
    if (resultData.status === 'completed' && isUpdating) {
      setIsUpdating(false)
      setUpdateProgress(null)
      addAlert('success', 'Knowledge base updated successfully!')
    }
    if (resultData.status === 'failed' && isUpdating) {
      setIsUpdating(false)
      setUpdateProgress(null)
      addAlert('error', 'Failed to update knowledge base. Please try again. Error: ' + resultData.error_message)
    }
  }

  useEffect(() => {
    if (debug) console.log('Component mounted, calling getProgress')
    getProgress()
  }, [])

  // If it's updating, refresh every 5 seconds
  useEffect(() => {
    if (debug) console.log('isUpdating changed [2], calling getProgress')
    if (isUpdating) {
      const interval = setInterval(() => {
        getProgress()
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [isUpdating])

  const handleUpdateKnowledgeBase = async () => {
    setIsUpdating(true)
    setStats(prev => ({ ...prev, status: 'updating' }))
    try {
      const forceRefresh = false
      const response = await fetch(`${baseUrl}/update-knowledge-base`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          force_refresh: forceRefresh
        })
      })

      if (response.ok && response.status === 200) {
        const result = await response.json()
        const resultData = result.data
        if (debug) console.log('update-knowledge-base | resultData', resultData)
        setStats(prev => ({
          ...prev,
          documentCount: resultData.statistics.total_documents ?? prev.documentCount,
          lastUpdated: new Date().toLocaleString(),
          status: 'healthy'
        }))
        addAlert('success', resultData.status ?? 'Knowledge base updated successfully! [2]')
      } else {
        // throw new Error('Failed to update knowledge base')
        setStats(prev => ({ ...prev, status: 'error' }))
        addAlert('error', (response.statusText ?? 'Failed to update knowledge base') + ' [' + response.status + ']')
        setIsUpdating(false)
        return
      }
    } catch (error) {
      setStats(prev => ({ ...prev, status: 'error' }))
      addAlert('error', 'Failed to update knowledge base. Please try again.')
      setIsUpdating(false)
    }
  }

  const handleFileUpload = async () => {
    if (!uploadFiles || uploadFiles.length === 0) return

    setIsUploading(true)
    try {
      const formData = new FormData()
      Array.from(uploadFiles).forEach(file => {
        formData.append('files', file)
      })

      const response = await fetch(`${baseUrl}/upload-document`, {
        method: 'POST',
        body: formData
      })

      if (response.ok && response.status === 200) {
        const result = await response.json()
        setStats(prev => ({
          ...prev,
          documentCount: prev.documentCount + result.statistics.total_documents
        }))
        addAlert('success', `Successfully uploaded ${uploadFiles.length} document(s)`)
        setUploadFiles(null)
        // Reset file input
        const fileInput = document.getElementById('file-upload') as HTMLInputElement
        if (fileInput) fileInput.value = ''
      } else {
        throw new Error('Upload failed')
      }
    } catch (error) {
      addAlert('error', 'Failed to upload documents. Please try again.')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Knowledge Base Management</h1>
        <p className="text-muted-foreground">
          Manage and update the GenericSuite documentation repository
        </p>
      </div>

      {/* Alerts */}
      <KnowledgeBaseAlerts
        alerts={alerts}
      />

      {/* Knowledge Base Status */}
      <KnowledgeBaseStatus
        handleUpdateKnowledgeBase={handleUpdateKnowledgeBase}
        stats={stats}
        isUpdating={isUpdating}
        updateProgress={updateProgress}
      />

      {/* Document Upload */}
      <DocumentUpload
        handleFileUpload={handleFileUpload}
        setUploadFiles={setUploadFiles}
        uploadFiles={uploadFiles}
        isUploading={isUploading}
      />

      {/* Knowledge Base Search */}
      <KnowledgeBaseSearch />
    </div>
  )
}