import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { AlertCircle, CheckCircle, Database, FileText, Info, RefreshCw, Upload } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'

import { baseUrl, debug } from '@/lib/api'

const remoteRepoUrl: string = process.env.VITE_REMOTE_REPO_URL || ''
const remoteRepoBranch: string = process.env.VITE_REMOTE_REPO_BRANCH || ''
const alertTimeout: number = 10000

interface KnowledgeBaseStats {
  documentCount: number
  lastUpdated: string
  repositoryUrl: string
  repositoryBranch: string
  status: 'healthy' | 'updating' | 'error' | 'scheduled'
}

interface UpdateProgress {
  stage: string
  progress: number
  message: string
}

const IngestionStatus = {
    scheduled: "Scheduled",
    not_started: "Not started",
    cloning: "Cloning",
    processing_files: "Processing files",
    chunking: "Chunking",
    generating_embeddings: "Generating embeddings",
    storing_vectors: "Storing vectors",
    completed: "Completed",
    failed: "Failed",
}

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
  const [alerts, setAlerts] = useState<Array<{id: string, type: 'success' | 'error' | 'info', message: string}>>([])

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
    // if (!isUpdating) return
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

  // useEffect(() => {
  //   if (debug) console.log('isUpdating changed [1], calling getProgress')
  //   getProgress()
  // }, [isUpdating])

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

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const files = e.dataTransfer.files
    if (files.length > 0) {
      setUploadFiles(files)
    }
  }, [])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'updating': return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />
      case 'error': return <AlertCircle className="h-4 w-4 text-red-500" />
      default: return <Info className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      healthy: 'default',
      updating: 'secondary',
      error: 'destructive'
    } as const
    
    return (
      <Badge variant={variants[status as keyof typeof variants] || 'secondary'}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    )
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
      {alerts.map(alert => (
        <Alert key={alert.id} className={alert.type === 'error' ? 'border-red-200 bg-red-50' : alert.type === 'success' ? 'border-green-200 bg-green-50' : ''}>
          <AlertDescription className={alert.type === 'error' ? 'text-red-800' : alert.type === 'success' ? 'text-green-800' : ''}>
            {alert.message}
          </AlertDescription>
        </Alert>
      ))}

      {/* Knowledge Base Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Knowledge Base Status
          </CardTitle>
          <CardDescription>
            Current status and statistics of the knowledge base
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium">Status</span>
                {getStatusIcon(stats.status)}
              </div>
              {getStatusBadge(stats.status)}
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Documents</span>
              <div className="text-2xl font-bold">{stats.documentCount.toLocaleString()}</div>
            </div>
            <div className="space-y-2">
              <span className="text-sm font-medium">Last Updated</span>
              <div className="text-sm text-muted-foreground">{stats.lastUpdated}</div>
            </div>
          </div>
          <div className="space-y-2">
            <span className="text-sm font-medium">Repository URL</span>
            <div className="text-sm text-muted-foreground font-mono bg-muted p-2 rounded">
              {stats.repositoryUrl}
            </div>
          </div>
          <div className="space-y-2">
            <span className="text-sm font-medium">Repository Branch</span>
            <div className="text-sm text-muted-foreground font-mono bg-muted p-2 rounded">
              {stats.repositoryBranch}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Update Progress */}
      {isUpdating && updateProgress && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <RefreshCw className="h-5 w-5 animate-spin" />
              Updating Knowledge Base
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{updateProgress.stage}</span>
                <span>{updateProgress.progress.toFixed(2)}%</span>
              </div>
              <Progress value={updateProgress.progress} />
              <p className="text-sm text-muted-foreground">{updateProgress.message}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Repository Update */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <RefreshCw className="h-5 w-5" />
            Update Repository
          </CardTitle>
          <CardDescription>
            Refresh the knowledge base with the latest GenericSuite documentation
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button 
            onClick={handleUpdateKnowledgeBase} 
            disabled={isUpdating}
            className="w-full md:w-auto"
          >
            {isUpdating ? (
              <>
                <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                Updating...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                Update Knowledge Base
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Document Upload */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Additional Documents
          </CardTitle>
          <CardDescription>
            Add custom documents to enhance the knowledge base context
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center hover:border-muted-foreground/50 transition-colors"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <FileText className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
            <div className="space-y-2">
              <p className="text-sm font-medium">
                Drag and drop files here, or click to select
              </p>
              <p className="text-xs text-muted-foreground">
                Supports: PDF, TXT, MD, and code files (.py, .js, .ts, .jsx, .tsx, .json)
              </p>
            </div>
            <Input
              id="file-upload"
              type="file"
              multiple
              accept=".pdf,.txt,.md,.py,.js,.ts,.jsx,.tsx,.json"
              className="mt-4"
              onChange={(e) => setUploadFiles(e.target.files)}
            />
          </div>
          
          {uploadFiles && uploadFiles.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium">Selected files:</p>
              <div className="space-y-1">
                {Array.from(uploadFiles).map((file, index) => (
                  <div key={index} className="text-sm text-muted-foreground flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  </div>
                ))}
              </div>
              <Button 
                onClick={handleFileUpload} 
                disabled={isUploading}
                className="w-full md:w-auto"
              >
                {isUploading ? (
                  <>
                    <Upload className="mr-2 h-4 w-4 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Upload {uploadFiles.length} file(s)
                  </>
                )}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}