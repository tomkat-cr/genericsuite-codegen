import { useState, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { RefreshCw, Upload, Database, FileText, AlertCircle, CheckCircle, Info } from 'lucide-react'

import { baseUrl } from '@/lib/api'

interface KnowledgeBaseStats {
  documentCount: number
  lastUpdated: string
  repositoryUrl: string
  status: 'healthy' | 'updating' | 'error'
}

interface UpdateProgress {
  stage: string
  progress: number
  message: string
}

export function KnowledgeBasePage() {
  const [stats, setStats] = useState<KnowledgeBaseStats>({
    documentCount: 0,
    lastUpdated: 'Never',
    repositoryUrl: 'https://github.com/tomkat-cr/genericsuite-basecamp.git',
    status: 'healthy'
  })
  
  const [isUpdating, setIsUpdating] = useState(false)
  const [updateProgress, setUpdateProgress] = useState<UpdateProgress | null>(null)
  const [uploadFiles, setUploadFiles] = useState<FileList | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [alerts, setAlerts] = useState<Array<{id: string, type: 'success' | 'error' | 'info', message: string}>>([])

  const addAlert = (type: 'success' | 'error' | 'info', message: string) => {
    const id = Date.now().toString()
    setAlerts(prev => [...prev, { id, type, message }])
    setTimeout(() => {
      setAlerts(prev => prev.filter(alert => alert.id !== id))
    }, 5000)
  }

  const handleUpdateKnowledgeBase = async () => {
    setIsUpdating(true)
    setStats(prev => ({ ...prev, status: 'updating' }))
    
    try {
      // Simulate progress updates
      const stages = [
        { stage: 'Cloning repository', progress: 20, message: 'Downloading latest documentation...' },
        { stage: 'Processing documents', progress: 50, message: 'Extracting text from files...' },
        { stage: 'Generating embeddings', progress: 80, message: 'Creating vector embeddings...' },
        { stage: 'Updating database', progress: 100, message: 'Storing in knowledge base...' }
      ]

      for (const stage of stages) {
        setUpdateProgress(stage)
        await new Promise(resolve => setTimeout(resolve, 1500))
      }

      const forceRefresh = false

      // Make actual API call
      const response = await fetch(`${baseUrl}/api/update-knowledge-base`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          force_refresh: forceRefresh
        })
      })

      if (response.ok) {
        const result = await response.json()
        const resultData = result.result
        if (!resultData.statistics.success) {
          setStats(prev => ({ ...prev, status: 'error' }))
          addAlert('error', resultData.statistics.error ?? 'Failed to update knowledge base')
          return
        }
        setStats(prev => ({
          ...prev,
          // documentCount: result.documentCount || prev.documentCount + 150,
          documentCount: resultData.statistics.total_documents ?? prev.documentCount,
          lastUpdated: new Date().toLocaleString(),
          status: 'healthy'
        }))
        addAlert('success', 'Knowledge base updated successfully!')
      } else {
        throw new Error('Failed to update knowledge base')
      }
    } catch (error) {
      setStats(prev => ({ ...prev, status: 'error' }))
      addAlert('error', 'Failed to update knowledge base. Please try again.')
    } finally {
      setIsUpdating(false)
      setUpdateProgress(null)
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

      const response = await fetch(`${baseUrl}/api/upload-document`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const result = await response.json()
        setStats(prev => ({
          ...prev,
          documentCount: prev.documentCount + result.addedDocuments
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
                <span>{updateProgress.progress}%</span>
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