import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Code, 
  Download, 
  FileText, 
  Settings, 
  Wand2, 
  Copy, 
  Check,
  Package,
  Database,
  Wrench,
  Globe,
  Server,
  Loader2
} from 'lucide-react'

interface GeneratedFile {
  name: string
  content: string
  type: 'json' | 'python' | 'javascript' | 'typescript' | 'jsx' | 'tsx'
  size: number
}

interface GenerationRequest {
  type: 'json-config' | 'langchain-tool' | 'mcp-tool' | 'frontend' | 'backend'
  requirements: string
  framework?: string
  tableName?: string
  toolName?: string
  description?: string
}

export function CodeGenerationPage() {
  const [activeTab, setActiveTab] = useState('json-config')
  const [requirements, setRequirements] = useState('')
  const [tableName, setTableName] = useState('')
  const [toolName, setToolName] = useState('')
  const [description, setDescription] = useState('')
  const [framework, setFramework] = useState('fastapi')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedFiles, setGeneratedFiles] = useState<GeneratedFile[]>([])
  const [selectedFile, setSelectedFile] = useState<GeneratedFile | null>(null)
  const [copiedContent, setCopiedContent] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generateCode = async () => {
    if (!requirements.trim()) {
      setError('Please provide requirements for code generation')
      return
    }

    setIsGenerating(true)
    setError(null)
    setGeneratedFiles([])
    setSelectedFile(null)

    const request: GenerationRequest = {
      type: activeTab as GenerationRequest['type'],
      requirements: requirements.trim(),
      framework: activeTab === 'backend' ? framework : undefined,
      tableName: activeTab === 'json-config' ? tableName : undefined,
      toolName: ['langchain-tool', 'mcp-tool'].includes(activeTab) ? toolName : undefined,
      description: description.trim() || undefined
    }

    try {
      const response = await fetch('/api/generate-code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      })

      if (!response.ok) {
        throw new Error('Failed to generate code')
      }

      const result = await response.json()
      const files: GeneratedFile[] = result.files || []
      
      setGeneratedFiles(files)
      if (files.length > 0) {
        setSelectedFile(files[0])
      }
    } catch (error) {
      setError('Failed to generate code. Please try again.')
      console.error('Code generation error:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const downloadFile = (file: GeneratedFile) => {
    const blob = new Blob([file.content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = file.name
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const downloadAllFiles = () => {
    if (generatedFiles.length === 0) return

    // Create a zip-like structure by downloading individual files
    generatedFiles.forEach(file => {
      setTimeout(() => downloadFile(file), 100)
    })
  }

  const copyToClipboard = async (content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedContent(true)
      setTimeout(() => setCopiedContent(false), 2000)
    } catch (error) {
      console.error('Failed to copy content:', error)
    }
  }

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'json': return <Database className="h-4 w-4" />
      case 'python': return <Code className="h-4 w-4" />
      case 'javascript':
      case 'typescript': return <FileText className="h-4 w-4" />
      case 'jsx':
      case 'tsx': return <Globe className="h-4 w-4" />
      default: return <FileText className="h-4 w-4" />
    }
  }

  const getLanguageForHighlighting = (type: string) => {
    switch (type) {
      case 'json': return 'json'
      case 'python': return 'python'
      case 'javascript': return 'javascript'
      case 'typescript': return 'typescript'
      case 'jsx': return 'jsx'
      case 'tsx': return 'tsx'
      default: return 'text'
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Code Generation</h1>
        <p className="text-muted-foreground">
          Generate JSON configs, Python tools, and application code based on GenericSuite patterns
        </p>
      </div>

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertDescription className="text-red-800">
            {error}
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Generation Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Wand2 className="h-5 w-5" />
              Code Generation
            </CardTitle>
            <CardDescription>
              Specify your requirements and generate code following GenericSuite patterns
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2 lg:grid-cols-3">
                <TabsTrigger value="json-config" className="text-xs">
                  <Database className="h-3 w-3 mr-1" />
                  JSON Config
                </TabsTrigger>
                <TabsTrigger value="langchain-tool" className="text-xs">
                  <Wrench className="h-3 w-3 mr-1" />
                  Langchain
                </TabsTrigger>
                <TabsTrigger value="mcp-tool" className="text-xs">
                  <Settings className="h-3 w-3 mr-1" />
                  MCP Tool
                </TabsTrigger>
                <TabsTrigger value="frontend" className="text-xs">
                  <Globe className="h-3 w-3 mr-1" />
                  Frontend
                </TabsTrigger>
                <TabsTrigger value="backend" className="text-xs">
                  <Server className="h-3 w-3 mr-1" />
                  Backend
                </TabsTrigger>
              </TabsList>

              <TabsContent value="json-config" className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="table-name">Table Name</Label>
                  <Input
                    id="table-name"
                    placeholder="e.g., users, products, orders"
                    value={tableName}
                    onChange={(e) => setTableName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="json-requirements">Table Requirements</Label>
                  <Textarea
                    id="json-requirements"
                    placeholder="Describe the table structure, fields, relationships, and any special requirements..."
                    value={requirements}
                    onChange={(e) => setRequirements(e.target.value)}
                    className="min-h-[120px]"
                  />
                </div>
              </TabsContent>

              <TabsContent value="langchain-tool" className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="langchain-tool-name">Tool Name</Label>
                  <Input
                    id="langchain-tool-name"
                    placeholder="e.g., search_documents, process_data"
                    value={toolName}
                    onChange={(e) => setToolName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="langchain-description">Tool Description</Label>
                  <Input
                    id="langchain-description"
                    placeholder="Brief description of what the tool does"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="langchain-requirements">Tool Requirements</Label>
                  <Textarea
                    id="langchain-requirements"
                    placeholder="Describe the tool functionality, parameters, return values, and implementation details..."
                    value={requirements}
                    onChange={(e) => setRequirements(e.target.value)}
                    className="min-h-[120px]"
                  />
                </div>
              </TabsContent>

              <TabsContent value="mcp-tool" className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="mcp-tool-name">Tool Name</Label>
                  <Input
                    id="mcp-tool-name"
                    placeholder="e.g., file_processor, api_client"
                    value={toolName}
                    onChange={(e) => setToolName(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mcp-description">Tool Description</Label>
                  <Input
                    id="mcp-description"
                    placeholder="Brief description of what the tool does"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="mcp-requirements">Tool Requirements</Label>
                  <Textarea
                    id="mcp-requirements"
                    placeholder="Describe the MCP tool functionality, schema, and implementation details..."
                    value={requirements}
                    onChange={(e) => setRequirements(e.target.value)}
                    className="min-h-[120px]"
                  />
                </div>
              </TabsContent>

              <TabsContent value="frontend" className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="frontend-requirements">Frontend Requirements</Label>
                  <Textarea
                    id="frontend-requirements"
                    placeholder="Describe the React components, pages, functionality, and UI requirements..."
                    value={requirements}
                    onChange={(e) => setRequirements(e.target.value)}
                    className="min-h-[120px]"
                  />
                </div>
              </TabsContent>

              <TabsContent value="backend" className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="backend-framework">Backend Framework</Label>
                  <Select value={framework} onValueChange={setFramework}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select framework" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fastapi">FastAPI (Recommended)</SelectItem>
                      <SelectItem value="flask">Flask</SelectItem>
                      <SelectItem value="chalice">AWS Chalice</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="backend-requirements">Backend Requirements</Label>
                  <Textarea
                    id="backend-requirements"
                    placeholder="Describe the API endpoints, data models, business logic, and functionality..."
                    value={requirements}
                    onChange={(e) => setRequirements(e.target.value)}
                    className="min-h-[120px]"
                  />
                </div>
              </TabsContent>
            </Tabs>

            <Button 
              onClick={generateCode} 
              disabled={isGenerating || !requirements.trim()}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating Code...
                </>
              ) : (
                <>
                  <Wand2 className="mr-2 h-4 w-4" />
                  Generate Code
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Generated Code Preview */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Code className="h-5 w-5" />
                  Generated Code
                </CardTitle>
                <CardDescription>
                  Preview and download generated files
                </CardDescription>
              </div>
              {generatedFiles.length > 0 && (
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={downloadAllFiles}>
                    <Package className="h-4 w-4 mr-1" />
                    Download All
                  </Button>
                  {selectedFile && (
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={() => copyToClipboard(selectedFile.content)}
                    >
                      {copiedContent ? (
                        <Check className="h-4 w-4 mr-1" />
                      ) : (
                        <Copy className="h-4 w-4 mr-1" />
                      )}
                      Copy
                    </Button>
                  )}
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {generatedFiles.length > 0 ? (
              <>
                {/* File List */}
                <div className="space-y-2">
                  <Label>Generated Files ({generatedFiles.length})</Label>
                  <div className="grid grid-cols-1 gap-2">
                    {generatedFiles.map((file, index) => (
                      <div
                        key={index}
                        className={`p-3 rounded-lg border cursor-pointer transition-colors ${
                          selectedFile?.name === file.name
                            ? 'border-primary bg-primary/5'
                            : 'border-border hover:bg-muted'
                        }`}
                        onClick={() => setSelectedFile(file)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {getFileIcon(file.type)}
                            <span className="font-medium text-sm">{file.name}</span>
                            <Badge variant="outline" className="text-xs">
                              {file.type.toUpperCase()}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground">
                              {formatFileSize(file.size)}
                            </span>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={(e) => {
                                e.stopPropagation()
                                downloadFile(file)
                              }}
                              className="h-6 w-6 p-0"
                            >
                              <Download className="h-3 w-3" />
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Code Preview */}
                {selectedFile && (
                  <div className="space-y-2">
                    <Label>Code Preview - {selectedFile.name}</Label>
                    <ScrollArea className="h-[400px] w-full rounded-md border">
                      <pre className="p-4 text-sm">
                        <code className={`language-${getLanguageForHighlighting(selectedFile.type)}`}>
                          {selectedFile.content}
                        </code>
                      </pre>
                    </ScrollArea>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Code className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-sm font-medium">No code generated yet</p>
                <p className="text-xs">Fill in the requirements and click "Generate Code" to start</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}