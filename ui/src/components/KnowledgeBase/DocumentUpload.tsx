import { useCallback } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { FileText, Upload } from 'lucide-react'

export const DocumentUpload = (
    {
        handleFileUpload,
        setUploadFiles,
        uploadFiles,
        isUploading
    }: {
        handleFileUpload: () => void,
        setUploadFiles: (files: FileList | null) => void,
        uploadFiles: FileList | null,
        isUploading: boolean
    }
) => {

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

    return (
        <>
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
        </>
    )
}
