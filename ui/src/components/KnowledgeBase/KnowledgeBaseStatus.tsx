import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { AlertCircle, CheckCircle, Database, Info, RefreshCw } from 'lucide-react'

import { KnowledgeBaseStats, UpdateProgress } from '@/components/KnowledgeBase/interfaces'

export const KnowledgeBaseStatus = (
    {
        handleUpdateKnowledgeBase,
        stats,
        isUpdating,
        updateProgress,
    }: {
        handleUpdateKnowledgeBase: () => void,
        stats: KnowledgeBaseStats,
        isUpdating: boolean,
        updateProgress: UpdateProgress | null
    }
) => {
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
        <>
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
        </>
    )
}
