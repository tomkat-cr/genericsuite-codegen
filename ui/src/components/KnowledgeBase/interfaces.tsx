export interface KnowledgeBaseStats {
    documentCount: number
    lastUpdated: string
    repositoryUrl: string
    repositoryBranch: string
    status: 'healthy' | 'updating' | 'error' | 'scheduled'
}

export interface UpdateProgress {
    stage: string
    progress: number
    message: string
}

export const IngestionStatus = {
    scheduled: "Scheduled",
    not_started: "Not started",
    cloning: "Cloning",
    processing_files: "Processing files",
    chunking: "Chunking",
    generating_embeddings: "Generating embeddings",
    storing_vectors: "Storing vectors",
    copying_files: "Copying files",
    completed: "Completed",
    failed: "Failed",
}

export interface SearchResult {
    content: string
    document_path: string
    file_type: string
    similarity_score: number
    chunk_index: number
    metadata: Record<string, any>
}
