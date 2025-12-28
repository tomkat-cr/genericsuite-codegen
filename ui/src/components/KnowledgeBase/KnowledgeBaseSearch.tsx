import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { baseUrl } from '@/lib/api'
import { ChevronDown, ChevronRight, FileText, Search } from 'lucide-react'

import { SearchResult } from '@/components/KnowledgeBase/interfaces'
import { debug } from '@/lib/api'

export function KnowledgeBaseSearch() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [expandedResults, setExpandedResults] = useState<Set<number>>(new Set())

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsSearching(true)
    setSearchError(null)
    setSearchResults([])
    setExpandedResults(new Set())

    try {
      const response = await fetch(`${baseUrl}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: searchQuery }),
      })

      if (response.ok && response.status === 200) {
        const data = await response.json()
        if (data.data && data.data.results) {
          setSearchResults(data.data.results)
          if (debug) console.log('>>> searchResults:', data.data.results)
        } else {
          // Handle case where success is true but data structure might be different or empty results
          setSearchResults([])
        }
      } else {
        setSearchError('Failed to perform search. Please try again.')
      }
    } catch (error) {
      setSearchError('An error occurred while searching.')
      console.error('Search error:', error)
    } finally {
      setIsSearching(false)
    }
  }

  const toggleResult = (index: number) => {
    const newExpanded = new Set(expandedResults)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedResults(newExpanded)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Search Knowledge Base
        </CardTitle>
        <CardDescription>
          Search for information within the ingested documents
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex gap-2">
          <Input
            placeholder="Enter search query..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          />
          <Button onClick={handleSearch} disabled={isSearching}>
            {isSearching ? 'Searching...' : 'Search'}
          </Button>
        </div>

        {searchError && (
          <div className="text-sm text-red-500">{searchError}</div>
        )}

        <div className="space-y-2">
          {searchResults.map((result, index) => (
            <div key={index} className="border rounded-lg p-3">
              <div
                className="flex items-center gap-2 cursor-pointer hover:bg-muted/50 p-1 rounded"
                onClick={() => toggleResult(index)}
              >
                {expandedResults.has(index) ? (
                  <ChevronDown className="h-4 w-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                )}
                <FileText className="h-4 w-4 text-blue-500" />
                <span className="text-sm font-medium truncate flex-1">
                  {result.document_path}
                </span>
                <span className="text-xs text-muted-foreground">
                  {Math.round(result.similarity_score * 100)}% match
                </span>
              </div>

              {expandedResults.has(index) && (
                <div className="mt-3 pl-6 space-y-2 text-sm">
                  <>
                    <div className="text-xs text-muted-foreground font-semibold mb-1">Content:</div>
                    <div className="p-2 bg-muted rounded text-xs font-mono whitespace-pre-wrap">
                      {result.content}
                    </div>
                  </>

                  <div className="text-xs text-muted-foreground flex items-center gap-2">
                    <div className="font-semibold">Document Path:</div>
                    <a
                      href={result.document_path}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-medium truncate flex-1 text-blue-500 hover:underline"
                    >
                      {result.document_path}
                    </a>
                  </div>

                  <div className="text-xs text-muted-foreground flex items-center gap-2">
                    <div className="font-semibold mb-1">File Type:</div>
                    <div className="inline-flex items-center rounded-full border px-2.5 py-0.5 font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 text-foreground text-xs">
                      {result.file_type}
                    </div>
                  </div>
                  {Object.keys(result.metadata).length > 0 && (
                    <div className="text-xs text-muted-foreground">
                      <div className="font-semibold mb-1">Metadata:</div>
                      <div className="p-2 bg-muted rounded text-xs font-mono whitespace-pre-wrap">
                        {JSON.stringify(result.metadata, null, 2)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          {searchResults.length === 0 && !isSearching && searchQuery && !searchError && (
            <div className="text-sm text-muted-foreground text-center py-4">
              No results found.
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
