import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Menu, X, Code2 } from 'lucide-react'

export function Header() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Logo and Brand */}
        <Link to="/" className="flex items-center space-x-2">
          <Code2 className="h-6 w-6 text-primary" />
          <span className="font-bold text-xl">GenericSuite CodeGen</span>
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center space-x-6">
          <Link 
            to="/" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Home
          </Link>
          <Link 
            to="/knowledge-base" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Knowledge Base
          </Link>
          <Link 
            to="/chat" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            AI Chat
          </Link>
          <Link 
            to="/code-generation" 
            className="text-sm font-medium transition-colors hover:text-primary"
          >
            Code Generation
          </Link>
        </nav>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden p-2 rounded-md hover:bg-accent"
          onClick={toggleMobileMenu}
          aria-label="Toggle mobile menu"
        >
          {isMobileMenuOpen ? (
            <X className="h-5 w-5" />
          ) : (
            <Menu className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMobileMenuOpen && (
        <div className="md:hidden border-t bg-background">
          <nav className="flex flex-col space-y-1 p-4">
            <Link 
              to="/" 
              className="px-3 py-2 text-sm font-medium rounded-md hover:bg-accent transition-colors"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Home
            </Link>
            <Link 
              to="/knowledge-base" 
              className="px-3 py-2 text-sm font-medium rounded-md hover:bg-accent transition-colors"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Knowledge Base
            </Link>
            <Link 
              to="/chat" 
              className="px-3 py-2 text-sm font-medium rounded-md hover:bg-accent transition-colors"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              AI Chat
            </Link>
            <Link 
              to="/code-generation" 
              className="px-3 py-2 text-sm font-medium rounded-md hover:bg-accent transition-colors"
              onClick={() => setIsMobileMenuOpen(false)}
            >
              Code Generation
            </Link>
          </nav>
        </div>
      )}
    </header>
  )
}