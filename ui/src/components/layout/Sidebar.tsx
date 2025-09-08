import { Link, useLocation } from 'react-router-dom'
import { 
  Home, 
  Database, 
  MessageSquare, 
  Code, 
  Settings
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigationItems = [
  {
    name: 'Home',
    href: '/',
    icon: Home,
  },
  {
    name: 'Knowledge Base',
    href: '/knowledge-base',
    icon: Database,
  },
  {
    name: 'AI Chat',
    href: '/chat',
    icon: MessageSquare,
  },
  {
    name: 'Code Generation',
    href: '/code-generation',
    icon: Code,
  },
]

export function Sidebar() {
  const location = useLocation()

  return (
    <aside className={cn(
      "fixed left-0 top-16 z-40 h-[calc(100vh-4rem)] w-64",
      "hidden lg:block",
      "border-r bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
    )}>
      <div className="flex h-full flex-col">
        <nav className="flex-1 space-y-1 p-4">
          {navigationItems.map((item) => {
            const isActive = location.pathname === item.href
            const Icon = item.icon
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={cn(
                  "flex items-center space-x-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )}
              >
                <Icon className="h-4 w-4" />
                <span>{item.name}</span>
              </Link>
            )
          })}
        </nav>

        {/* Footer section */}
        <div className="border-t p-4">
          <Link
            to="/settings"
            className={cn(
              "flex items-center space-x-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              location.pathname === '/settings'
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </Link>
        </div>
      </div>
    </aside>
  )
}