
import { type AlertElement } from '@/components/KnowledgeBase/types'
import { Alert, AlertDescription } from '@/components/ui/alert'

export const alertTimeout: number = 10000

export const KnowledgeBaseAlerts = ({ alerts }: { alerts: Array<AlertElement> }) => {
    return alerts.map(alert => (
        <Alert key={alert.id} className={alert.type === 'error' ? 'border-red-200 bg-red-50' : alert.type === 'success' ? 'border-green-200 bg-green-50' : ''}>
            <AlertDescription className={alert.type === 'error' ? 'text-red-800' : alert.type === 'success' ? 'text-green-800' : ''}>
                {alert.message}
            </AlertDescription>
        </Alert>
    ))
}
