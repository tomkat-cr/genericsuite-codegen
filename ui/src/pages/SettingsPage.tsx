import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { apiService, SettingItem, SettingItemType } from '@/lib/api'
import { AlertCircle, CheckCircle2, RefreshCw, Save } from 'lucide-react'
import { useEffect, useState } from 'react'

import { debug } from '@/lib/api'

export function SettingsPage() {
    const [settings, setSettings] = useState<SettingItem[]>([])
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)
    const [modifiedValues, setModifiedValues] = useState<{ [key: string]: string }>({})

    const fetchSettings = async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await apiService.getSettings()
            if (response.success && response.data) {
                setSettings(response.data.settings)
                if (debug) console.log("Settings fetched successfully:", response.data.settings)
                // Initialize modified values with current values
                const initialValues: { [key: string]: string } = {}
                response.data.settings.forEach(item => {
                    if (item.type === SettingItemType.VARIABLE && item.name) {
                        initialValues[item.name] = item.value || ''
                    }
                })
                setModifiedValues(initialValues)
                if (debug) console.log("Initial values:", initialValues)
            } else {
                setError(response.error || 'Failed to fetch settings')
            }
        } catch (err) {
            setError('An unexpected error occurred while fetching settings')
        } finally {
            setLoading(false)
        }
    }

    const convertEnvVarToLabel = (envVar: string) => {
        return envVar.toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }

    useEffect(() => {
        fetchSettings()
    }, [])

    const handleInputChange = (name: string, value: string) => {
        setModifiedValues(prev => ({
            ...prev,
            [name]: value
        }))
    }

    const handleSave = async () => {
        setSaving(true)
        setError(null)
        setSuccess(null)
        try {
            const response = await apiService.updateSettings({ settings: modifiedValues })
            if (response.success) {
                setSuccess('Settings saved successfully')
                // Refresh settings to get resolved values (though they should match)
                await fetchSettings()
            } else {
                setError(response.error || 'Failed to save settings')
            }
        } catch (err) {
            setError('An unexpected error occurred while saving settings')
        } finally {
            setSaving(false)
        }
    }

    if (loading && settings.length === 0) {
        return (
            <div className="flex items-center justify-center h-[calc(100vh-10rem)]">
                <RefreshCw className="h-8 w-8 animate-spin text-primary" />
                <span className="ml-2 text-lg font-medium">Loading settings...</span>
            </div>
        )
    }

    return (
        <div className="container mx-auto py-6 space-y-6 max-w-4xl">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
                    <p className="text-muted-foreground">
                        Configure your application environment variables and settings.
                    </p>
                </div>
                <Button
                    onClick={handleSave}
                    disabled={saving || loading}
                    className="flex items-center gap-2"
                >
                    {saving ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                    {saving ? 'Saving...' : 'Save Changes'}
                </Button>
            </div>

            {error && (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {success && (
                <Alert className="border-green-500 text-green-600 dark:text-green-400">
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                    <AlertTitle>Success</AlertTitle>
                    <AlertDescription>{success}</AlertDescription>
                </Alert>
            )}

            <Card>
                <CardHeader>
                    <CardTitle>Configuration Editor</CardTitle>
                    <CardDescription>
                        Variables are read from <code>.env.example</code> and saved to <code>main_config.json</code>.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {/* <ScrollArea className="h-[calc(100vh-22rem)] pr-4"> */}
                    <div className="space-y-2">
                        {settings.map((item, index) => {

                            if (item.type === SettingItemType.LABEL) {
                                if (item.label.startsWith("http")) {
                                    return (
                                        // <div key={index} className={index === 0 ? "" : "pt-4"}>
                                        <div key={index} className={"pt-1 text-xs text-muted-foreground underline"}>
                                            <a href={item.label} target="_blank">{item.label}</a>
                                        </div>
                                    )
                                }
                                if (item.label.startsWith("#")) {
                                    return (
                                        <div key={index} className={"pt-1 text-xs text-muted-foreground"}>
                                            <span>{item.label.substring(1)}</span>
                                        </div>
                                    )
                                }
                                return (
                                    <div key={index} className={index === 0 ? "pt-2" : "pt-4"}>
                                        <h3 className="text-lg font-semibold text-primary">{item.label}</h3>
                                        <Separator className="mt-2" />
                                    </div>
                                )
                            }

                            if (item.type === SettingItemType.VARIABLE && item.name) {
                                const finalLabel = convertEnvVarToLabel(item.label || item.name)
                                return (
                                    <div key={index} className="grid w-full items-center gap-1.5 px-1">
                                        <Label htmlFor={item.name} className="text-sm font-medium">
                                            {finalLabel}
                                            <span className="ml-2 text-xs text-muted-foreground font-mono">
                                                ({item.name})
                                            </span>
                                        </Label>
                                        {!item.select_options && (
                                            <Input
                                                type="text"
                                                id={item.name}
                                                placeholder={`Enter ${item.name}...`}
                                                value={modifiedValues[item.name] || ''}
                                                onChange={(e) => handleInputChange(item.name!, e.target.value)}
                                                className="font-mono text-sm"
                                            />
                                        )}
                                        {item.select_options && (
                                            <Select
                                                name={item.name}
                                                value={modifiedValues[item.name] || ''}
                                                onValueChange={(value) => handleInputChange(item.name!, value)}
                                            >
                                                <SelectTrigger
                                                    className="w-full font-mono text-sm"
                                                >
                                                    <SelectValue
                                                        placeholder={`Select ${item.name}...`}
                                                    >
                                                        {modifiedValues[item.name] || ''}
                                                    </SelectValue>
                                                </SelectTrigger>
                                                <SelectContent>
                                                    {item.select_options.map((option) => (
                                                        <SelectItem
                                                            key={option || "_blank_option_"}
                                                            value={option || "_blank_option_"}
                                                            className="font-mono text-sm"
                                                        >
                                                            {option}
                                                        </SelectItem>
                                                    ))}
                                                </SelectContent>
                                            </Select>
                                        )}
                                    </div>
                                )
                            }

                            return null
                        })}
                    </div>
                    {/* </ScrollArea> */}
                </CardContent>
                <CardFooter className="flex justify-end border-t pt-6">
                    <Button
                        variant="outline"
                        onClick={fetchSettings}
                        disabled={saving || loading}
                        className="mr-2"
                    >
                        <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                        Reset
                    </Button>
                    <Button
                        onClick={handleSave}
                        disabled={saving || loading}
                    >
                        {saving ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
                        Save Changes
                    </Button>
                </CardFooter>
            </Card>
        </div>
    )
}
