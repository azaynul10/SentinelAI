
import { ScrollArea } from "@/components/ui/scroll-area"
import { AlertCircle, Check, Info } from "lucide-react"
import { useEffect, useState } from "react"

interface EventLogProps {
    currentState: string
}

interface LogEntry {
    id: number
    time: string
    type: 'alert' | 'info' | 'success'
    message: string
}

export function EventLog({ currentState }: EventLogProps) {
    const [logs, setLogs] = useState<LogEntry[]>([])

    // Initialize on client side to avoid hydration mismatch
    useEffect(() => {
        setLogs([
            { id: 1, time: new Date().toLocaleTimeString(), type: 'info', message: 'System Initialized' }
        ])
    }, [])

    // Effect to add logs on state change
    useEffect(() => {
        if (!currentState || currentState === "WAITING") return

        const type = (currentState === "FALLEN" || currentState === "FALLING") ? 'alert' : 'success'
        if (type === 'success' && Math.random() > 0.1) return // Don't log every 'Standing' frame

        // Limit log spam
        setLogs(prev => {
            const last = prev[0]
            // Debounce same message
            if (last && last.message.includes(currentState) && (Date.now() - new Date("1/1/1970 " + last.time).getTime() < 2000)) return prev

            const newLog: LogEntry = {
                id: Date.now(),
                time: new Date().toLocaleTimeString(),
                type: type,
                message: `State Change: ${currentState}`
            }
            return [newLog, ...prev].slice(0, 20)
        })
    }, [currentState])

    return (
        <div className="w-full h-full bg-slate-900/50 border-l border-slate-800 flex flex-col">
            <div className="p-4 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Recent Events</h3>
            </div>
            <ScrollArea className="flex-1 p-4">
                <div className="space-y-4">
                    {logs.map(log => (
                        <div key={log.id} className="flex gap-3 items-start group">
                            <div className="mt-1">
                                {log.type === 'alert' ? <AlertCircle className="w-4 h-4 text-red-500" /> :
                                    log.type === 'success' ? <Check className="w-4 h-4 text-emerald-500/50" /> :
                                        <Info className="w-4 h-4 text-slate-500" />}
                            </div>
                            <div>
                                <div className="text-xs font-mono text-slate-500 mb-0.5">{log.time}</div>
                                <div className={`text-sm font-medium ${log.type === 'alert' ? 'text-red-400' : 'text-slate-300'}`}>
                                    {log.message}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </ScrollArea>
        </div>
    )
}
