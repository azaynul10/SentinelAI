
import { cn } from "@/lib/utils"
import { CheckCircle, AlertOctagon, RefreshCw, Activity } from "lucide-react"

interface StatusPillProps {
    state: string
    confidence: number
}

export function StatusPill({ state, confidence }: StatusPillProps) {

    // Determine visuals based on high-level state groups
    let style = "bg-slate-800 border-slate-700 text-slate-400"
    let icon = <Activity className="w-6 h-6" />
    let text = "MONITORING SYSTEM"
    let subtext = "Waiting for input..."
    let animation = ""

    const confPct = (confidence * 100).toFixed(0)

    if (state === "FALLEN" || state === "FALLING") {
        style = "bg-red-500 text-white border-red-400 shadow-[0_0_30px_rgba(239,68,68,0.6)]"
        icon = <AlertOctagon className="w-8 h-8" />
        text = "🚨 FALL DETECTED 🚨"
        subtext = `CONFIDENCE: ${confPct}%`
        animation = "animate-pulse" // Flash
    } else if (state === "STANDING" || state === "SITTING" || state === "BENDING") {
        style = "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 shadow-[0_0_20px_rgba(16,185,129,0.1)]"
        icon = <CheckCircle className="w-6 h-6" />
        text = `SYSTEM MONITORING: ${state}`
        subtext = "Zone Secure • Normal Activity"
    } else if (state === "LYING") {
        style = "bg-orange-500/10 text-orange-400 border-orange-500/20"
        icon = <Activity className="w-6 h-6" />
        text = "⚠️ ACTIVITY WARNING: LYING"
        subtext = "Monitoring Duration..."
    }

    return (
        <div className={cn("flex items-center justify-between p-4 rounded-2xl border-2 transition-all duration-300 w-full", style, animation)}>
            <div className="flex items-center gap-4">
                <div className={cn("p-3 rounded-full bg-white/10 backdrop-blur-sm")}>
                    {icon}
                </div>
                <div className="flex flex-col text-left">
                    <div className="font-black text-lg tracking-tight uppercase leading-none mb-1">
                        {text}
                    </div>
                    <div className="font-mono text-xs opacity-80 uppercase tracking-widest">
                        {subtext}
                    </div>
                </div>
            </div>

            {/* Optional Right Action (e.g. Acknowledge button) could go here */}
            {state === "FALLEN" && (
                <div className="hidden lg:block">
                    <div className="px-3 py-1 bg-white/20 rounded text-xs font-bold animate-pulse">
                        ALARM TRIGGERED
                    </div>
                </div>
            )}
        </div>
    )
}
