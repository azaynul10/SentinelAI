
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Camera, Maximize, MoreVertical, Mic, MicOff, Aperture, Video } from "lucide-react"
import { useState } from "react"

interface SmartFrameProps {
    children: React.ReactNode
    fps: number
    mode: string
    isLive: boolean
    onSnapshot: () => void
    className?: string
}

export function SmartFrame({ children, fps, mode, isLive, onSnapshot, className }: SmartFrameProps) {
    const [muted, setMuted] = useState(true)
    const [recording, setRecording] = useState(false)

    return (
        <div className={cn("relative group overflow-hidden rounded-3xl border border-slate-800 bg-slate-950 shadow-2xl", className)}>

            {/* VIDEO CONTENT */}
            <div className="w-full h-full relative z-0">
                {children}
            </div>

            {/* OVERLAY: Top HUD */}
            <div className="absolute top-0 left-0 w-full p-4 flex justify-between items-start z-10 bg-gradient-to-b from-black/80 to-transparent pointer-events-none">

                {/* Top Left: Camera Info */}
                <div className="flex flex-col gap-1 pointer-events-auto">
                    <div className="flex items-center gap-2 text-xs font-mono text-emerald-400">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
                        LIVE FEED
                    </div>
                    <div className="text-sm font-bold text-white tracking-widest flex items-center gap-2">
                        <Camera className="w-4 h-4 opacity-70" />
                        CAMERA 01
                    </div>
                    <div className="text-[10px] text-slate-400 font-mono">
                        {fps} FPS • 1080p
                    </div>
                </div>

                {/* Top Right: Mode Selector */}
                <div className="pointer-events-auto">
                    <Badge variant="outline" className="bg-black/40 border-slate-700 text-slate-300 backdrop-blur-md px-3 py-1 font-mono text-xs">
                        {mode}
                    </Badge>
                </div>
            </div>

            {/* OVERLAY: Bottom Action Bar */}
            <div className="absolute bottom-0 left-0 w-full p-6 flex flex-col items-center justify-end z-20 bg-gradient-to-t from-black/90 via-black/50 to-transparent h-1/3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">

                {/* Floating Toolbar */}
                <div className="flex items-center gap-4 bg-black/60 backdrop-blur-xl border border-white/10 rounded-full px-6 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.5)] transform translate-y-2 group-hover:translate-y-0 transition-transform">

                    <Button
                        size="icon"
                        variant="ghost"
                        className="rounded-full w-10 h-10 text-white hover:bg-white/20 hover:text-white"
                        onClick={onSnapshot}
                        title="Snapshot"
                    >
                        <Aperture className="w-5 h-5" />
                    </Button>

                    <Button
                        size="icon"
                        variant="ghost"
                        className={cn("rounded-full w-12 h-12 border-2", recording ? "border-red-500 text-red-500 animate-pulse" : "border-white/20 text-white hover:bg-white/20")}
                        onClick={() => setRecording(!recording)}
                        title="Record Clip"
                    >
                        <div className={cn("transition-all", recording ? "w-3 h-3 bg-red-500 rounded-sm" : "w-4 h-4 bg-red-500 rounded-full")} />
                    </Button>

                    <Button
                        size="icon"
                        variant="ghost"
                        className="rounded-full w-10 h-10 text-white hover:bg-white/20 hover:text-white"
                        onClick={() => setMuted(!muted)}
                        title={muted ? "Unmute" : "Mute"}
                    >
                        {muted ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                    </Button>

                </div>
            </div>

            {/* CORNER ACCENTS (Cyberpunk style) */}
            <div className="absolute top-4 left-4 w-16 h-16 border-t-2 border-l-2 border-emerald-500/20 rounded-tl-xl pointer-events-none" />
            <div className="absolute top-4 right-4 w-16 h-16 border-t-2 border-r-2 border-emerald-500/20 rounded-tr-xl pointer-events-none" />
            <div className="absolute bottom-4 left-4 w-16 h-16 border-b-2 border-l-2 border-emerald-500/20 rounded-bl-xl pointer-events-none" />
            <div className="absolute bottom-4 right-4 w-16 h-16 border-b-2 border-r-2 border-emerald-500/20 rounded-br-xl pointer-events-none" />

        </div>
    )
}
