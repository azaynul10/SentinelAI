"use client"

import { useState, useRef, useEffect } from "react"
import Webcam from "react-webcam"
import { useLiveDetection } from "@/hooks/use-live-detection"
import { SmartFrame } from "@/components/smart-frame"
import { StatusPill } from "@/components/status-pill"
import { TimelineScrubber } from "@/components/timeline-scrubber"
import { EventLog } from "@/components/event-log"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Upload, Video, Camera, Volume2 } from "lucide-react"

export function LiveMonitor() {
    const { webcamRef, videoRef, data, isConnected, sourceMode, setSourceMode } = useLiveDetection()
    const [videoSrc, setVideoSrc] = useState<string | null>(null)

    const handleTestAudio = async () => {
        try {
            await fetch('http://127.0.0.1:5000/test_audio', { method: 'POST' })
        } catch (e) {
            console.error("Audio test failed:", e)
        }
    }

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            const url = URL.createObjectURL(file)
            setVideoSrc(url)
            setSourceMode("video")
        }
    }

    const currentState = data?.state || "WAITING"
    const currentConf = data?.confidence_score || 0
    const fps = (data as any)?.fps || 30 // Safe access

    // Snapshot Logic (Frontend Only)
    const handleSnapshot = () => {
        const canvas = document.createElement('canvas')
        const w = 1280; const h = 720; // Assume HD
        canvas.width = w; canvas.height = h;
        const ctx = canvas.getContext('2d')
        // Try to draw from webcam or video
        if (sourceMode === "camera" && webcamRef.current?.video) {
            ctx?.drawImage(webcamRef.current.video, 0, 0, w, h)
        } else if (sourceMode === "video" && videoRef.current) {
            ctx?.drawImage(videoRef.current, 0, 0, w, h)
        }
        // Save
        const link = document.createElement('a')
        link.download = `snapshot_${Date.now()}.png`
        link.href = canvas.toDataURL()
        link.click()
    }

    // Force Play when video loaded
    useEffect(() => {
        if (sourceMode === "video" && videoRef.current && videoSrc) {
            videoRef.current.load()
            const playPromise = videoRef.current.play()
            if (playPromise !== undefined) {
                playPromise.catch(error => {
                    console.log("Auto-play prevented:", error)
                })
            }
        }
    }, [sourceMode, videoSrc])

    const togglePlay = () => {
        if (videoRef.current) {
            if (videoRef.current.paused) videoRef.current.play()
            else videoRef.current.pause()
        }
    }

    return (
        <div className="h-screen w-full bg-slate-950 p-4 flex flex-col gap-4 overflow-hidden">

            {/* 1. Header Row (Controls) */}
            <header className="flex-none h-16 flex items-center justify-between">
                <div className="flex flex-col">
                    <h1 className="text-2xl font-black text-white tracking-tighter">
                        SENTINEL <span className="text-emerald-500">AI</span>
                    </h1>
                    <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">
                        Multimodal Fall Detection System v2.0
                    </span>
                </div>

                <div className="flex gap-2">
                    <Button
                        variant={sourceMode === "camera" ? "default" : "outline"}
                        onClick={() => setSourceMode("camera")}
                        className="gap-2 bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700"
                    >
                        <Camera className="w-4 h-4" /> Live
                    </Button>
                    <div className="relative">
                        <Button
                            variant={sourceMode === "video" ? "default" : "outline"}

                            onClick={() => document.getElementById('video-upload')?.click()}
                            className="gap-2 bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700"
                        >
                            <Video className="w-4 h-4" /> Load
                        </Button>
                        <input
                            id="video-upload"
                            type="file"
                            accept="video/*"
                            className="hidden"
                            onChange={handleFileUpload}
                        />
                    </div>
                    <Button
                        variant="outline"
                        onClick={handleTestAudio}
                        className="gap-2 bg-indigo-950/30 border-indigo-500/30 text-indigo-400 hover:bg-indigo-900/50 hover:text-white ml-2 text-xs"
                    >
                        <Volume2 className="w-3 h-3" /> Test Voice
                    </Button>
                </div>
            </header>

            {/* 2. Main Grid */}
            <div className="flex-1 grid grid-cols-12 gap-6 min-h-0 pb-2">

                {/* A. Left Column: Video Feed (Span 8) */}
                <div className="col-span-12 lg:col-span-8 flex flex-col gap-4 min-h-0">
                    {/* Hero Card */}
                    <SmartFrame
                        fps={fps}
                        mode={data?.detection_method || "Hybrid"}
                        isLive={sourceMode === "camera"}
                        onSnapshot={handleSnapshot}
                        className="w-full h-full"
                    >
                        <div className="w-full h-full relative bg-black flex items-center justify-center">
                            <div className="absolute bottom-4 left-4 z-30 flex gap-2">
                                <div className="bg-black/40 backdrop-blur-md px-3 py-1 rounded-full text-xs font-mono text-white/70 border border-white/10">
                                    {data ? `FPS: ${(data as any).fps || 30}` : "Connecting..."}
                                </div>
                                {/* Debug Indicator */}
                                <div className="bg-black/40 backdrop-blur-md px-3 py-1 rounded-full text-xs font-mono text-emerald-400 border border-emerald-500/20">
                                    {isConnected ? "API: OK" : "API: Waiting"}
                                </div>
                            </div>
                            {/* Webcam */}
                            <Webcam
                                ref={webcamRef}
                                width={1280}
                                height={720}
                                className={cn("absolute w-full h-full object-contain pointer-events-none opacity-0")}
                                muted
                            />
                            {/* Video: Always visible to browser (opacity-100), but sitting at z-0 behind the overlay */}
                            {videoSrc && (
                                <video
                                    ref={videoRef}
                                    src={videoSrc}
                                    autoPlay
                                    playsInline
                                    loop
                                    muted
                                    className={cn("absolute w-full h-full object-contain z-0",
                                        sourceMode === "video" ? "block" : "hidden"
                                    )}
                                    onLoadedMetadata={(e) => e.currentTarget.play()}
                                />
                            )}
                            {/* AI Overlay: sits at z-10 on top of the raw video */}
                            {data?.annotated_frame && (
                                <img
                                    src={data.annotated_frame}
                                    alt="Processed Feed"
                                    className="relative z-10 w-full h-full object-contain"
                                />
                            )}
                        </div>
                    </SmartFrame>
                </div>

                {/* B. Right Column: Status & Logs (Span 4) */}
                <div className="col-span-12 lg:col-span-4 flex flex-col gap-4 min-h-0 h-full">
                    {/* Live Status Card */}
                    <div className="flex-none">
                        <StatusPill state={currentState} confidence={currentConf} />
                    </div>

                    {/* Event Log (Fills remaining height) */}
                    <div className="flex-1 bg-slate-900 rounded-3xl overflow-hidden border border-slate-800 shadow-xl min-h-[200px]">
                        <EventLog currentState={currentState} />
                    </div>
                </div>

                {/* C. Bottom Row: Timeline Scrubber (Span 12) */}
                <div className="col-span-12 h-32 bg-slate-900 rounded-3xl overflow-hidden border border-slate-800 shadow-xl">
                    <TimelineScrubber currentScore={currentConf} />
                </div>
            </div>
        </div>
    )
}
