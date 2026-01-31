
"use client"

import { useEffect, useState } from "react"
import { ResponsiveContainer, AreaChart, Area, XAxis, Tooltip, ReferenceLine } from "recharts"
import { Button } from "@/components/ui/button"
import { RefreshCw, PlayCircle } from "lucide-react"

interface TimelineScrubberProps {
    currentScore: number
}

export function TimelineScrubber({ currentScore }: TimelineScrubberProps) {
    const [data, setData] = useState<{ time: number; score: number }[]>([])
    const [replayUrl, setReplayUrl] = useState<string | null>(null)

    useEffect(() => {
        setData((prev) => {
            const now = Date.now()
            const newEntry = { time: now, score: currentScore }
            return [...prev, newEntry].slice(-100) // Longer history for scrubber (10s)
        })
    }, [currentScore])

    const handleReplayRequest = async () => {
        try {
            const res = await fetch('http://localhost:5000/request_instant_replay', { method: 'POST' })
            const json = await res.json()
            if (json.url) {
                setReplayUrl(`http://localhost:5000${json.url}`)
            }
        } catch (e) {
            console.error("Replay fetch failed", e)
        }
    }

    return (
        <div className="w-full h-full flex flex-col bg-slate-950 border-t border-slate-800">
            {/* Header / Controls */}
            <div className="flex items-center justify-between px-4 py-2 border-b border-slate-900 bg-slate-900/50">
                <div className="flex items-center gap-4">
                    <span className="text-xs font-mono text-slate-500 uppercase">Live Telemetry (15s Window)</span>
                    {/* Add Replay Button directly here too */}
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" className="h-6 text-[10px] bg-slate-800 border-slate-700 text-amber-500 hover:text-amber-400 hover:bg-slate-700" onClick={handleReplayRequest}>
                        <RefreshCw className="w-3 h-3 mr-1" /> INSTANT REPLAY
                    </Button>
                </div>
            </div>

            {/* Video Overlay (If Replay Active) */}
            {replayUrl && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-10">
                    <div className="bg-slate-900 p-4 rounded-3xl border border-slate-700 shadow-2xl relative max-w-4xl w-full">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-amber-500 font-bold flex items-center gap-2">
                                <RefreshCw className="w-5 h-5" /> REPLAY REVIEW
                            </h3>
                            <Button variant="ghost" className="rounded-full w-8 h-8 p-0" onClick={() => setReplayUrl(null)}>X</Button>
                        </div>
                        <video src={replayUrl} autoPlay controls className="w-full rounded-xl bg-black" />
                    </div>
                </div>
            )}

            {/* Scrubber Chart */}
            <div className="flex-1 w-full min-h-0 relative cursor-crosshair group pl-0">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                        data={data}
                        onClick={handleReplayRequest} // Click anywhere to trigger replay (implied logic: "click back on spike")
                    >
                        <defs>
                            <linearGradient id="scoreAudit" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <Tooltip
                            contentStyle={{ backgroundColor: "#0f172a", border: "none", color: "#f8fafc", fontSize: "10px" }}
                            itemStyle={{ color: "#f59e0b" }}
                            labelStyle={{ display: "none" }}
                            formatter={(val: number) => [(val * 100).toFixed(0) + "%", "Risk"]}
                            cursor={{ stroke: '#f8fafc', strokeWidth: 1, strokeDasharray: '3 3' }}
                        />
                        <ReferenceLine y={0.6} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'right', value: 'THRESHOLD', fill: '#ef4444', fontSize: 10 }} />
                        <Area
                            type="monotone"
                            dataKey="score"
                            stroke="#f59e0b"
                            fillOpacity={1}
                            fill="url(#scoreAudit)"
                            strokeWidth={2}
                            isAnimationActive={false}
                        />
                    </AreaChart>
                </ResponsiveContainer>

                {/* Hover hints */}
                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity">
                    <div className="bg-black/60 backdrop-blur px-3 py-1 rounded text-xs text-white">
                        Click to Replay Event
                    </div>
                </div>
            </div>
        </div>
    )
}
