"use client"

import { useEffect, useState } from "react"
import {
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    YAxis,
    XAxis,
    CartesianGrid
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ConfidenceChartProps {
    currentScore: number // 0.0 to 1.0
}

export function ConfidenceChart({ currentScore }: ConfidenceChartProps) {
    const [data, setData] = useState<{ time: number; score: number }[]>([])

    useEffect(() => {
        setData((prev) => {
            const now = Date.now()
            const newEntry = { time: now, score: currentScore }
            // Keep last 50 points (approx 5 seconds at 10fps)
            const newData = [...prev, newEntry].slice(-50)
            return newData
        })
    }, [currentScore])

    return (
        <div className="h-full w-full min-h-[150px]">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} stroke="#ffffff" />
                    <YAxis
                        domain={[0, 1]}
                        hide={false}
                        width={30}
                        tickFormatter={(val) => val.toFixed(1)}
                        axisLine={false}
                        tickLine={false}
                        fontSize={10}
                        stroke="#64748b"
                    />
                    <Tooltip
                        contentStyle={{ backgroundColor: "#0f172a", borderColor: "#1e293b", color: "#f8fafc" }}
                        itemStyle={{ color: "#38bdf8" }}
                        labelStyle={{ display: "none" }}
                        formatter={(val: number | undefined) => [val ? val.toFixed(2) : "0.00", "Score"]}
                    />
                    <Line
                        type="monotone"
                        dataKey="score"
                        stroke="#f59e0b" // Amber/Orange
                        strokeWidth={3}
                        dot={false}
                        isAnimationActive={false} // Performance
                        filter="url(#glow)"
                    />
                    {/* SVG Filters for Neon Glow */}
                    <defs>
                        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                            <feMerge>
                                <feMergeNode in="coloredBlur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>
                </LineChart>
            </ResponsiveContainer>
        </div>
    )
}
