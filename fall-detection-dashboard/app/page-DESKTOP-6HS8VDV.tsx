"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { LiveMonitor } from "@/components/live-monitor"

// Create a client
const queryClient = new QueryClient()

export default function Home() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 p-8">
        <main className="max-w-7xl mx-auto space-y-8">
          <div className="flex flex-col gap-2">
            <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl text-slate-900 dark:text-slate-50">
              Fall Detection Dashboard
            </h1>
            <p className="leading-7 text-slate-600 dark:text-slate-400">
              Multimodal Real-time Monitoring System
            </p>
          </div>

          <LiveMonitor />
        </main>
      </div>
    </QueryClientProvider>
  )
}
