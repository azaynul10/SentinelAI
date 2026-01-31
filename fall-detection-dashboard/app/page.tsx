"use client"

import { LiveMonitor } from "@/components/live-monitor";

export default function DashboardPage() {
  return (
    <main className="h-screen w-full bg-[#020617] text-slate-50 overflow-hidden">
      <LiveMonitor />
    </main>
  );
}
