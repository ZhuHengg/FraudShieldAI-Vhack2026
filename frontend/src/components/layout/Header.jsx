import React from 'react'
import { Bell, RefreshCw, Play, Pause, Flame } from 'lucide-react'
import clsx from 'clsx'

const PAGE_TITLES = {
  dashboard: 'Dashboard',
  search: 'TRANSACTION INVESTIGATION',
  radar: 'Risk Radar',
  live: 'Live Simulator',
  analysis: 'Model Insights',
  simulator: 'Fraud Simulator',
}

export default function Header({ engine, activeTab }) {
  const { resetParams, triggerAttackBurst, isRunning, setIsRunning } = engine
  const title = PAGE_TITLES[activeTab] || 'Dashboard'

  return (
    <header className="h-[80px] bg-bg-base flex items-center justify-between px-8 border-b border-border z-10 relative">
      <div>
        <h2 className="font-mono font-bold text-[22px] text-text-primary uppercase tracking-tight">
          {title}
        </h2>
        <p className="font-mono text-[11px] text-text-muted mt-1">
          {new Date().toISOString().split('T')[0]} | Production Env
        </p>
      </div>

      <div className="flex items-center gap-4">
        
        {/* Connection Pulse */}
        <div className="hidden md:flex items-center gap-2 mr-4 bg-bg-100 px-3 py-1.5 rounded-full border border-border">
          <div className={clsx("w-2 h-2 rounded-full shadow-glow-cyan animate-pulse", isRunning ? 'bg-cyan-500' : 'bg-text-muted')} />
          <span className="font-mono text-[11px] font-bold text-text-secondary uppercase">
            {isRunning ? 'TXN Stream Active' : 'Stream Paused'}
          </span>
        </div>

        {/* Global Controls */}
        <div className="flex items-center gap-2 bg-bg-50 p-1.5 rounded-2xl border border-border shadow-sm">
          <button 
            onClick={resetParams}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-mono font-bold text-text-secondary hover:text-text-primary hover:bg-bg-200 transition-colors uppercase tracking-wide"
          >
            <RefreshCw size={14} />
            Reset Defaults
          </button>
          
          <div className="w-[1px] h-6 bg-border mx-1" />

          {/* Yellow styling for Play/Pause as per screenshot rules inference */}
          <button 
            onClick={() => setIsRunning(p => !p)}
            className={clsx(
              "flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-mono font-bold transition-all border uppercase tracking-wide",
              isRunning 
                ? "bg-[rgba(245,158,11,0.1)] text-[#f59e0b] border-[rgba(245,158,11,0.3)] hover:bg-[rgba(245,158,11,0.2)]"
                : "bg-bg-300 text-text-primary border-border hover:bg-bg-200"
            )}
          >
            {isRunning ? <Pause size={14} /> : <Play size={14} />}
            {isRunning ? 'Pause Sim' : 'Resume Sim'}
          </button>

          {/* Red styling for Attack Burst */}
          <button 
            onClick={triggerAttackBurst}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[12px] font-mono font-bold text-[#ef4444] bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.3)] hover:bg-[rgba(239,68,68,0.2)] transition-colors shadow-glow-block ml-1 uppercase tracking-wide"
          >
            <Flame size={14} />
            Attack Burst
          </button>
        </div>

        {/* User / Alerts */}
        <div className="ml-4 flex items-center gap-3">
          <button className="w-10 h-10 rounded-full bg-bg-100 border border-border flex items-center justify-center text-text-secondary hover:text-text-primary transition-colors relative">
            <Bell size={18} />
            <span className="absolute top-2 right-2.5 w-2 h-2 rounded-full bg-block" />
          </button>
          <div className="w-10 h-10 rounded-full bg-gradient-signature p-[2px]">
            <div className="w-full h-full rounded-full bg-bg-base border-2 border-transparent flex items-center justify-center">
              <span className="font-mono font-bold text-[13px] text-white">AD</span>
            </div>
          </div>
        </div>

      </div>
    </header>
  )
}
