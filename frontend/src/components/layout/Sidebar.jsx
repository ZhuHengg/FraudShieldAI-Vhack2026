import React from 'react'
import { ShieldCheck, LayoutDashboard, Search, Activity, ShieldAlert, Zap, FlaskConical } from 'lucide-react'
import clsx from 'clsx'

export default function Sidebar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'search', label: 'Investigation', icon: Search },
    { id: 'radar', label: 'Risk Radar', icon: Activity },

    { id: 'analysis', label: 'Model Insights', icon: ShieldAlert },
    { id: 'simulator', label: 'Model Tuning Lab', icon: Zap },
    { id: 'lab', label: 'Transaction Lab', icon: FlaskConical },
  ]

  return (
    <div className="w-64 h-full bg-bg-50 border-r border-border flex flex-col pt-6 pb-6 shadow-[4px_0_24px_rgba(0,0,0,0.5)] z-20 relative">
      
      {/* Branding */}
      <div className="px-6 mb-10">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 rounded-lg bg-gradient-signature flex items-center justify-center shadow-btn-primary">
            <ShieldCheck className="text-white" size={20} />
          </div>
          <h1 className="font-mono font-bold text-[20px] tracking-tight text-white uppercase">
            Fraud Shield
          </h1>
        </div>
        <p className="font-mono text-[10px] text-cyan-400 uppercase tracking-[0.2em] pl-[44px]">
          AI Defense Matrix
        </p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-1">
        {navItems.map(item => {
          const Icon = item.icon
          const isActive = activeTab === item.id
          
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={clsx(
                "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group relative overflow-hidden",
                isActive 
                  ? "bg-bg-300 text-cyan-400 shadow-[inset_2px_0_0_0_#06b6d4]" 
                  : "text-text-secondary hover:text-text-primary hover:bg-bg-200"
              )}
            >
              <Icon 
                size={18} 
                className={clsx(
                  "transition-colors",
                  isActive ? "text-cyan-400" : "text-text-muted group-hover:text-text-primary"
                )} 
              />
              <span className="font-mono font-bold text-[13px] uppercase tracking-wider">{item.label}</span>
            </button>
          )
        })}
      </nav>

      {/* Connection Status Floor */}
      <div className="px-6 mt-auto">
        <div className="bg-bg-100 rounded-xl p-4 border border-border">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-approve animate-pulse-slow shadow-glow-approve" />
            <span className="font-mono text-[10px] text-text-muted uppercase tracking-[0.15em] font-bold">System Status</span>
          </div>
          <div className="font-mono text-[12px] text-text-primary font-bold">
            All nodes operational
          </div>
          <div className="font-mono text-[11px] text-text-secondary mt-1">
            Latency: 12ms
          </div>
        </div>
      </div>
    </div>
  )
}
