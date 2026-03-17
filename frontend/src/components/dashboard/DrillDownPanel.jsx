import React from 'react'
import { X, Globe2, Monitor, MapPin, Tag, Activity } from 'lucide-react'
import clsx from 'clsx'
import { formatCurrency, formatTime } from '../../utils/formatters'

export default function DrillDownPanel({ txn, onClose }) {
  if (!txn) return null

  // Ensure click outside closes
  const handleBackdrop = (e) => {
    if (e.target === e.currentTarget) onClose()
  }

  const ensembleScoreNum = Math.round(txn.ensembleScore * 100)

  const BarPill = ({ label, pct, colorStr }) => (
    <div className="flex items-center gap-3 mb-3">
      <span className="font-mono text-[11px] text-text-secondary w-20 truncate uppercase tracking-wide">{label}</span>
      <div className="flex-1 h-2 bg-bg-300 rounded-full overflow-hidden">
        <div 
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${Math.round(pct * 100)}%`, background: colorStr }}
        />
      </div>
      <span className="font-mono text-[11px] font-bold text-text-primary w-10 text-right">
        {Math.round(pct * 100)}%
      </span>
    </div>
  )

  return (
    <div 
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[rgba(10,14,23,0.7)] backdrop-blur-md animate-fade-in p-4"
      onClick={handleBackdrop}
    >
      <div className="bg-bg-100 w-[420px] rounded-[24px] shadow-card relative flex flex-col pointer-events-auto border border-border">
        
        <button 
          onClick={onClose}
          className="absolute top-5 right-5 text-text-muted hover:text-white transition-colors bg-bg-200 hover:bg-bg-300 rounded-full p-1"
        >
          <X size={18} />
        </button>

        <div className="p-6 pb-4 border-b border-border">
          <p className="font-mono text-text-muted text-[11px] mb-1">{txn.id}</p>
          <h2 className="font-mono font-bold text-[18px] text-text-primary mb-1 uppercase tracking-tight">
            Transaction Detail
          </h2>
          <p className="font-mono text-text-secondary text-[12px]">
            {formatTime(txn.timestamp)}
          </p>
        </div>

        <div className="p-6 overflow-y-auto custom-scrollbar max-h-[70vh]">
          {/* Hero Score Block */}
          <div className="bg-bg-50 rounded-2xl p-6 flex flex-col items-center justify-center mb-6 relative border border-border">
             <div className="mb-1 section-label">Ensemble Risk Score</div>
             <div className="font-mono font-bold text-[56px] leading-none mb-3 gradient-text">
               {ensembleScoreNum}
             </div>
             <div className="bg-bg-200 px-3 py-1 rounded-pill font-mono text-[10px] text-text-secondary border border-border">
               Latency: {txn.latencyMs}ms
             </div>
          </div>

          {/* Model Contributions */}
          <div className="mb-6">
            <h3 className="section-label mb-3">Model Contributions</h3>
            <BarPill label="LightGBM" pct={txn.lgbScore || 0} colorStr="var(--cyan-500)" />
            <BarPill label="Iso Forest" pct={txn.isoScore || 0} colorStr="var(--purple-500)" />
            <BarPill label="Behavioral" pct={txn.behScore || 0} colorStr="var(--amber-500)" />
            <BarPill label="Ensemble" pct={txn.ensembleScore || 0} colorStr="linear-gradient(135deg, #06b6d4, #14b8a6)" />
          </div>

          {/* Metadata Grid */}
          <div className="grid grid-cols-2 gap-3 mb-6">
            <div className="bg-bg-50 p-3 rounded-xl border border-border">
              <div className="flex items-center gap-1.5 text-text-muted mb-1">
                <Globe2 size={12} /> <span className="section-label">Origin</span>
              </div>
              <p className="font-mono text-text-primary text-[13px] font-bold">{txn.country}</p>
            </div>
            <div className="bg-bg-50 p-3 rounded-xl border border-border">
              <div className="flex items-center gap-1.5 text-text-muted mb-1">
                <Monitor size={12} /> <span className="section-label">Device</span>
              </div>
              <p className="font-mono text-text-primary text-[13px] font-bold uppercase tracking-tight">{txn.deviceType}</p>
            </div>
            <div className="bg-bg-50 p-3 rounded-xl border border-border">
              <div className="flex items-center gap-1.5 text-text-muted mb-1">
                <Tag size={12} /> <span className="section-label">Category</span>
              </div>
              <p className="font-mono text-text-primary text-[13px] font-bold uppercase tracking-tight">{txn.merchantCategory}</p>
            </div>
            <div className="bg-bg-50 p-3 rounded-xl border border-border">
              <div className="flex items-center gap-1.5 text-text-muted mb-1">
                <Activity size={12} /> <span className="section-label">User ID</span>
              </div>
              <p className="font-mono text-text-primary text-[13px] font-bold">{txn.userId}</p>
            </div>
          </div>

          <div className="flex items-end justify-between mb-6 pb-6 border-b border-border">
            <div className="section-label">Total Amount</div>
            <div className="font-mono font-bold text-[20px] text-text-primary">
              {formatCurrency(txn.amount)}
            </div>
          </div>

          {/* Risk Signals */}
          <div className="mb-6">
            <h3 className="section-label mb-3">Risk Signals Detected</h3>
            {txn.riskFactors.length === 0 ? (
              <p className="text-[12px] text-text-muted font-mono">No suspicious signals detected.</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {txn.riskFactors.map((rf, i) => (
                  <span key={i} className="bg-[rgba(245,158,11,0.1)] text-[#f59e0b] border border-[rgba(245,158,11,0.3)] px-2.5 py-1 rounded-pill text-[10px] font-bold font-mono flex items-center gap-1.5 uppercase tracking-wide">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#f59e0b]" />
                    {rf}
                  </span>
                ))}
            </div>
            )}
          </div>

          {/* Ground Truth */}
          <div>
             <h3 className="section-label mb-2">Ground Truth Analysis</h3>
             {txn.isFraud ? (
                <div className="inline-flex items-center px-3 py-1.5 rounded-pill bg-[rgba(239,68,68,0.1)] text-[#ef4444] border border-[rgba(239,68,68,0.3)] text-[11px] font-bold font-mono uppercase tracking-widest">
                  FRAUDULENT
                </div>
             ) : (
                <div className="inline-flex items-center px-3 py-1.5 rounded-pill bg-[rgba(16,185,129,0.1)] text-[#10b981] border border-[rgba(16,185,129,0.3)] text-[11px] font-bold font-mono uppercase tracking-widest">
                  LEGITIMATE
                </div>
             )}
          </div>
        </div>
      </div>
    </div>
  )
}
