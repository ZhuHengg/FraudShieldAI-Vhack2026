import React from 'react'
import Panel from '../shared/Panel'
import { Zap, RefreshCcw } from 'lucide-react'
import clsx from 'clsx'

export default function TuningPanel({ 
  params, updateParam, weights, thresholds, 
  triggerAttackBurst, resetParams, className
}) {
  const SliderRow = ({ label, value, keyName, min, max, step }) => {
    const pct = ((value - min) / (max - min)) * 100
    
    return (
      <div className="mb-5 last:mb-0">
        <div className="flex justify-between items-center mb-2">
          <span className="font-mono text-[11px] text-text-secondary font-bold uppercase tracking-wider">{label}</span>
          <span className="font-mono text-[13px] font-bold text-cyan-400">{value.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => updateParam(keyName, parseFloat(e.target.value))}
          className="w-full"
          style={{
            background: `linear-gradient(to right, var(--cyan-500) 0%, var(--cyan-500) ${pct}%, var(--bg-300) ${pct}%, var(--bg-300) 100%)`
          }}
        />
      </div>
    )
  }

  // Use thresholds from backend, defaults to 0.35 and 0.60 (per requirement)
  let approveThreshold = thresholds?.approve ?? 0.35
  let blockThreshold   = thresholds?.block   ?? 0.60

  // Normalize if backend returns 0-100 scale
  if (approveThreshold > 1) approveThreshold /= 100
  if (blockThreshold > 1)   blockThreshold   /= 100

  const greenPct = approveThreshold * 100
  const amberPct = (blockThreshold - approveThreshold) * 100
  const redPct   = (1 - blockThreshold) * 100

  // Model weights (from score_fusion.py)
  const wLgb = weights?.lgb ?? 0.55
  const wIso = weights?.iso ?? 0.25
  const wBeh = weights?.beh ?? 0.20

  return (
    <Panel className={clsx("border border-border shrink-0", className)}>
      <div className="flex justify-between items-center mb-6">
        <h3 className="section-label">Simulation Tuning</h3>
      </div>

      <div className="space-y-1 mb-8">
        <SliderRow label="Simulation Speed" value={params.simulationSpeed} keyName="simulationSpeed" min={0.10} max={5.00} step={0.10} />
        <SliderRow label="SMOTE Aggressiveness" value={params.smoteLevel} keyName="smoteLevel" min={0.00} max={1.00} step={0.05} />
      </div>

      <div className="grid grid-cols-2 gap-3 mb-8">
        <button
          onClick={triggerAttackBurst}
          className="flex items-center justify-center gap-2 py-2 px-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 rounded-xl font-mono text-[11px] font-bold uppercase tracking-wider transition-all active:scale-95"
        >
          <Zap size={14} className="fill-current" />
          Attack Burst
        </button>
        <button
          onClick={resetParams}
          className="flex items-center justify-center gap-2 py-2 px-3 bg-white/5 hover:bg-white/10 text-text-muted border border-white/10 rounded-xl font-mono text-[11px] font-bold uppercase tracking-wider transition-all active:scale-95"
        >
          <RefreshCcw size={14} />
          Reset Specs
        </button>
      </div>

      {/* Model Weights (read-only display) */}
      <div className="mb-6">
        <h4 className="section-label mb-3">Ensemble Model Weights</h4>
        <div className="space-y-2 font-mono text-[11px]">
          <div className="flex justify-between items-center">
            <span className="text-cyan-400">LightGBM</span>
            <span className="text-text-secondary font-bold">{(wLgb * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-purple-400">Isolation Forest</span>
            <span className="text-text-secondary font-bold">{(wIso * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-amber-400">Behavioral Rules</span>
            <span className="text-text-secondary font-bold">{(wBeh * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Decision Boundaries (fixed from real model) */}
      <div>
        <h4 className="section-label mb-2">Decision Boundaries</h4>
        <div className="flex h-3 rounded-full overflow-hidden w-full bg-bg-200 shadow-inner">
          <div className="h-full bg-[#10b981] transition-all duration-300" style={{ width: `${greenPct}%` }} />
          <div className="h-full bg-[#f59e0b] transition-all duration-300" style={{ width: `${amberPct}%` }} />
          <div className="h-full bg-[#ef4444] transition-all duration-300" style={{ width: `${redPct}%` }} />
        </div>
        <div className="flex justify-between mt-2 font-mono text-[10px] text-text-muted">
          <span>0.0</span>
          <span className="text-[#10b981]">APPROVE ≤ {approveThreshold.toFixed(2)}</span>
          <span className="text-[#f59e0b]">FLAG</span>
          <span className="text-[#ef4444]">BLOCK &gt; {blockThreshold.toFixed(2)}</span>
          <span>1.0</span>
        </div>
      </div>
    </Panel>
  )
}
