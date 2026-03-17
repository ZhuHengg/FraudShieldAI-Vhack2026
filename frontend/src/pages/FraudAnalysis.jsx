import React, { useState, useMemo } from 'react'
import {
  Play, Pause, Flame, RefreshCw, X, AlertTriangle, ShieldCheck, Search
} from 'lucide-react'
import clsx from 'clsx'
import Panel from '../components/shared/Panel'
import { formatCurrency } from '../utils/formatters'
import TransactionInspector from '../components/TransactionInspector'

// --- Simple Pearson Correlation ---
function pearsonCorrelation(x, y) {
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  const n = x.length;
  if (n === 0) return 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i]; sumY += y[i]; sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i]; sumY2 += y[i] * y[i];
  }
  const numerator = (n * sumXY) - (sumX * sumY);
  const denominator = Math.sqrt(((n * sumX2) - (sumX * sumX)) * ((n * sumY2) - (sumY * sumY)));
  if (denominator === 0) return 0;
  return numerator / denominator;
}

export default function FraudAnalysis({ engine }) {
  const allTxns = engine?.allTransactions || []
  const {
    params, isRunning, setIsRunning, triggerAttackBurst, resetParams,
    selectedTxn, setSelectedTxn
  } = engine || {}

  // Fallback to internal selection if setSelectedTxn missing
  const [internalSelectedTxn, setInternalSelectedTxn] = useState(null)
  const currentTxn = selectedTxn || internalSelectedTxn
  const handleSelectTxn = (t) => {
    if (setSelectedTxn) setSelectedTxn(t)
    else setInternalSelectedTxn(t)
  }

  // --- STATE ---
  const [weights, setWeights] = useState({ lgb: 0.55, iso: 0.25, beh: 0.20 })
  const [thresholds, setThresholds] = useState({ approve: 35, flag: 60 })

  const handleWeightChange = (key, value) => {
    let newVal = Math.max(0, Math.min(100, value)) / 100
    setWeights(prev => {
      const delta = newVal - prev[key]
      const others = Object.keys(prev).filter(k => k !== key)
      let sumOthers = prev[others[0]] + prev[others[1]]
      
      let next = { ...prev, [key]: newVal }
      if (sumOthers === 0) {
        next[others[0]] = (1.0 - newVal) / 2
        next[others[1]] = (1.0 - newVal) / 2
      } else {
        next[others[0]] = Math.max(0, prev[others[0]] - delta * (prev[others[0]] / sumOthers))
        next[others[1]] = Math.max(0, prev[others[1]] - delta * (prev[others[1]] / sumOthers))
      }
      
      // Normalize to fix float math issues
      const total = next.lgb + next.iso + next.beh
      return { lgb: next.lgb/total, iso: next.iso/total, beh: next.beh/total }
    })
  }

  // Action Button component
  const ActionBtn = ({ icon: Icon, label, color, onClick }) => (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-3 py-1.5 rounded-lg font-mono text-[10px] font-bold tracking-widest uppercase transition-all border whitespace-nowrap",
        color === 'yellow' ? "bg-[rgba(245,158,11,0.1)] text-[#f59e0b] border-[rgba(245,158,11,0.3)] hover:bg-[rgba(245,158,11,0.15)]" :
        color === 'red' ? "bg-[rgba(239,68,68,0.1)] text-[#ef4444] border-[rgba(239,68,68,0.3)] hover:bg-[rgba(239,68,68,0.15)]" :
        "bg-transparent text-text-secondary border-border hover:bg-bg-200"
      )}
    >
      <Icon size={12} />
      {label}
    </button>
  )

  // Top Stats calculations
  const topStats = useMemo(() => {
    const lgbAvg = allTxns.length ? allTxns.reduce((s, t) => s + (t.lgbScore || 0), 0) / allTxns.length : 0
    const isoRate = allTxns.length ? allTxns.filter(t => (t.isoScore || 0) > 0.5).length / allTxns.length : 0
    const behRate = allTxns.length ? allTxns.filter(t => (t.behScore || 0) > 0).length / allTxns.length : 0
    const disRate = allTxns.length ? allTxns.filter(t => Math.abs((t.lgbScore || 0) - (t.isoScore || 0)) > 0.3).length / allTxns.length : 0
    return { lgbAvg, isoRate, behRate, disRate }
  }, [allTxns])

  // Live Recalculations
  const sim = useMemo(() => {
    let approve = 0, flag = 0, block = 0
    let tp = 0, fp = 0, fn = 0, tn = 0

    allTxns.forEach(t => {
      const liveScore = (
        (t.lgbScore || 0) * weights.lgb +
        (t.isoScore || 0) * weights.iso +
        (t.behScore || 0) * weights.beh
      ) * 100

      let decision = 'APPROVE'
      if (liveScore >= thresholds.flag) decision = 'BLOCK'
      else if (liveScore >= thresholds.approve) decision = 'FLAG'

      if (decision === 'APPROVE') approve++
      else if (decision === 'FLAG') flag++
      else block++

      // For precision/recall: BLOCK is Positive prediction
      const predictedFraud = decision === 'BLOCK'
      const actualFraud = !!t.isFraud

      if (predictedFraud && actualFraud) tp++
      else if (predictedFraud && !actualFraud) fp++
      else if (!predictedFraud && actualFraud) fn++
      else tn++
    })

    const len = allTxns.length || 1
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0
    const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0

    return { 
      approve, flag, block,
      approvePct: (approve / len) * 100,
      flagPct: (flag / len) * 100,
      blockPct: (block / len) * 100,
      precision, recall, fpr, f1 
    }
  }, [allTxns, weights, thresholds])

  // Matrix calculations
  const matrixData = useMemo(() => {
    let bothSafe = 0, isoFlags = 0, lgbFlags = 0, bothFraud = 0
    allTxns.forEach(t => {
      const lgb = (t.lgbScore || 0) > 0.5
      const iso = (t.isoScore || 0) > 0.5
      if (!lgb && !iso) bothSafe++
      else if (!lgb && iso) lgbFlags++
      else if (lgb && !iso) isoFlags++
      else bothFraud++
    })
    const tot = allTxns.length || 1
    const disagree = isoFlags + lgbFlags
    return { 
      bothSafe, isoFlags, lgbFlags, bothFraud, disagree,
      pctBothSafe: (bothSafe/tot)*100, pctIso: (isoFlags/tot)*100, pctLgb: (lgbFlags/tot)*100, pctBothFraud: (bothFraud/tot)*100
    }
  }, [allTxns])

  // Behavioral Contributions
  const behData = useMemo(() => {
    const rules = [
      { id: 'drain', name: 'Drain → Unknown', weight: '35%', color: '#ef4444', 
        features: 'sender_fully_drained + is_new_recipient',
        trigger: t => t.senderFullyDrained && t.isNewRecipient 
      },
      { id: 'amt', name: 'Amt Deviation', weight: '25%', color: '#f59e0b', 
        features: 'amount_vs_avg_ratio',
        trigger: t => (t.amountVsAvgRatio || 1) > 1.5 
      },
      { id: 'ctx', name: 'Risky Context', weight: '20%', color: '#a855f7', 
        features: 'ip_risk_score + country_mismatch + is_new_device',
        trigger: t => (t.ipRiskScore || 0) > 0.5 || t.countryMismatch || t.isNewDevice 
      },
      { id: 'rap', name: 'Rapid Session', weight: '20%', color: '#00e5ff', 
        features: 'tx_count_24h + session_duration_seconds',
        trigger: t => (t.txCount24h > 5) || (t.sessionDurationSeconds < 60) 
      },
    ]

    return rules.map(r => {
      let triggered = 0
      let riskWhenTrig = 0
      let riskWhenNot = 0
      
      allTxns.forEach(t => {
        const isTrig = r.trigger(t)
        const risk = t.ensembleScore || 0
        if (isTrig) {
          triggered++
          riskWhenTrig += risk
        } else {
          riskWhenNot += risk
        }
      })

      const notTriggered = allTxns.length - triggered
      return {
        ...r,
        pct: allTxns.length ? (triggered / allTxns.length) * 100 : 0,
        avgRiskTrig: triggered ? (riskWhenTrig / triggered) * 100 : 0,
        avgRiskNotTrig: notTriggered ? (riskWhenNot / notTriggered) * 100 : 0
      }
    })
  }, [allTxns])

  // Feature Importance Correlations
  const featuresList = useMemo(() => {
    const scores = allTxns.map(t => t.ensembleScore || 0)
    
    const extract = (key, defaultVal=0, isBool=false) => {
      return allTxns.map(t => {
         const v = t[key] ?? defaultVal
         return isBool ? (v ? 1 : 0) : v
      })
    }

    const map = [
      { key: 'amountVsAvgRatio', label: 'amount_vs_avg_ratio', rule: 'Amt Deviation', color: '#f59e0b' },
      { key: 'ipRiskScore', label: 'ip_risk_score', rule: 'Risky Context', color: '#a855f7' },
      { key: 'txCount24h', label: 'tx_count_24h', rule: 'Rapid Session', color: '#00e5ff' },
      { key: 'sessionDurationSeconds', label: 'session_duration_seconds', rule: 'Rapid Session', color: '#00e5ff' },
      { key: 'isNewDevice', label: 'is_new_device', rule: 'Risky Context', isBool: true, color: '#a855f7' },
      { key: 'countryMismatch', label: 'country_mismatch', rule: 'Risky Context', isBool: true, color: '#a855f7' },
      { key: 'senderFullyDrained', label: 'sender_fully_drained', rule: 'Drain → Unknown', isBool: true, color: '#ef4444' },
    ]

    return map.map(f => {
      const vals = extract(f.key, 0, f.isBool)
      const avg = vals.length ? vals.reduce((a,b)=>a+b, 0) / vals.length : 0
      const corr = pearsonCorrelation(vals, scores)
      return { ...f, avg, corr }
    }).sort((a,b) => Math.abs(b.corr) - Math.abs(a.corr))
  }, [allTxns])


  const topDisagreements = useMemo(() => {
    return [...allTxns]
      .filter(t => t.lgbScore !== undefined && t.isoScore !== undefined)
      .map(t => ({
        ...t,
        delta: Math.abs((t.lgbScore || 0) - (t.isoScore || 0))
      }))
      .sort((a, b) => b.delta - a.delta)
      .slice(0, 10)
  }, [allTxns])

  const MetricRangeBar = ({ value, label, reverseRange = false }) => {
     let color = 'bg-white/20'
     if (!reverseRange) {
         if (value >= 0.8) color = 'bg-emerald-400'
         else if (value >= 0.5) color = 'bg-amber-400'
         else color = 'bg-red-400'
     } else {
         if (value <= 0.1) color = 'bg-emerald-400'
         else if (value <= 0.3) color = 'bg-amber-400'
         else color = 'bg-red-400'
     }
     
     return (
        <div className="flex flex-col gap-1 w-full">
            <div className="flex justify-between items-end">
                <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</span>
                <span className="font-mono text-xs font-bold">{(value * 100).toFixed(1)}%</span>
            </div>
            <div className="h-1 bg-white/5 w-full rounded-full overflow-hidden">
                <div className={clsx("h-full rounded-full", color)} style={{ width: `${Math.min(value * 100, 100)}%` }} />
            </div>
        </div>
     )
  }

  return (
    <div className="max-w-[1400px] mx-auto space-y-4 font-mono pb-8">
      {/* PAGE HEADER */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-wider text-foreground">MODEL INSIGHTS</h1>
          <p className="text-xs text-muted-foreground tracking-widest uppercase mt-1">
            Ensemble Explainability · LightGBM + IsoForest + Behavioral Engine
          </p>
        </div>
        <div className="flex gap-2">
          {setIsRunning && (
             <ActionBtn icon={isRunning ? Pause : Play} label={isRunning ? "PAUSE SIMULATION" : "RESUME SIM"} color={isRunning ? "yellow" : "green"} onClick={() => setIsRunning(!isRunning)} />
          )}
          {triggerAttackBurst && (
             <ActionBtn icon={Flame} label="ATTACK BURST" color="red" onClick={triggerAttackBurst} />
          )}
          {resetParams && (
             <ActionBtn icon={RefreshCw} label="RESET DEFAULTS" onClick={resetParams} />
          )}
        </div>
      </div>

      {/* Top Stats Bar */}
      <div className="grid grid-cols-4 gap-3">
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#00e5ff' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">LGB Avg Score</p>
          <p className="text-2xl font-bold mt-1" style={{ color: '#00e5ff' }}>
            {(topStats.lgbAvg * 100).toFixed(1)}%
          </p>
          <p className="text-[9px] text-muted-foreground mt-1 tracking-wider">avg fraud probability from LightGBM</p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#a855f7' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">IsoForest Anomaly Rate</p>
          <p className="text-2xl font-bold mt-1" style={{ color: '#a855f7' }}>
            {(topStats.isoRate * 100).toFixed(1)}%
          </p>
          <p className="text-[9px] text-muted-foreground mt-1 tracking-wider">% of txns where unsupervised {'>'} 0.5</p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#f59e0b' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Behavioral Rule Hit Rate</p>
          <p className="text-2xl font-bold mt-1" style={{ color: '#f59e0b' }}>
            {(topStats.behRate * 100).toFixed(1)}%
          </p>
          <p className="text-[9px] text-muted-foreground mt-1 tracking-wider">% where at least one rule fired</p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#ef4444' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Model Disagreement Rate</p>
          <p className="text-2xl font-bold mt-1" style={{ color: '#ef4444' }}>
            {(topStats.disRate * 100).toFixed(1)}%
          </p>
          <p className="text-[9px] text-muted-foreground mt-1 tracking-wider">% where |LGB - ISO| {'>'} 0.3</p>
        </Panel>
      </div>

      <div className="grid grid-cols-12 gap-4 items-start">
        {/* LEFT PANEL: Ensemble Controls */}
        <div className="col-span-3 space-y-4">
          <Panel title="Ensemble Controls">
            <div className="space-y-6 pt-2">
              
              {/* Section 1: Live Weight Tuner */}
              <div className="space-y-4">
                 <h3 className="text-[10px] text-muted-foreground uppercase tracking-widest mb-1 border-b border-border pb-1">Live Weight Tuner</h3>
                 <div className="space-y-3">
                   <div className="space-y-1 mt-2 mb-2">
                     <p className="text-[9px] text-muted-foreground leading-tight">Drag sliders to simulate rebalancing the ensemble.</p>
                   </div>
                   <div className="space-y-1">
                     <div className="flex justify-between items-center text-xs">
                       <span className="font-bold uppercase tracking-widest" style={{ color: '#00e5ff' }}>LightGBM</span>
                       <span className="px-1 py-0.5 rounded text-[10px]" style={{ color: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.1)' }}>{(weights.lgb).toFixed(2)}</span>
                     </div>
                     <input type="range" min="0" max="100" value={weights.lgb * 100} onChange={e => handleWeightChange('lgb', Number(e.target.value))} className="w-full" style={{ accentColor: '#00e5ff' }} />
                   </div>
                   <div className="space-y-1">
                     <div className="flex justify-between items-center text-xs">
                       <span className="font-bold uppercase tracking-widest" style={{ color: '#a855f7' }}>IsoForest</span>
                       <span className="px-1 py-0.5 rounded text-[10px]" style={{ color: '#a855f7', backgroundColor: 'rgba(168,85,247,0.1)' }}>{(weights.iso).toFixed(2)}</span>
                     </div>
                     <input type="range" min="0" max="100" value={weights.iso * 100} onChange={e => handleWeightChange('iso', Number(e.target.value))} className="w-full" style={{ accentColor: '#a855f7' }} />
                   </div>
                   <div className="space-y-1">
                     <div className="flex justify-between items-center text-xs">
                       <span className="font-bold uppercase tracking-widest" style={{ color: '#f59e0b' }}>Behavioral</span>
                       <span className="px-1 py-0.5 rounded text-[10px]" style={{ color: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)' }}>{(weights.beh).toFixed(2)}</span>
                     </div>
                     <input type="range" min="0" max="100" value={weights.beh * 100} onChange={e => handleWeightChange('beh', Number(e.target.value))} className="w-full" style={{ accentColor: '#f59e0b' }} />
                   </div>
                 </div>

                 {/* Live preview */}
                 <div className="bg-bg-200 border border-border p-3 rounded-xl space-y-2">
                    <div className="flex justify-between items-center bg-white/[0.02] border border-white/[0.05] p-2 rounded-lg">
                      <span className="text-[10px] tracking-wider text-[#10b981]">APPROVE</span>
                      <div className="flex gap-2 items-center">
                         <span className="font-bold text-[#10b981]">{sim.approve}</span>
                         <span className="text-[9px] text-[#10b981]/50 w-8 text-right">({sim.approvePct.toFixed(1)}%)</span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center bg-white/[0.02] border border-white/[0.05] p-2 rounded-lg">
                      <span className="text-[10px] tracking-wider text-amber-500">FLAG</span>
                      <div className="flex gap-2 items-center">
                         <span className="font-bold text-amber-500">{sim.flag}</span>
                         <span className="text-[9px] text-amber-500/50 w-8 text-right">({sim.flagPct.toFixed(1)}%)</span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center bg-white/[0.02] border border-white/[0.05] p-2 rounded-lg">
                      <span className="text-[10px] tracking-wider text-red-500">BLOCK</span>
                      <div className="flex gap-2 items-center">
                         <span className="font-bold text-red-500">{sim.block}</span>
                         <span className="text-[9px] text-red-500/50 w-8 text-right">({sim.blockPct.toFixed(1)}%)</span>
                      </div>
                    </div>
                 </div>
              </div>

              {/* Section 2: Threshold Sensitivity */}
              <div className="space-y-4 pt-4 border-t border-border">
                <h3 className="text-[10px] text-muted-foreground uppercase tracking-widest mb-1 border-b border-border pb-1">Threshold Sensitivity</h3>
                <div className="space-y-2 mt-2">
                   <p className="text-[9px] text-muted-foreground leading-tight">Lower approve threshold = stricter = higher recall, more false positives</p>
                </div>
                
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="tracking-wider text-[#10b981]">Approve Thresh</span>
                      <span className="text-[#10b981]">{thresholds.approve}</span>
                    </div>
                    <input type="range" min="0" max={thresholds.flag - 1} value={thresholds.approve} onChange={e => setThresholds(prev => ({...prev, approve: Number(e.target.value)}))} className="w-full" style={{ accentColor: '#10b981' }} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="tracking-wider text-amber-500">Flag Thresh</span>
                      <span className="text-amber-500">{thresholds.flag}</span>
                    </div>
                    <input type="range" min={thresholds.approve + 1} max="100" value={thresholds.flag} onChange={e => setThresholds(prev => ({...prev, flag: Number(e.target.value)}))} className="w-full" style={{ accentColor: '#f59e0b' }} />
                  </div>
                </div>

                <div className="bg-bg-200 border border-border p-3 rounded-xl grid grid-cols-2 gap-3 mt-4">
                   <MetricRangeBar label="Precision" value={sim.precision} />
                   <MetricRangeBar label="Recall" value={sim.recall} />
                   <MetricRangeBar label="F-Pos Rate" value={sim.fpr} reverseRange />
                   <MetricRangeBar label="F1 Score" value={sim.f1} />
                </div>
              </div>

            </div>
          </Panel>
        </div>

        {/* RIGHT PANEL: 4 Sections */}
        <div className="col-span-9 space-y-4">
          
          {/* Section 1: Model Agreement Matrix */}
          <Panel title="Model Agreement Matrix (Top Models)">
             <div className="grid grid-cols-2 gap-4 mb-3">
                <div className="rounded-xl border border-[#10b981]/30 bg-[#10b981]/5 p-4 flex flex-col justify-center items-center relative overflow-hidden transition-all">
                   <ShieldCheck className="absolute -left-2 -bottom-2 w-20 h-20 text-[#10b981]/5" strokeWidth={1} />
                   <p className="text-[11px] uppercase tracking-widest mb-1 z-10 text-emerald-400 font-bold">Both Agree Safe</p>
                   <p className="text-3xl font-bold text-emerald-400 z-10">{matrixData.bothSafe}</p>
                   <p className="text-[10px] mt-1 z-10 px-2 py-0.5 rounded bg-emerald-400/10 text-emerald-400">
                      {matrixData.pctBothSafe.toFixed(1)}%
                   </p>
                </div>
                
                <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 flex flex-col justify-center items-center relative overflow-hidden shadow-[0_0_15px_rgba(245,158,11,0.15)] animate-pulse-slow">
                   <AlertTriangle className="absolute -right-2 -bottom-2 w-20 h-20 text-amber-500/10" strokeWidth={1} />
                   <p className="text-[11px] uppercase tracking-widest text-[#f59e0b] mb-1 z-10 font-bold text-center">
                     ⚠️ ISO flags, LGB misses
                   </p>
                   <p className="text-3xl font-bold text-[#f59e0b] z-10">{matrixData.isoFlags}</p>
                   <p className="text-[10px] mt-1 z-10 px-2 py-0.5 rounded bg-[#f59e0b]/10 text-amber-500">
                      {matrixData.pctIso.toFixed(1)}%
                   </p>
                </div>

                <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 flex flex-col justify-center items-center relative overflow-hidden shadow-[0_0_15px_rgba(245,158,11,0.15)] animate-pulse-slow">
                   <AlertTriangle className="absolute -left-2 -bottom-2 w-20 h-20 text-amber-500/10" strokeWidth={1} />
                   <p className="text-[11px] uppercase tracking-widest text-[#f59e0b] mb-1 z-10 font-bold text-center">
                     ⚠️ LGB flags, ISO misses
                   </p>
                   <p className="text-3xl font-bold text-[#f59e0b] z-10">{matrixData.lgbFlags}</p>
                   <p className="text-[10px] mt-1 z-10 px-2 py-0.5 rounded bg-[#f59e0b]/10 text-amber-500">
                      {matrixData.pctLgb.toFixed(1)}%
                   </p>
                </div>

                <div className="rounded-xl border py-4 px-2 border-[#ef4444]/30 bg-[#ef4444]/10 flex flex-col justify-center items-center relative overflow-hidden">
                   <p className="text-[11px] uppercase tracking-widest mb-1 z-10 font-bold text-red-500">🚨 Both Agree Fraud</p>
                   <p className="text-3xl font-bold text-[#ef4444] z-10">{matrixData.bothFraud}</p>
                   <p className="text-[10px] mt-1 z-10 px-2 py-0.5 rounded bg-red-500/10 text-red-500">
                      {matrixData.pctBothFraud.toFixed(1)}%
                   </p>
                </div>
             </div>
             
             <div className="text-center pt-2">
                <span className="text-xs uppercase tracking-widest text-amber-500 font-bold px-4 py-1.5 rounded-full border border-amber-500/30 bg-amber-500/5">
                   {matrixData.disagree} transactions need human review (models disagree)
                </span>
             </div>
          </Panel>

          <div className="grid grid-cols-2 gap-4">
             {/* Section 2: Behavioral Rule Breakdown */}
             <Panel title="RULE ENGINE BREAKDOWN">
               <div className="space-y-4 pt-2">
                 {behData.map(r => (
                   <div key={r.id} className="relative">
                     <div className="flex justify-between items-center mb-1">
                        <div className="flex items-center gap-2">
                           <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded" style={{ backgroundColor: `${r.color}20`, color: r.color, border: `1px solid ${r.color}40` }}>
                             {r.weight}
                           </span>
                           <span className="text-xs font-bold leading-none">{r.name}</span>
                        </div>
                        <span className="text-[10px] font-bold text-muted-foreground">{r.pct.toFixed(1)}% triggered</span>
                     </div>
                     <p className="text-[9px] text-muted-foreground mb-1.5 ml-1 pt-0.5 italic">{r.features}</p>
                     
                     <div className="h-1.5 w-full bg-bg-200 rounded-full overflow-hidden mb-2">
                        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${r.pct}%`, backgroundColor: r.color }} />
                     </div>

                     <div className="flex items-center justify-between text-[9px] uppercase tracking-wider pl-1 pr-1 bg-white/[0.02] border border-white/[0.05] py-1 rounded">
                        <span className="text-foreground/70">Risk when triggered: <span className="text-red-400 font-bold">{r.avgRiskTrig.toFixed(1)}%</span></span>
                        <span className="text-foreground/70">When not: <span className="text-emerald-400 font-bold">{r.avgRiskNotTrig.toFixed(1)}%</span></span>
                     </div>
                   </div>
                 ))}
               </div>
             </Panel>

             {/* Section 3: Feature Correlation Table */}
             <Panel title="FEATURE → RISK CORRELATION" className="flex flex-col h-full overflow-hidden">
                <div className="flex-1 overflow-y-auto custom-scrollbar -mr-2 pr-2">
                  <table className="w-full text-left border-collapse">
                    <thead className="sticky top-0 bg-bg-100 z-10 border-b border-border shadow-sm">
                      <tr>
                        <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground">Feature</th>
                        <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-right w-16">Avg Val</th>
                        <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-center w-28">Corr with Risk</th>
                        <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-right">Rule</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {featuresList.map(f => (
                        <tr key={f.key} className="hover:bg-bg-200/50 transition-colors">
                          <td className="py-2.5 text-[10px] text-cyan-50/80 max-w-[120px] truncate pr-2">{f.label}</td>
                          <td className="py-2.5 text-[10px] text-right text-foreground w-16">
                            {f.isBool ? `${(f.avg*100).toFixed(0)}%` : f.avg > 1000 ? (f.avg/1000).toFixed(1)+'k' : f.avg.toFixed(1)}
                          </td>
                          <td className="py-2.5 align-middle w-28 pr-2 pl-2">
                             <div className="flex items-center justify-center w-full">
                                <div className="w-full h-1.5 bg-bg-200 rounded-full flex relative">
                                   {/* Center 0 marker */}
                                   <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/30 z-10"/>
                                   {f.corr > 0 
                                      ? <div className="absolute top-0 bottom-0 left-1/2 bg-red-400 rounded-r-full" style={{ width: `${Math.min(f.corr*100, 50)}%` }}/>
                                      : <div className="absolute top-0 bottom-0 right-1/2 bg-emerald-400 rounded-l-full" style={{ width: `${Math.min(Math.abs(f.corr)*100, 50)}%` }}/>
                                   }
                                </div>
                             </div>
                             <div className="text-[8px] text-center mt-1 text-muted-foreground tracking-wider">{f.corr > 0 ? '+' : ''}{f.corr.toFixed(2)}</div>
                          </td>
                          <td className="py-2.5 text-[9px] uppercase text-right pr-2">
                             <span className="px-1.5 py-0.5 rounded border leading-none inline-block mt-0.5" style={{ color: f.color, backgroundColor: `${f.color}10`, borderColor: `${f.color}30` }}>
                               {f.rule}
                             </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
             </Panel>
          </div>
        </div>
      </div>

      {/* Section 4: Top Model Disagreement Cases */}
      <Panel title="MODEL DISAGREEMENT CASES — where LightGBM and IsoForest conflict most">
        <div className="overflow-x-auto min-h-[300px]">
          <table className="w-full text-left border-collapse">
            <thead className="border-b border-border">
              <tr>
                <th className="py-2 pl-4 text-[10px] uppercase tracking-widest text-muted-foreground w-28">TX ID</th>
                <th className="py-2 text-[10px] uppercase tracking-widest text-muted-foreground text-right w-24">Amount</th>
                <th className="py-2 text-[10px] uppercase tracking-widest text-right w-16" style={{ color: '#00e5ff' }}>LGB</th>
                <th className="py-2 text-[10px] uppercase tracking-widest text-right w-16" style={{ color: '#a855f7' }}>ISO</th>
                <th className="py-2 text-[10px] uppercase tracking-widest text-right w-16" style={{ color: '#f59e0b' }}>BEH</th>
                <th className="py-2 text-[10px] uppercase tracking-widest text-right pr-4 w-24">Final</th>
                <th className="py-2 text-[10px] uppercase tracking-widest font-bold text-center w-20">Delta</th>
                <th className="py-2 text-[10px] uppercase tracking-widest text-muted-foreground">Reason</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {topDisagreements.map(t => {
                const idRaw = t.transactionId || t.id || 'Unknown'
                const isHighDelta = t.delta > 0.5
                const deltaColor = isHighDelta ? 'text-red-500 font-bold' : (t.delta > 0.3 ? 'text-amber-500' : 'text-emerald-400')

                return (
                  <tr
                    key={idRaw}
                    onClick={() => handleSelectTxn(t)}
                    className={clsx(
                      "hover:bg-bg-200 transition-colors cursor-pointer relative",
                      isHighDelta && "border-l-2 border-l-red-500/50 hover:border-l-red-500",
                      !isHighDelta && "border-l-2 border-l-transparent"
                    )}
                  >
                    <td className="py-3 pl-4 text-[11px] text-text-primary">
                      <span className="font-mono bg-white/5 px-1 rounded">{idRaw.slice(0, 8)}...</span>
                    </td>
                    <td className="py-3 text-[11px] text-right text-text-primary">
                      {formatCurrency(t.amount || 0).replace('MYR', 'RM')}
                    </td>
                    <td className="py-3 text-[11px] text-right" style={{ color: '#00e5ff' }}>
                      {((t.lgbScore || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 text-[11px] text-right" style={{ color: '#a855f7' }}>
                      {((t.isoScore || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 text-[11px] text-right" style={{ color: '#f59e0b' }}>
                      {((t.behScore || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 text-[10px] text-right pr-4">
                      <span className={clsx("px-2 py-0.5 rounded font-bold uppercase",
                        t.decision === 'BLOCK' ? 'text-red-500 bg-red-500/10' :
                          t.decision === 'FLAG' ? 'text-amber-500 bg-amber-500/10' :
                            'text-emerald-400 bg-emerald-400/10'
                      )}>
                        {t.decision}
                      </span>
                    </td>
                    <td className={clsx("py-3 text-[11px] text-center", deltaColor)}>
                      ±{(t.delta * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 text-[9px] text-muted-foreground pr-2 truncate max-w-[200px]">
                      {t.reasons?.[0]?.slice(0, 40) || 'No explicit reason recorded'}...
                    </td>
                  </tr>
                )
              })}
              {topDisagreements.length === 0 && (
                <tr><td colSpan={8} className="text-center py-8 text-xs text-muted-foreground w-full">Waiting for transactions...</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Transaction Details Inspector Modal Overlay */}
      {currentTxn && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-end sm:items-center justify-center p-4">
           <div className="absolute inset-0" onClick={() => handleSelectTxn(null)} />
           <div className="relative z-10 w-full sm:w-[460px] max-h-[90vh] flex flex-col shadow-2xl animate-in fade-in zoom-in-95 duration-200">
             <div className="absolute top-2 right-2 z-20">
                <button 
                  onClick={() => handleSelectTxn(null)}
                  className="p-1.5 bg-bg-200/80 hover:bg-bg-200 rounded-full text-text-muted hover:text-white transition-colors"
                >
                  <X size={16} />
                </button>
             </div>
             <div className="flex-1 overflow-hidden rounded-2xl ring-1 ring-white/10">
               <TransactionInspector 
                  selectedTxn={currentTxn} 
                  engine={engine} 
               />
             </div>
           </div>
        </div>
      )}

    </div>
  )
}
