/**
 * Model Tuning Lab — 7-row right panel layout.
 * Reads from window.__fraudShieldStore.txnHistory (persists across tabs).
 * All re-scoring is frontend-only.
 */
import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  Zap, AlertTriangle, ShieldAlert, Check, RefreshCw,
  BarChart2, Info, TrendingUp, TrendingDown, Target,
  Settings2, Activity, Play, ShieldCheck, HelpCircle, Search, Scale, Minus, Star
} from 'lucide-react'
import clsx from 'clsx'
import { useTransactionEngine } from '../hooks/useTransactionEngine'

import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, PointElement, LineElement,
  Title, Tooltip as ChartTooltip, Legend, Filler
} from 'chart.js'
import { Bar, Line } from 'react-chartjs-2'
import annotationPlugin from 'chartjs-plugin-annotation'

ChartJS.register(
  CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Title, ChartTooltip, Legend, Filler
)
// Register annotation plugin if available
try { ChartJS.register(annotationPlugin) } catch {}

// ── Constants ────────────────────────────────────────────
const BEH_W = { drain: 0.35, deviation: 0.25, context: 0.20, velocity: 0.20 }
const DEPLOYED_W = { lgb: 0.55, iso: 0.25, beh: 0.20 }
const DEPLOYED_T = { approve: 35, flag: 60 }
const SL_COLORS = { lgb: '#00e5ff', iso: '#9c27b0', beh: '#ff9800', approve: '#00e5a0', flag: '#ff9800' }
const CARD = { bg: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }
const FONT = { family: 'JetBrains Mono, monospace', size: 9 }
const GRID_C = 'rgba(255,255,255,0.07)'
const TICK_C = 'rgba(255,255,255,0.35)'

// ── Re-scoring ───────────────────────────────────────────
function rescore(txn, w, t) {
  const lgb = (txn.supervised_score || 0) * 100
  const iso = (txn.unsupervised_score || 0) * 100
  const rb = txn.rule_breakdown || {}
  const beh = (
    ((rb.drain_score || 0) / BEH_W.drain) * BEH_W.drain +
    ((rb.deviation_score || 0) / BEH_W.deviation) * BEH_W.deviation +
    ((rb.context_score || 0) / BEH_W.context) * BEH_W.context +
    ((rb.velocity_score || 0) / BEH_W.velocity) * BEH_W.velocity
  ) * 100
  const score = Math.min(100, Math.max(0, lgb * w.lgb + iso * w.iso + beh * w.beh))
  const dec = score < t.approve ? 'LOW' : score < t.flag ? 'MEDIUM' : 'HIGH'
  return { score, dec }
}

function calcMetrics(hist, w, t) {
  let TP = 0, FP = 0, TN = 0, FN = 0
  hist.forEach(txn => {
    const { dec } = rescore(txn, w, t)
    const pF = dec === 'HIGH', aF = txn.ground_truth === 'FRAUD'
    if (pF && aF) TP++; else if (pF && !aF) FP++
    else if (!pF && !aF) TN++; else FN++
  })
  const p = TP + FP > 0 ? TP / (TP + FP) : 0
  const r = TP + FN > 0 ? TP / (TP + FN) : 0
  const f = p + r > 0 ? 2 * p * r / (p + r) : 0
  const fp = FP + TN > 0 ? FP / (FP + TN) : 0
  return { p, r, f, fp, TP, FP, TN, FN,
    approve: hist.filter(x => rescore(x, w, t).dec === 'LOW').length,
    flag: hist.filter(x => rescore(x, w, t).dec === 'MEDIUM').length,
    block: hist.filter(x => rescore(x, w, t).dec === 'HIGH').length,
  }
}

// ── Convergence-based empirical tuning ───────────────────
// Approve Balance: where P, R, F1 converge (minimum spread), penalised by FPR
function findApproveBalance(curve) {
  const lower = curve.filter(p => p.t <= 55)
  if (lower.length === 0) return curve[0]
  return lower.reduce((best, pt) => {
    const vals = [pt.p, pt.r, pt.f]
    const spread = Math.max(...vals) - Math.min(...vals)
    const score = (1 - spread) * (1 - pt.fp)
    return score > best.score ? { ...pt, spread, score } : best
  }, { ...lower[0], score: -1 })
}

// Block Balance: where precision stabilises at max, high scores
function findBlockBalance(curve) {
  const upper = curve.filter(p => p.t >= 45)
  if (upper.length === 0) return curve[curve.length - 1]
  return upper.reduce((best, pt) => {
    const score = pt.p * (1 - pt.fp)
    return score > best.score ? { ...pt, score } : best
  }, { ...upper[0], score: -1 })
}

/* ══════ MAIN ══════════════════════════════════════════════ */
export default function ModelTuningLab() {
  const engine = useTransactionEngine()
  const [buffer, setBuffer] = useState(() => [...(window.__fraudShieldStore?.txnHistory || [])])
  const [weights, setWeights] = useState({ ...DEPLOYED_W })
  const [thresholds, setThresholds] = useState({ ...DEPLOYED_T })
  const [applyState, setApplyState] = useState('idle')
  const [appliedAt, setAppliedAt] = useState(null)
  const [toast, setToast] = useState(null)
  const [balancePoints, setBalancePoints] = useState(null) // { approve: {...}, block: {...} }

  // Subscribe to new txns
  useEffect(() => {
    const h = () => setBuffer([...(window.__fraudShieldStore?.txnHistory || [])])
    window.addEventListener('fraudshield:newtxn', h)
    return () => window.removeEventListener('fraudshield:newtxn', h)
  }, [])

  // Weight auto-balance
  const setWeight = useCallback((key, val) => {
    setWeights(prev => {
      const others = ['lgb', 'iso', 'beh'].filter(k => k !== key)
      const rem = Math.max(0, 1 - val)
      const oSum = prev[others[0]] + prev[others[1]]
      const nw = { ...prev, [key]: val }
      if (oSum === 0) { nw[others[0]] = rem / 2; nw[others[1]] = rem / 2 }
      else { nw[others[0]] = (prev[others[0]] / oSum) * rem; nw[others[1]] = (prev[others[1]] / oSum) * rem }
      return nw
    })
  }, [])

  const wSum = weights.lgb + weights.iso + weights.beh
  const wSumOk = Math.abs(wSum - 1.0) < 0.005

  // Metrics
  const tunedM = useMemo(() => calcMetrics(buffer, weights, thresholds), [buffer, weights, thresholds])
  const deployedM = useMemo(() => calcMetrics(buffer, DEPLOYED_W, DEPLOYED_T), [buffer])

  const diffs = useMemo(() => {
    let total = 0, aToB = 0, bToA = 0
    buffer.forEach(txn => {
      const d = rescore(txn, DEPLOYED_W, DEPLOYED_T).dec
      const t = rescore(txn, weights, thresholds).dec
      if (d !== t) { total++; if (d === 'LOW' && t === 'HIGH') aToB++; if (d === 'HIGH' && t === 'LOW') bToA++ }
    })
    return { total, aToB, bToA }
  }, [buffer, weights, thresholds])

  const hasLabels = buffer.filter(t => t.ground_truth).length >= 10

  // Threshold vs Metrics sweep
  const threshSweep = useMemo(() => {
    if (!hasLabels) return null
    const pts = []
    for (let T = 0; T <= 100; T += 2) {
      const m = calcMetrics(buffer, weights, { approve: T, flag: T })
      pts.push({ t: T, p: m.p, r: m.r, f: m.f, fp: m.fp })
    }
    return pts
  }, [buffer, weights, hasLabels])

  const balanceResult = useMemo(() => {
    if (!threshSweep) return null
    return { approve: findApproveBalance(threshSweep), block: findBlockBalance(threshSweep) }
  }, [threshSweep])

  // Score distribution bins
  const scoreDist = useMemo(() => {
    const bins = Array.from({ length: 20 }, () => ({ legit: 0, fraud: 0 }))
    buffer.forEach(txn => {
      const s = rescore(txn, weights, thresholds).score
      const idx = Math.min(19, Math.floor(s / 5))
      if (txn.ground_truth === 'FRAUD') bins[idx].fraud++
      else bins[idx].legit++
    })
    return bins
  }, [buffer, weights, thresholds])

  // PR Curve
  const prPoints = useMemo(() => {
    if (buffer.length < 10) return null
    const pts = []
    for (let T = 0; T <= 100; T += 2) {
      let TP = 0, FP = 0, FN = 0
      buffer.forEach(txn => {
        const s = rescore(txn, weights, thresholds).score
        const pred = s >= T, actual = txn.ground_truth === 'FRAUD'
        if (pred && actual) TP++; if (pred && !actual) FP++; if (!pred && actual) FN++
      })
      const p = TP + FP > 0 ? TP / (TP + FP) : 1
      const r = TP + FN > 0 ? TP / (TP + FN) : 0
      pts.push({ x: r, y: p, t: T })
    }
    return pts.sort((a, b) => a.x - b.x)
  }, [buffer, weights, thresholds])

  const aucPR = useMemo(() => {
    if (!prPoints) return '—'
    let a = 0
    for (let i = 1; i < prPoints.length; i++)
      a += Math.abs(prPoints[i].x - prPoints[i - 1].x) * (prPoints[i].y + prPoints[i - 1].y) / 2
    return a.toFixed(3)
  }, [prPoints])

  // Business impact
  const impact = useMemo(() => {
    let fraudCaught = 0, fpFriction = 0, tpCount = 0, fnCount = 0
    buffer.forEach(txn => {
      const { dec } = rescore(txn, weights, thresholds)
      const isF = txn.ground_truth === 'FRAUD'
      if (dec === 'HIGH' && isF) { fraudCaught += (txn.amount || 0); tpCount++ }
      if (dec === 'HIGH' && !isF) fpFriction += (txn.amount || 0)
      if (dec !== 'HIGH' && isF) fnCount++
    })
    const protRate = tpCount + fnCount > 0 ? (tpCount / (tpCount + fnCount)) * 100 : 0
    return { fraudCaught, fpFriction, net: fraudCaught - fpFriction, protRate, fpCount: tunedM.FP }
  }, [buffer, weights, thresholds, tunedM])

  // Actions
  const handleApply = () => {
    window.__fraudShieldStore.activeWeights = { ...weights }
    window.__fraudShieldStore.activeThresholds = { ...thresholds }
    window.__activeTuning = { weights: { ...weights }, thresholds: { ...thresholds } }
    setApplyState('active')
    setAppliedAt(new Date().toLocaleTimeString())
    setToast('✅ Config applied to live stream')
    setTimeout(() => setToast(null), 2500)
  }
  const handleReset = () => {
    setWeights({ ...DEPLOYED_W })
    setThresholds({ ...DEPLOYED_T })
    window.__fraudShieldStore.activeWeights = { ...DEPLOYED_W }
    window.__fraudShieldStore.activeThresholds = { ...DEPLOYED_T }
    window.__activeTuning = null
    setApplyState('idle')
  }
  const handleEmpiricalTune = () => {
    if (balanceResult) {
      const aT = balanceResult.approve.t
      const bT = balanceResult.block.t
      setThresholds({ approve: aT, flag: bT })
      setBalancePoints(balanceResult)
      setToast(`🎯 Empirically tuned: Approve T=${aT} (spread ${(balanceResult.approve.spread*100).toFixed(0)}%) · Block T=${bT}`)
      setTimeout(() => setToast(null), 4000)
    }
  }

  const fmtRM = v => `RM ${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`

  /* ═══════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════ */
  return (
    <div className="flex flex-col h-full relative">
      {/* Toast */}
      {toast && (
        <div className="absolute top-4 right-8 z-[100] px-4 py-2.5 rounded-lg bg-[rgba(0,229,160,0.12)] border border-[rgba(0,229,160,0.3)] text-[#00e5a0] font-mono text-xs font-bold animate-fade-in shadow-lg backdrop-blur-sm">
          {toast}
        </div>
      )}

      {/* HEADER */}
      <div className="shrink-0 flex items-center justify-between px-8 py-4 border-b border-border">
        <div>
          <div className="flex items-center gap-3">
            <span className="text-xl">⚗</span>
            <h1 className="font-mono font-bold text-lg text-text-primary uppercase tracking-tight">Model Tuning Lab</h1>
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-[rgba(16,185,129,0.1)] border border-[rgba(16,185,129,0.25)]">
              <span className="w-1.5 h-1.5 rounded-full bg-[#10b981] animate-pulse-slow" />
              <span className="text-[9px] font-mono font-bold text-[#10b981] uppercase tracking-wider">LIVE</span>
            </span>
          </div>
          <p className="font-mono text-[11px] text-text-muted mt-0.5 pl-[36px]">Tune weights &amp; thresholds · Re-scoring is instant, frontend only</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <span className="section-label block">Buffer</span>
            <span className="font-mono font-bold text-[20px] text-cyan-400 leading-tight">{buffer.length}</span>
            <span className="font-mono text-[9px] text-text-muted block">txns</span>
          </div>
          <button onClick={() => engine.triggerAttackBurst()} className="flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-mono font-bold text-[#ef4444] bg-[rgba(239,68,68,0.08)] border border-[rgba(239,68,68,0.25)] hover:bg-[rgba(239,68,68,0.15)] transition-colors uppercase tracking-wide">
            <Zap size={13} /> Attack Burst
          </button>
        </div>
      </div>

      {/* BODY */}
      <div className="flex-1 flex overflow-hidden min-h-0">

        {/* ═══ LEFT PANEL (260px, sticky) ═══ */}
        <div className="w-[260px] shrink-0 border-r border-border bg-bg-50 overflow-y-auto py-5 px-4 space-y-4" style={{ position: 'sticky', top: 0 }}>

          {/* Ensemble Weights */}
          <section>
            <h3 className="section-label mb-2.5">Ensemble Weights</h3>
            <div className="space-y-3.5">
              <Slider label="LightGBM" cls="slider-lgb" color={SL_COLORS.lgb} ico={<Activity size={11}/>} value={weights.lgb} onChange={v => setWeight('lgb', v)} />
              <Slider label="IsoForest" cls="slider-iso" color={SL_COLORS.iso} ico={<BarChart2 size={11}/>} value={weights.iso} onChange={v => setWeight('iso', v)} />
              <Slider label="Behavioral" cls="slider-beh" color={SL_COLORS.beh} ico={<Settings2 size={11}/>} value={weights.beh} onChange={v => setWeight('beh', v)} />
            </div>
            <div className="mt-3 flex items-center justify-between px-2 py-2 rounded-lg" style={{ background: CARD.bg, border: CARD.border }}>
              <span className="section-label">Total</span>
              <div className="flex items-center gap-1.5">
                <span className={clsx("font-mono text-[12px] font-bold", wSumOk ? "text-[#00e5a0]" : "text-[#ff4d6d]")}>{wSum.toFixed(2)}</span>
                {wSumOk ? <Check size={12} className="text-[#00e5a0]" /> : <ShieldAlert size={12} className="text-[#ff4d6d]" />}
              </div>
            </div>
          </section>

          <hr className="border-[rgba(255,255,255,0.06)]" />

          {/* Decision Thresholds */}
          <section>
            <h3 className="section-label mb-2.5">Decision Thresholds</h3>
            <div className="space-y-3.5">
              <Slider label="Approve" cls="slider-approve" color={SL_COLORS.approve} value={thresholds.approve} min={10} max={90} step={1} fmt={v => v}
                onChange={v => { if (v < thresholds.flag) setThresholds(p => ({ ...p, approve: v })) }} />
              <Slider label="Flag" cls="slider-flag" color={SL_COLORS.flag} value={thresholds.flag} min={10} max={90} step={1} fmt={v => v}
                onChange={v => { if (v > thresholds.approve) setThresholds(p => ({ ...p, flag: v })) }} />
            </div>
            {/* Zone bar */}
            <div className="mt-3">
              <div className="flex h-[5px] w-full rounded-full overflow-hidden">
                <div style={{ width: `${thresholds.approve}%`, background: '#00e5a0', opacity: 0.5 }} />
                <div style={{ width: `${thresholds.flag - thresholds.approve}%`, background: '#ff9800', opacity: 0.5 }} />
                <div style={{ width: `${100 - thresholds.flag}%`, background: '#ff4d6d', opacity: 0.5 }} />
              </div>
              <div className="flex justify-between mt-1 font-mono text-[7px] text-text-muted uppercase tracking-widest">
                <span>Approve</span><span>Flag</span><span>Block</span>
              </div>
              <div className="relative h-3 font-mono text-[7px] text-text-muted">
                <span className="absolute left-0">0</span>
                <span className="absolute" style={{ left: `${thresholds.approve}%`, transform: 'translateX(-50%)' }}>{thresholds.approve}</span>
                <span className="absolute" style={{ left: `${thresholds.flag}%`, transform: 'translateX(-50%)' }}>{thresholds.flag}</span>
                <span className="absolute right-0">100</span>
              </div>
            </div>
          </section>

          <hr className="border-[rgba(255,255,255,0.06)]" />

          {/* Empirical Tuning */}
          {hasLabels && (
            <section>
              <h3 className="section-label mb-2">Empirical Tuning</h3>
              <div className="rounded-lg p-3 space-y-2.5" style={{ background: CARD.bg, border: CARD.border }}>
                <p className="font-mono text-[8px] text-text-muted leading-relaxed">
                  Finds where Precision, Recall &amp; F1 converge (minimum spread) for the Approve threshold, and where Precision stabilises for the Block threshold.
                </p>
                {balanceResult && (
                  <div className="space-y-1.5 pt-1.5 border-t border-[rgba(255,255,255,0.06)]">
                    <div className="flex justify-between font-mono text-[9px]">
                      <span className="text-[#00e5a0]">Approve Balance</span>
                      <span className="text-text-primary font-bold">T={balanceResult.approve.t}</span>
                    </div>
                    <div className="flex justify-between font-mono text-[9px]">
                      <span className="text-[#ff4d6d]">Block Balance</span>
                      <span className="text-text-primary font-bold">T={balanceResult.block.t}</span>
                    </div>
                    <div className="flex justify-between font-mono text-[9px]">
                      <span className="text-[#ff9800]">Flag Zone</span>
                      <span className="text-text-muted">{balanceResult.approve.t} – {balanceResult.block.t}</span>
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}

          <hr className="border-[rgba(255,255,255,0.06)]" />

          {/* Actions */}
          <section className="space-y-2">
            {hasLabels && balanceResult && (
              <button onClick={handleEmpiricalTune}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg font-mono text-[10px] font-bold uppercase tracking-wider border border-[rgba(255,215,0,0.3)] text-[#ffd700] bg-[rgba(255,215,0,0.06)] hover:bg-[rgba(255,215,0,0.12)] transition-colors">
                <Target size={12} /> Empirically Tune
              </button>
            )}
            <button onClick={handleApply} disabled={!wSumOk}
              className={clsx(
                "w-full flex items-center justify-center gap-2 py-2.5 rounded-lg font-mono text-[10px] font-bold uppercase tracking-wider transition-all duration-300 border",
                applyState === 'active'
                  ? "border-[#00e5a0] text-[#00e5a0] bg-[rgba(0,229,160,0.08)]"
                  : wSumOk
                    ? "border-[rgba(0,229,255,0.3)] text-[#00e5ff] bg-[rgba(0,229,255,0.06)] hover:bg-[rgba(0,229,255,0.12)]"
                    : "border-[rgba(255,255,255,0.08)] text-text-muted cursor-not-allowed opacity-50"
              )}>
              {applyState === 'active' ? <ShieldCheck size={13} /> : <Play size={13} />}
              {applyState === 'active' ? `● Active · ${appliedAt}` : 'Apply to Live Stream'}
            </button>
            <button onClick={handleReset}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-lg font-mono text-[10px] font-bold uppercase tracking-wider border border-[rgba(255,255,255,0.08)] text-text-muted hover:bg-[rgba(255,255,255,0.04)] transition-colors">
              <RefreshCw size={12} /> Reset to Defaults
            </button>
          </section>

          <hr className="border-[rgba(255,255,255,0.06)]" />

          {/* Delta Summary */}
          <section>
            <h3 className="section-label mb-2">Delta Summary</h3>
            <div className="rounded-lg p-3 space-y-2" style={{ background: CARD.bg, border: CARD.border }}>
              <div className="flex justify-between font-mono text-[10px]">
                <span className="text-text-muted">Reclassified</span>
                <span className="text-text-primary font-bold">{diffs.total} txns</span>
              </div>
              {diffs.total > 0 && (
                <div className="space-y-1.5 pt-2 border-t border-[rgba(255,255,255,0.06)]">
                  {diffs.aToB > 0 && <div className="flex justify-between font-mono text-[9px] text-[#ff4d6d]"><span>⚠ {diffs.aToB} APPROVE → BLOCK</span><span className="font-bold">review</span></div>}
                  {diffs.bToA > 0 && <div className="flex justify-between font-mono text-[9px] text-[#ff4d6d]"><span>⚠ {diffs.bToA} BLOCK → APPROVE</span><span className="font-bold">verify</span></div>}
                </div>
              )}
            </div>
          </section>
        </div>

        {/* ═══ RIGHT PANEL (scrollable) ═══ */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">

          {buffer.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center gap-3 opacity-40">
              <span className="text-4xl">⚗</span>
              <p className="font-mono text-[12px] text-text-muted uppercase tracking-widest">Run simulation first to populate buffer</p>
            </div>
          ) : (
            <>
              {/* ── ROW 1: Operational Cards ──── */}
              <div className="grid grid-cols-3 gap-3">
                <OpCard label="Approve" val={tunedM.approve} base={deployedM.approve} total={buffer.length} good />
                <OpCard label="Flag" val={tunedM.flag} base={deployedM.flag} total={buffer.length} />
                <OpCard label="Block" val={tunedM.block} base={deployedM.block} total={buffer.length} invertDelta />
              </div>

              {/* ── ROW 2: 2×2 Performance Matrix ── */}
              {!hasLabels ? (
                <div className="rounded-lg p-4 flex items-center justify-center gap-3" style={{ background: CARD.bg, border: CARD.border }}>
                  <HelpCircle size={15} className="text-[#ff9800]" />
                  <span className="font-mono text-[10px] text-[#ff9800] uppercase tracking-wider">Run Attack Burst to generate labeled ground truth data</span>
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <PerfCard label="Precision" icon={<Target size={26} />} val={tunedM.p} base={deployedM.p} fmt="pct"
                      tip={`Of all blocked, ${(tunedM.p * 100).toFixed(1)}% were actual fraud.\nHigh precision = fewer innocents blocked.`}
                      thr={[0.70, 0.90]} extra={`${tunedM.TP} TP / ${tunedM.TP + tunedM.FP} blocked`} />
                    <PerfCard label="Recall" icon={<Search size={26} />} val={tunedM.r} base={deployedM.r} fmt="pct"
                      tip={`Of all fraud, caught ${(tunedM.r * 100).toFixed(1)}%.\nHigh recall = fewer fraudsters escape.`}
                      thr={[0.70, 0.90]} extra={`${tunedM.TP} caught / ${tunedM.TP + tunedM.FN} total`} />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <PerfCard label="F1 Score" icon={<Scale size={26} />} val={tunedM.f} base={deployedM.f} fmt="f"
                      tip="Harmonic mean of Precision and Recall.\nBest single metric for imbalanced data."
                      thr={[0.60, 0.85]} />
                    <PerfCard label="FP Rate" icon={tunedM.fp > 0.05 ? <AlertTriangle size={26} /> : null} val={tunedM.fp} base={deployedM.fp} fmt="pct"
                      tip={`% of legit txns incorrectly blocked.\n${tunedM.FP} false positives.`}
                      thr={[0.05, 0.15]} invert extra={`${tunedM.FP} false positives`} />
                  </div>
                  <div className="font-mono text-[9px] text-text-muted px-0.5">
                    {diffs.total} reclassified · <span className="text-[#00e5a0]">{diffs.bToA} → Approve</span> · <span className="text-[#ff4d6d]">{diffs.aToB} → Block</span>
                  </div>
                </>
              )}

              {/* ── ROW 3: Threshold vs Metrics ── */}
              <ChartCard title="Threshold vs Metrics" subtitle="How each metric changes across thresholds 0-100" height="220px">
                {threshSweep ? (
                  <ThresholdChart data={threshSweep} currentT={thresholds.approve} currentFlagT={thresholds.flag} bp={balancePoints}
                    onSelect={t => { setThresholds({ approve: t, flag: Math.min(90, t + 20) }); setToast(`Threshold set to ${t}`); setTimeout(() => setToast(null), 2000) }} />
                ) : (
                  <EmptyChart msg="Need ≥ 10 labeled transactions" />
                )}
              </ChartCard>

              {/* ── ROW 4: Score Distribution ── */}
              <ChartCard title="Score Distribution" subtitle="Legit vs Fraud score distributions with decision boundary" height="180px">
                {buffer.length > 0 ? (
                  <ScoreDistChart bins={scoreDist} threshold={thresholds.approve} />
                ) : (
                  <EmptyChart msg="No data" />
                )}
              </ChartCard>

              {/* ── ROW 5: Precision-Recall Curve ── */}
              <div className="rounded-lg overflow-hidden" style={{ background: CARD.bg, border: CARD.border }}>
                <div className="flex justify-between items-center px-4 pt-4 pb-1">
                  <div>
                    <h3 className="section-label">Precision-Recall Curve</h3>
                    <p className="font-mono text-[9px] text-text-muted mt-0.5">Empirical from {buffer.length} txns · Click to set threshold</p>
                  </div>
                  {prPoints && <span className="px-2.5 py-1 rounded-md bg-[rgba(0,229,255,0.1)] border border-[rgba(0,229,255,0.25)] font-mono text-[10px] text-[#00e5ff] font-bold">AUC-PR: {aucPR}</span>}
                </div>
                <div className="px-4 pb-2" style={{ height: '280px' }}>
                  {prPoints ? (
                    <PRChart points={prPoints} dPt={{ x: deployedM.r, y: deployedM.p }} tPt={{ x: tunedM.r, y: tunedM.p }}
                      onSelect={t => { setThresholds(p => ({ approve: t, flag: Math.max(t + 5, p.flag) })); setToast(`Threshold set to ${t}`); setTimeout(() => setToast(null), 2000) }} />
                  ) : (
                    <EmptyChart msg={buffer.length < 10 ? 'Need ≥ 10 labeled transactions' : 'Insufficient variance'} />
                  )}
                </div>
                {prPoints && (
                  <div className="flex justify-between items-center px-4 pb-3 font-mono text-[9px] text-text-muted">
                    <span className="italic">↖ upper-left = better (high precision AND recall)</span>
                    {tunedM.f >= deployedM.f
                      ? <span className="text-[#00e5a0] font-bold">✅ F1 +{((tunedM.f - deployedM.f) * 100).toFixed(1)}%</span>
                      : <span className="text-[#ff4d6d]">⚠ F1 -{((deployedM.f - tunedM.f) * 100).toFixed(1)}%</span>}
                  </div>
                )}
              </div>

              {/* ── ROW 6: Confusion Matrix ── */}
              <ChartCard title="Confusion Matrix" subtitle={`${buffer.length} transactions · Updates live with sliders`} height="auto">
                <ConfusionMatrix m={tunedM} total={buffer.length} />
              </ChartCard>

              {/* ── ROW 7: Business Impact Bar ── */}
              <div className="rounded-lg flex items-center justify-between px-5 py-3" style={{ background: 'rgba(255,255,255,0.02)', borderTop: '1px solid rgba(255,255,255,0.06)', height: '56px' }}>
                <ImpactItem icon="💰" label="Fraud Caught" value={fmtRM(impact.fraudCaught)} color="#00e5a0" />
                <ImpactItem icon="😤" label="Legit Blocked" value={`${impact.fpCount} txns · ${fmtRM(impact.fpFriction)}`} color="#ff9800" />
                <ImpactItem icon="📊" label="Net Protection" value={fmtRM(impact.net)} color={impact.net >= 0 ? '#00e5a0' : '#ff4d6d'} />
                <ImpactItem icon="📈" label="Protection Rate" value={`${impact.protRate.toFixed(1)}%`}
                  color={impact.protRate > 80 ? '#00e5a0' : impact.protRate > 50 ? '#ff9800' : '#ff4d6d'} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════════════
   SUB-COMPONENTS
   ══════════════════════════════════════════════════════════ */

function Slider({ label, cls, color, ico, value, onChange, min = 0, max = 1, step = 0.01, fmt }) {
  const display = fmt ? fmt(value) : `${(value * 100).toFixed(0)}%`
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <div className="flex items-center gap-1.5 font-mono text-[10px] text-text-secondary uppercase tracking-wider font-bold">
          {ico && <span style={{ color }}>{ico}</span>}{label}
        </div>
        <span className="font-mono text-[11px] font-bold" style={{ color }}>{display}</span>
      </div>
      <div className="relative h-[14px]">
        <div className="absolute top-[5px] left-0 h-[4px] rounded-full" style={{ width: `${pct}%`, background: color, opacity: 0.45 }} />
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          className={clsx("absolute inset-0 w-full cursor-pointer", cls)}
          style={{ '--slider-color': color }} />
      </div>
    </div>
  )
}

function OpCard({ label, val, base, total, good, invertDelta }) {
  const delta = val - base
  const pct = ((val / (total || 1)) * 100).toFixed(1)
  const better = invertDelta ? delta < 0 : delta > 0
  const color = delta === 0 ? 'text-text-muted' : better ? 'text-[#00e5a0]' : 'text-[#ff4d6d]'
  const Icon = delta > 0 ? TrendingUp : delta < 0 ? TrendingDown : Minus
  return (
    <div className="rounded-lg p-3.5 flex flex-col items-center" style={{ background: CARD.bg, border: CARD.border }}>
      <span className="section-label mb-1">{label}</span>
      <span className="font-mono font-bold text-[24px] text-text-primary leading-none">{val}</span>
      <div className={clsx("flex items-center gap-1 mt-1.5 font-mono text-[10px] font-bold", color)}>
        <Icon size={12} />{delta === 0 ? 'no change' : `(${delta > 0 ? '+' : ''}${delta})`}
      </div>
      <span className="font-mono text-[8px] text-text-muted mt-1">{pct}% total</span>
    </div>
  )
}

function PerfCard({ label, icon, val, base, fmt, tip, thr: [lo, hi], invert, extra }) {
  const delta = val - base
  const better = invert ? delta < 0 : delta > 0
  const dColor = delta === 0 ? 'text-text-muted' : better ? 'text-[#00e5a0]' : 'text-[#ff4d6d]'
  let bc
  if (invert) bc = val < lo ? '#00e5a0' : val < hi ? '#ff9800' : '#ff4d6d'
  else bc = val > hi ? '#00e5a0' : val > lo ? '#ff9800' : '#ff4d6d'
  const dv = fmt === 'pct' ? `${(val * 100).toFixed(1)}%` : val.toFixed(3)
  const dd = fmt === 'pct' ? `${(Math.abs(delta) * 100).toFixed(1)}%` : Math.abs(delta).toFixed(3)
  const bw = fmt === 'pct' ? val * 100 : val * 100
  return (
    <div className="group rounded-lg p-3.5 relative overflow-hidden" style={{ background: CARD.bg, border: `1px solid ${bc}30` }}>
      {icon && <div className="absolute bottom-2 right-2 opacity-[0.04] text-text-muted">{icon}</div>}
      <div className="flex justify-between items-start mb-2.5 relative z-10">
        <span className="section-label">{label}</span>
        <div className="relative">
          <Info size={10} className="text-text-muted/40 group-hover:text-text-secondary cursor-help transition-colors" />
          <div className="absolute bottom-full right-0 mb-2 w-48 p-2 rounded-lg shadow-lg text-[9px] text-text-primary font-mono opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none whitespace-pre-line" style={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)' }}>{tip}</div>
        </div>
      </div>
      <div className="font-mono font-bold text-[22px] text-text-primary leading-none mb-1 relative z-10">{dv}</div>
      <div className={clsx("font-mono text-[10px] font-bold mb-2.5 relative z-10", dColor)}>
        {delta === 0 ? 'stable' : `(${better ? '+' : '-'}${dd}) ${better ? '↑' : '↓'}`}
      </div>
      <div className="h-[3px] w-full rounded-full overflow-hidden relative z-10" style={{ background: 'rgba(255,255,255,0.06)' }}>
        <div className="h-full rounded-full transition-all duration-300" style={{ width: `${Math.min(100, bw)}%`, background: bc }} />
      </div>
      {extra && <p className="font-mono text-[8px] text-text-muted mt-1.5 relative z-10">{extra}</p>}
    </div>
  )
}

function ChartCard({ title, subtitle, height, children }) {
  return (
    <div className="rounded-lg overflow-hidden" style={{ background: CARD.bg, border: CARD.border }}>
      <div className="px-4 pt-3 pb-1">
        <h3 className="section-label">{title}</h3>
        {subtitle && <p className="font-mono text-[9px] text-text-muted mt-0.5">{subtitle}</p>}
      </div>
      <div className="px-4 pb-3" style={{ height }}>{children}</div>
    </div>
  )
}

function EmptyChart({ msg }) {
  return <div className="h-full flex items-center justify-center text-text-muted font-mono text-[10px] uppercase tracking-widest opacity-40">{msg}</div>
}

/* ── Threshold vs Metrics Chart ────────────────────────── */
function ThresholdChart({ data, currentT, currentFlagT, bp, onSelect }) {
  const annotations = {}
  // Categorical x-axis: labels are [0,2,4,...100] so index = T/2
  const ix = t => t / 2
  // Current Approve threshold line (amber)
  annotations.approveLine = { type: 'line', xMin: ix(currentT), xMax: ix(currentT), borderColor: '#ff9800', borderWidth: 2, label: { display: true, content: `Approve T=${currentT}`, position: 'start', backgroundColor: 'rgba(255,152,0,0.8)', font: { ...FONT, size: 8 } } }
  // Current Flag/Block threshold line (red)
  annotations.blockLine = { type: 'line', xMin: ix(currentFlagT), xMax: ix(currentFlagT), borderColor: '#ff4d6d', borderWidth: 2, label: { display: true, content: `Block T=${currentFlagT}`, position: 'start', backgroundColor: 'rgba(255,77,109,0.8)', font: { ...FONT, size: 8 } } }
  // Approve Balance (green dotted) — appears after Empirically Tune
  if (bp) {
    annotations.approveBalance = { type: 'line', xMin: ix(bp.approve.t), xMax: ix(bp.approve.t), borderColor: 'rgba(0,229,160,0.7)', borderWidth: 1.5, borderDash: [6, 4], label: { display: true, content: `Approve Balance · T=${bp.approve.t}`, position: 'end', color: 'rgba(0,229,160,0.7)', font: { size: 9, family: 'monospace' }, backgroundColor: 'transparent', yAdjust: -8 } }
    annotations.blockBalance = { type: 'line', xMin: ix(bp.block.t), xMax: ix(bp.block.t), borderColor: 'rgba(255,77,109,0.7)', borderWidth: 1.5, borderDash: [6, 4], label: { display: true, content: `Block Balance · T=${bp.block.t}`, position: 'end', color: 'rgba(255,77,109,0.7)', font: { size: 9, family: 'monospace' }, backgroundColor: 'transparent', yAdjust: -8 } }
    // FLAG ZONE shading
    annotations.flagZone = { type: 'box', xMin: ix(bp.approve.t), xMax: ix(bp.block.t), backgroundColor: 'rgba(255,152,0,0.05)', borderColor: 'rgba(255,152,0,0.15)', borderWidth: 1, borderDash: [3, 3], label: { display: true, content: 'FLAG ZONE', color: 'rgba(255,152,0,0.4)', font: { size: 10, family: 'monospace' } } }
  }
  return (
    <Line
      data={{
        labels: data.map(d => d.t),
        datasets: [
          { label: 'Precision', data: data.map(d => d.p), borderColor: '#00e5a0', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'Recall', data: data.map(d => d.r), borderColor: '#00e5ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'F1', data: data.map(d => d.f), borderColor: '#ffd700', borderWidth: 2, pointRadius: 0, tension: 0.3 },
          { label: 'FP Rate', data: data.map(d => d.fp), borderColor: '#ff4d6d', borderWidth: 1.5, pointRadius: 0, tension: 0.3, borderDash: [4, 2] },
        ],
      }}
      options={{
        responsive: true, maintainAspectRatio: false, animation: false,
        onClick: (e, _, chart) => {
          const xVal = chart.scales.x.getValueForPixel(e.x)
          const nearest = data.reduce((b, p) => Math.abs(p.t - xVal) < Math.abs(b.t - xVal) ? p : b, data[0])
          onSelect(nearest.t)
        },
        scales: {
          x: { title: { display: true, text: 'THRESHOLD', color: TICK_C, font: FONT }, grid: { color: GRID_C, drawBorder: false }, ticks: { color: TICK_C, font: FONT, maxTicksLimit: 10 } },
          y: { min: 0, max: 1, grid: { color: GRID_C, drawBorder: false }, ticks: { color: TICK_C, font: FONT } },
        },
        plugins: {
          legend: { position: 'top', align: 'end', labels: { color: TICK_C, font: { ...FONT, size: 10 }, usePointStyle: true, pointStyleWidth: 8, boxHeight: 6 } },
          tooltip: { backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, cornerRadius: 6, titleFont: FONT, bodyFont: FONT },
          annotation: { annotations },
        },
      }}
    />
  )
}

/* ── Score Distribution Chart ──────────────────────────── */
function ScoreDistChart({ bins, threshold }) {
  const labels = bins.map((_, i) => i * 5)
  return (
    <Bar
      data={{
        labels,
        datasets: [
          { label: 'Legit', data: bins.map(b => b.legit), backgroundColor: 'rgba(0,229,160,0.4)', borderRadius: 2, barPercentage: 0.9, categoryPercentage: 0.85 },
          { label: 'Fraud', data: bins.map(b => b.fraud), backgroundColor: 'rgba(255,77,109,0.4)', borderRadius: 2, barPercentage: 0.9, categoryPercentage: 0.85 },
        ],
      }}
      options={{
        responsive: true, maintainAspectRatio: false, animation: false,
        scales: {
          x: { stacked: false, grid: { display: false }, ticks: { color: TICK_C, font: FONT } },
          y: { grid: { color: GRID_C, drawBorder: false }, ticks: { color: TICK_C, font: FONT } },
        },
        plugins: {
          legend: { position: 'top', align: 'end', labels: { color: TICK_C, font: { ...FONT, size: 10 }, usePointStyle: true, pointStyleWidth: 8, boxHeight: 6 } },
          tooltip: { backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, cornerRadius: 6, titleFont: FONT, bodyFont: FONT },
          annotation: {
            annotations: {
              threshLine: { type: 'line', xMin: threshold / 5, xMax: threshold / 5, borderColor: '#ff9800', borderWidth: 2, borderDash: [5, 3], label: { display: true, content: `T=${threshold}`, position: 'start', backgroundColor: 'rgba(255,152,0,0.8)', font: { ...FONT, size: 8 } } },
            }
          },
        },
      }}
    />
  )
}

/* ── PR Curve ──────────────────────────────────────────── */
function PRChart({ points, dPt, tPt, onSelect }) {
  return (
    <Line
      data={{
        datasets: [
          { label: 'PR Curve', data: points.map(p => ({ x: p.x, y: p.y })), borderColor: '#00e5ff', borderWidth: 2, backgroundColor: 'rgba(0,229,255,0.05)', fill: true, tension: 0.4, pointRadius: 0, showLine: true },
          { label: '● Deployed', data: [dPt], pointBackgroundColor: '#00e5ff', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 8, pointHoverRadius: 10, showLine: false },
          { label: '○ Tuned', data: [tPt], pointBackgroundColor: 'transparent', pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 8, pointHoverRadius: 10, showLine: false },
        ],
      }}
      options={{
        responsive: true, maintainAspectRatio: false, animation: false,
        onClick: (e, _, chart) => {
          const xV = chart.scales.x.getValueForPixel(e.x), yV = chart.scales.y.getValueForPixel(e.y)
          let best = points[0], bd = Infinity
          points.forEach(p => { const d = (p.x - xV) ** 2 + (p.y - yV) ** 2; if (d < bd) { bd = d; best = p } })
          onSelect(best.t)
        },
        scales: {
          x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'RECALL', color: TICK_C, font: FONT }, grid: { color: GRID_C, drawBorder: false }, ticks: { color: TICK_C, font: FONT } },
          y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'PRECISION', color: TICK_C, font: FONT }, grid: { color: GRID_C, drawBorder: false }, ticks: { color: TICK_C, font: FONT } },
        },
        plugins: {
          legend: { position: 'top', align: 'end', labels: { color: TICK_C, font: { ...FONT, size: 10 }, usePointStyle: true, pointStyleWidth: 8, boxHeight: 6 } },
          tooltip: { backgroundColor: '#111827', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, cornerRadius: 6, titleFont: FONT, bodyFont: FONT,
            callbacks: { label: ctx => {
              if (ctx.datasetIndex === 0) { const p = points[ctx.dataIndex]; return `T=${p?.t} P=${(ctx.parsed.y * 100).toFixed(1)}% R=${(ctx.parsed.x * 100).toFixed(1)}%` }
              return `P=${(ctx.parsed.y * 100).toFixed(1)}% R=${(ctx.parsed.x * 100).toFixed(1)}%`
            } },
          },
        },
      }}
    />
  )
}

/* ── Confusion Matrix ──────────────────────────────────── */
function ConfusionMatrix({ m, total }) {
  const maxVal = Math.max(m.TN, m.FP, m.FN, m.TP, 1)
  const cell = (label, val, isGood, sub) => {
    const intensity = Math.min(0.6, (val / maxVal) * 0.6)
    const bg = isGood ? `rgba(0,229,160,${intensity})` : `rgba(255,77,109,${intensity})`
    return (
      <div className="flex flex-col items-center justify-center p-3 rounded-lg transition-all duration-300" style={{ background: bg, border: CARD.border }}>
        <span className="font-mono text-[8px] text-text-muted uppercase tracking-widest mb-1">{label}</span>
        <span className="font-mono font-bold text-[28px] text-text-primary leading-none">{val}</span>
        <span className="font-mono text-[9px] text-text-muted mt-1">{total > 0 ? ((val / total) * 100).toFixed(1) : 0}%</span>
        <span className="font-mono text-[7px] text-text-muted mt-0.5 opacity-70">{sub}</span>
      </div>
    )
  }
  return (
    <div>
      {/* Header row */}
      <div className="grid grid-cols-[100px_1fr_1fr] gap-2 mb-2">
        <div />
        <div className="text-center font-mono text-[8px] text-text-muted uppercase tracking-widest">Predicted LEGIT</div>
        <div className="text-center font-mono text-[8px] text-text-muted uppercase tracking-widest">Predicted FRAUD</div>
      </div>
      {/* Actual LEGIT row */}
      <div className="grid grid-cols-[100px_1fr_1fr] gap-2 mb-2">
        <div className="flex items-center justify-end pr-2 font-mono text-[8px] text-text-muted uppercase tracking-widest">Actual Legit</div>
        {cell('TN', m.TN, true, 'Legit, Approved')}
        {cell('FP', m.FP, false, 'Legit, Blocked')}
      </div>
      {/* Actual FRAUD row */}
      <div className="grid grid-cols-[100px_1fr_1fr] gap-2 mb-3">
        <div className="flex items-center justify-end pr-2 font-mono text-[8px] text-text-muted uppercase tracking-widest">Actual Fraud</div>
        {cell('FN', m.FN, false, 'Fraud, Missed')}
        {cell('TP', m.TP, true, 'Fraud, Caught')}
      </div>
      {/* Insight lines */}
      <div className="space-y-1 font-mono text-[9px]">
        {m.FP > 0 && <p className="text-[#ff4d6d]">⚠ {m.FP} legitimate transactions incorrectly blocked</p>}
        {m.FN > 0 && <p className="text-[#ff4d6d]">⚠ {m.FN} fraud transactions slipped through</p>}
      </div>
    </div>
  )
}

/* ── Business Impact Item ──────────────────────────────── */
function ImpactItem({ icon, label, value, color }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm">{icon}</span>
      <div>
        <span className="font-mono text-[8px] text-text-muted uppercase tracking-widest block">{label}</span>
        <span className="font-mono text-[11px] font-bold" style={{ color }}>{value}</span>
      </div>
    </div>
  )
}
