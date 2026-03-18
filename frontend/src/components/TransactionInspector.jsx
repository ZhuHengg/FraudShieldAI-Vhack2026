import React, { useState, useEffect } from 'react'
import Panel from './shared/Panel'
import { Lock, Check, AlertTriangle, ShieldAlert, X, ArrowRight, Fingerprint, Globe, Zap } from 'lucide-react'
import clsx from 'clsx'

/* ─── Helpers ────────────────────────────────────────── */
const formatID = (id) => id ? id.slice(0, 12) : 'UNKNOWN'
const formatDate = (dateStr) => {
  try {
    const d = new Date(dateStr)
    return `${d.toLocaleDateString('en-CA')} · ${d.toLocaleTimeString('en-GB')}`
  } catch { return 'N/A' }
}
const channelMap = {
  'cash_out': 'ATM / Withdrawal',
  'CASH_OUT': 'ATM / Withdrawal',
  'transfer': 'P2P Transfer',
  'TRANSFER': 'P2P Transfer',
  'payment': 'Merchant Payment',
  'PAYMENT': 'Merchant Payment',
}

/* ─── Status Dot Component ───────────────────────────── */
const StatusDot = ({ status }) => {
  const colors = {
    danger:  'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]',
    warning: 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]',
    safe:    'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]',
  }
  return <div className={clsx('w-2 h-2 rounded-full shrink-0', colors[status] || colors.safe)} />
}

export default function TransactionInspector({ selectedTxn, engine }) {
  const [gaugeAnimated, setGaugeAnimated] = useState(false)
  const [barsAnimated, setBarsAnimated] = useState(false)

  const handleClose = () => engine?.setSelectedTxn?.(null)

  useEffect(() => {
    if (!selectedTxn) { setGaugeAnimated(false); setBarsAnimated(false); return }
    setGaugeAnimated(false); setBarsAnimated(false)
    const t1 = setTimeout(() => setGaugeAnimated(true), 100)
    const t2 = setTimeout(() => setBarsAnimated(true), 400)
    return () => { clearTimeout(t1); clearTimeout(t2) }
  }, [selectedTxn])

  if (!selectedTxn) {
    return (
      <div className="flex flex-col gap-3 w-[400px] h-full shrink-0 animate-in fade-in duration-300">
        <Panel className="flex-1 flex items-center justify-center border border-border bg-bg-100">
          <span className="font-mono text-[12px] text-text-muted tracking-widest uppercase text-center">
            SELECT A TRANSACTION<br />TO INSPECT
          </span>
        </Panel>
      </div>
    )
  }

  /* ─── Data Extraction ──────────────────────────────── */
  const scoreRaw = selectedTxn.ensembleScore ? (selectedTxn.ensembleScore * 100) : 0
  const score = Math.min(100, Math.max(0, scoreRaw))
  const w_lgb = 0.55, w_iso = 0.25, w_beh = 0.20
  const lgbScore = selectedTxn.lgbScore || 0
  const isoScore = selectedTxn.isoScore || 0
  const behScore = selectedTxn.behScore || 0
  const reasons = selectedTxn.reasons || selectedTxn.riskFactors || []
  let finalReasons = reasons.length ? reasons : ["Normal behavior pattern"]
  if (finalReasons.length === 1 && finalReasons[0].toLowerCase().includes('normal')) {
    finalReasons = ["NO_RULES_FIRED"]
  }

  const dpApplied = selectedTxn.privacy?.dp_applied ?? true
  const hashAlgo = selectedTxn.privacy?.hash_algorithm ?? 'SHA-256'
  const latencyMs = Math.round(selectedTxn.latencyMs || 42)

  // Feature snapshot (from backend or raw generated data)
  const fs = selectedTxn.feature_snapshot || {}
  const amtRatio = fs.amount_vs_avg_ratio ?? selectedTxn.amountVsAvgRatio ?? selectedTxn.amount_vs_avg_ratio ?? 1
  const ipRisk = fs.ip_risk_score ?? selectedTxn.ipRiskScore ?? selectedTxn.ip_risk_score ?? 0
  const txCount = fs.tx_count_24h ?? selectedTxn.txCount24h ?? selectedTxn.tx_count_24h ?? 0
  const sessionDur = fs.session_duration_seconds ?? selectedTxn.sessionDurationSeconds ?? selectedTxn.session_duration_seconds ?? 0
  const isNewDev = fs.is_new_device ?? selectedTxn.isNewDevice ?? selectedTxn.is_new_device ?? 0
  const countryMM = fs.country_mismatch ?? selectedTxn.countryMismatch ?? selectedTxn.country_mismatch ?? 0
  const senderDrained = fs.sender_fully_drained ?? selectedTxn.senderFullyDrained ?? selectedTxn.sender_account_fully_drained ?? 0
  const isNewRecip = fs.is_new_recipient ?? selectedTxn.isNewRecipient ?? selectedTxn.is_new_recipient ?? 0
  const accountAge = fs.account_age_days ?? selectedTxn.accountAgeDays ?? selectedTxn.account_age_days ?? 0
  const failedLogins = selectedTxn.failed_login_attempts ?? 0
  const isProxyIp = fs.is_proxy_ip ?? selectedTxn.isProxyIp ?? selectedTxn.is_proxy_ip ?? 0

  // Balances (from real generated data)
  const balBefore = selectedTxn.sender_balance_before ?? 0
  const balAfter = selectedTxn.sender_balance_after ?? 0
  const balChangePct = balBefore > 0 ? ((balAfter - balBefore) / balBefore * 100) : 0
  const recipRisk = selectedTxn.recipient_risk_profile_score ?? 0

  // Threshold / risk helpers
  const approve_threshold = 35, flag_threshold = 60
  let riskLevel = "LOW RISK", riskColor = "#34d399", decisionWord = "APPROVE"
  if (score > flag_threshold) { riskLevel = "HIGH RISK"; riskColor = "#ef4444"; decisionWord = "BLOCK" }
  else if (score >= approve_threshold) { riskLevel = "MEDIUM RISK"; riskColor = "#f59e0b"; decisionWord = "FLAG" }

  // Gauge math
  const radius = 70, circumference = 2 * Math.PI * radius, semiCircumference = circumference / 2
  const fillPct = gaugeAnimated ? (score / 100) : 0
  const strokeDashoffset = semiCircumference - (fillPct * semiCircumference)

  // Ground truth
  const isFraud = selectedTxn.isFraud ?? false
  const groundTruth = isFraud ? "FRAUD" : "LEGIT"
  const modelPredictedFraud = selectedTxn.decision !== 'APPROVE'
  let verdict, verdictColor, verdictIcon
  if (modelPredictedFraud && isFraud) { verdict = "TRUE POSITIVE"; verdictColor = "#10b981"; verdictIcon = "✅" }
  else if (!modelPredictedFraud && !isFraud) { verdict = "TRUE NEGATIVE"; verdictColor = "#10b981"; verdictIcon = "✅" }
  else if (modelPredictedFraud && !isFraud) { verdict = "FALSE POSITIVE"; verdictColor = "#f59e0b"; verdictIcon = "⚠" }
  else { verdict = "FALSE NEGATIVE"; verdictColor = "#ef4444"; verdictIcon = "✗" }

  // Feature rules fired count
  const rulesFireCount = [
    amtRatio > 1.5, senderDrained, ipRisk > 0.5, countryMM, isNewDev, txCount > 5, sessionDur < 60
  ].filter(Boolean).length

  const behContribution = behScore > 0.5 ? 'HIGH' : behScore > 0.2 ? 'MEDIUM' : 'LOW'

  // Ground truth bullets
  const gtBullets = []
  if (amtRatio > 5) gtBullets.push(`Amount ${amtRatio.toFixed(1)}× user's historical average`)
  else if (amtRatio > 1.5) gtBullets.push(`Amount ${amtRatio.toFixed(1)}× above average baseline`)
  if (senderDrained) gtBullets.push("Sender account completely emptied")
  if (countryMM) gtBullets.push("Transaction from foreign IP/location")
  if (isNewDev) gtBullets.push("Unrecognised device used")
  if (sessionDur > 0 && sessionDur < 60) gtBullets.push(`Transaction completed in ${Math.round(sessionDur)} seconds`)
  if (ipRisk > 0.7) gtBullets.push(`High-risk IP address (score: ${ipRisk.toFixed(2)})`)
  if (txCount > 5) gtBullets.push(`${txCount} transactions in 24h (velocity anomaly)`)
  if (gtBullets.length === 0) gtBullets.push("No anomalous feature thresholds breached")

  // Channel
  const txType = selectedTxn.transfer_type || selectedTxn.transaction_type || 'TRANSFER'
  const channel = channelMap[txType] || 'P2P Transfer'

  /* ─── Feature Table Rows ───────────────────────────── */
  const featureRows = [
    {
      param: 'amount_vs_avg_ratio',
      value: `${amtRatio.toFixed(2)}×`,
      threshold: '> 1.5×',
      status: amtRatio > 5 ? 'danger' : amtRatio > 1.5 ? 'warning' : 'safe',
    },
    {
      param: 'sender_account_drained',
      value: senderDrained ? 'YES' : 'NO',
      threshold: '—',
      status: senderDrained ? 'danger' : 'safe',
    },
    {
      param: 'ip_risk_score',
      value: ipRisk.toFixed(3),
      threshold: '> 0.5',
      status: ipRisk > 0.7 ? 'danger' : ipRisk > 0.5 ? 'warning' : 'safe',
    },
    {
      param: 'country_mismatch',
      value: countryMM ? 'YES (foreign)' : 'NO (domestic)',
      threshold: '—',
      status: countryMM ? 'danger' : 'safe',
    },
    {
      param: 'is_new_device',
      value: isNewDev ? 'YES' : 'NO',
      threshold: '—',
      status: isNewDev ? 'warning' : 'safe',
    },
    {
      param: 'tx_count_24h',
      value: txCount.toString(),
      threshold: '> 5',
      status: txCount > 5 ? 'danger' : 'safe',
    },
    {
      param: 'session_duration_secs',
      value: `${Math.round(sessionDur)}s`,
      threshold: '< 60s',
      status: sessionDur < 60 ? 'danger' : 'safe',
    },
    {
      param: 'account_age_days',
      value: `${Math.round(accountAge)}d`,
      threshold: '—',
      status: 'safe',
    },
    {
      param: 'failed_login_attempts',
      value: failedLogins.toString(),
      threshold: '> 0',
      status: failedLogins > 0 ? 'danger' : 'safe',
    },
  ]

  /* ─── Reason Badge Renderer ────────────────────────── */
  const renderReasonBadge = (reasonText, idx) => {
    if (reasonText === "NO_RULES_FIRED") {
      return (
        <div key={idx} className="flex flex-col mb-2">
          <span className="font-mono text-[10px] text-text-muted font-bold tracking-widest">NO RULES FIRED —</span>
          <span className="font-sans text-[12px] text-text-muted/60">Normal behavior pattern</span>
        </div>
      )
    }
    let dotColor = "bg-red-500", textColor = "text-red-400", ruleTag = "RULE FIRED", tagColor = "text-red-400 bg-red-500/10 border-red-500/20"
    const lower = reasonText.toLowerCase()
    if (lower.includes('drain')) { ruleTag = "DRAIN → UNKNOWN" }
    else if (lower.includes('deviation') || lower.includes('exceeds')) { ruleTag = "AMT DEVIATION"; dotColor = "bg-amber-500"; textColor = "text-amber-400"; tagColor = "text-amber-400 bg-amber-500/10 border-amber-500/20" }
    else if (lower.includes('context') || lower.includes('foreign') || lower.includes('device')) { ruleTag = "RISKY CONTEXT"; dotColor = "bg-purple-500"; textColor = "text-purple-400"; tagColor = "text-purple-400 bg-purple-500/10 border-purple-500/20" }
    else if (lower.includes('velocity') || lower.includes('rapid')) { ruleTag = "RAPID SESSION"; dotColor = "bg-cyan-500"; textColor = "text-cyan-400"; tagColor = "text-cyan-400 bg-cyan-500/10 border-cyan-500/20" }
    return (
      <div key={idx} className="flex items-start gap-3 mb-3 last:mb-0">
        <div className={clsx("w-2 h-2 rounded-full shrink-0 mt-1.5", dotColor)} />
        <div className="flex flex-col gap-1">
          <span className={clsx("font-mono text-[9px] font-bold tracking-widest px-1.5 py-0.5 rounded border uppercase w-fit", tagColor)}>{ruleTag}</span>
          <span className={clsx("font-sans text-[12px] leading-snug", textColor)}>{reasonText}</span>
        </div>
      </div>
    )
  }

  /* ═══════════════════════════════════════════════════════
     RENDER
     ═══════════════════════════════════════════════════════ */
  return (
    <div className="flex flex-col w-[420px] h-full bg-bg-100 border-l border-border shrink-0 animate-in slide-in-from-right-8 duration-300 relative rounded-r-2xl overflow-hidden shadow-2xl z-50">

      {/* Close Button */}
      <button onClick={handleClose} className="absolute top-4 right-4 z-10 p-1.5 rounded-full bg-bg-200 hover:bg-bg-300 text-text-muted hover:text-white transition-colors">
        <X size={16} />
      </button>

      <div className="flex-1 overflow-y-auto custom-scrollbar p-5 space-y-6">

        {/* ═══ SECTION 1: TRANSACTION PROFILE ═══ */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <div className="flex items-center gap-1.5 px-2 py-0.5 bg-[#10b981]/10 border border-[#10b981]/20 rounded text-[#10b981] font-mono text-[9px] font-bold tracking-widest">
              <div className="w-1.5 h-1.5 rounded-full bg-[#10b981] animate-pulse" />
              LIVE
            </div>
            <span className="font-mono text-[11px] text-text-primary font-bold">
              {(selectedTxn.transaction_id || selectedTxn.id || 'UNKNOWN').toUpperCase()}
            </span>
          </div>

          {/* Type badge + Channel + Timestamp */}
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-[9px] font-bold tracking-widest px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 uppercase">
              {txType}
            </span>
            <span className="font-mono text-[10px] text-text-muted">{channel}</span>
          </div>
          <div className="font-mono text-[10px] text-text-muted/60 tracking-widest mb-4">
            {formatDate(selectedTxn.timestamp || new Date())}
          </div>

          {/* Amount */}
          <div className="bg-bg-200/50 p-4 rounded-xl border border-border/50 mb-4">
            <div className="font-mono text-[32px] font-bold text-text-primary leading-none tracking-tight mb-4">
              RM {selectedTxn.amount?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>

            {/* SENDER Block */}
            <div className="mb-3 p-3 rounded-lg bg-bg-300/30 border border-border/30">
              <div className="font-mono text-[9px] text-text-muted/60 uppercase tracking-widest mb-1">SENDER</div>
              <div className="font-mono text-[12px] text-cyan-400 font-bold mb-1">{formatID(selectedTxn.userId || selectedTxn.name_sender)}</div>
              <div className="flex items-center gap-2 font-mono text-[11px] text-text-muted">
                <span>RM {balBefore.toFixed(2)}</span>
                <ArrowRight size={10} className="text-text-muted/40" />
                <span>RM {balAfter.toFixed(2)}</span>
                <span className={clsx("font-bold", balChangePct < -90 ? "text-red-400" : balChangePct < 0 ? "text-amber-400" : "text-emerald-400")}>
                  ({balChangePct.toFixed(1)}%)
                </span>
              </div>
              {(balAfter <= 0 || senderDrained) && (
                <div className="mt-1.5 inline-flex items-center gap-1 px-2 py-0.5 rounded bg-red-500/15 border border-red-500/30 text-red-400 font-mono text-[9px] font-bold tracking-widest">
                  ⚠ ACCOUNT FULLY DRAINED
                </div>
              )}
              {balChangePct < -90 && balAfter > 0 && (
                <div className="mt-1.5 inline-flex items-center gap-1 px-2 py-0.5 rounded bg-red-500/10 border border-red-500/20 text-red-400 font-mono text-[9px] font-bold tracking-widest">
                  ⚠ BALANCE CHANGE &gt; 90%
                </div>
              )}
            </div>

            {/* RECIPIENT Block */}
            <div className="p-3 rounded-lg bg-bg-300/30 border border-border/30">
              <div className="font-mono text-[9px] text-text-muted/60 uppercase tracking-widest mb-1">RECIPIENT</div>
              <div className="font-mono text-[12px] text-cyan-400 font-bold mb-1">{formatID(selectedTxn.receiverId || selectedTxn.name_recipient)}</div>
              {isNewRecip ? (
                <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-400 font-mono text-[9px] font-bold tracking-widest">
                  ⚠ FIRST-TIME RECIPIENT
                </div>
              ) : (
                <div className="font-mono text-[10px] text-text-muted/50">Known recipient</div>
              )}
              {recipRisk > 0 && (
                <div className="font-mono text-[10px] text-text-muted mt-1">
                  Risk profile: <span className={clsx("font-bold", recipRisk > 0.5 ? "text-red-400" : "text-text-secondary")}>{recipRisk.toFixed(2)}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ═══ SCORE & ENSEMBLE BREAKDOWN ═══ */}
        <div>
          <h3 className="section-label mb-5 border-b border-border/50 pb-2">SCORE & ENSEMBLE BREAKDOWN</h3>

          {/* Gauge */}
          <div className="flex flex-col items-center mb-8 relative">
            <div className="relative w-[180px] h-[95px] overflow-hidden flex justify-center">
              <svg viewBox="0 0 160 160" className="absolute top-0 w-[160px] h-[160px] rotate-180">
                <circle cx="80" cy="80" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="14" strokeDasharray={`${semiCircumference} ${circumference}`} />
                <circle cx="80" cy="80" r={radius} fill="none" stroke={riskColor} strokeWidth="14" strokeDasharray={`${semiCircumference} ${circumference}`} strokeDashoffset={strokeDashoffset} className="transition-all duration-[600ms] ease-out drop-shadow-[0_0_8px_currentColor]" />
              </svg>
              <div className="absolute bottom-2 w-full text-center flex flex-col items-center">
                <div className="font-mono text-[42px] font-bold leading-none tracking-tighter" style={{ color: riskColor }}>{Math.round(score)}</div>
                <div className="font-mono text-[10px] tracking-[0.2em] font-bold mt-1" style={{ color: riskColor }}>{riskLevel}</div>
              </div>
            </div>
            <div className="w-[170px] flex justify-between font-mono text-[10px] text-text-muted/50 mt-1">
              <span>0</span><span>100</span>
            </div>
          </div>

          {/* Bars */}
          <div className="flex flex-col gap-3 mb-6">
            {[
              { label: 'LIGHTGBM', pct: '55%', val: lgbScore, color: '#4FC3F7', delay: 100 },
              { label: 'ISOFOREST', pct: '25%', val: isoScore, color: '#ce93d8', delay: 200 },
              { label: 'BEHAVIORAL', pct: '20%', val: behScore, color: '#ffb74d', delay: 300 }
            ].map((bar, idx) => (
              <div key={idx} className="flex items-center gap-3">
                <div className="flex flex-col w-[110px] shrink-0">
                  <span className="font-mono text-[10px] tracking-widest text-text-secondary">{bar.label}</span>
                  <span className="font-mono text-[9px] text-text-muted/60">({bar.pct})</span>
                </div>
                <div className="flex-1 h-[6px] bg-bg-300 rounded-full relative overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ backgroundColor: bar.color, width: barsAnimated ? `${bar.val * 100}%` : '0%', transitionDelay: `${bar.delay}ms` }} />
                </div>
                <div className="w-[30px] text-right font-mono text-[11px] font-bold" style={{ color: bar.color }}>{(bar.val * 100).toFixed(0)}</div>
              </div>
            ))}
          </div>

          {/* Formula Block */}
          <div className="bg-[#050505] border border-white/10 rounded-lg p-3 relative overflow-hidden">
            <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-[#4FC3F7] via-[#ce93d8] to-[#ffb74d] opacity-50" />
            <div className="font-mono text-[11px] leading-[1.6] pl-2 whitespace-pre-wrap break-all">
              (<span className="text-[#4FC3F7]">{(lgbScore * 100).toFixed(1)}</span>×<span className="text-[#4FC3F7]">{w_lgb}</span>) +{' '}
              (<span className="text-[#ce93d8]">{(isoScore * 100).toFixed(1)}</span>×<span className="text-[#ce93d8]">{w_iso}</span>) +<br />
              (<span className="text-[#ffb74d]">{(behScore * 100).toFixed(1)}</span>×<span className="text-[#ffb74d]">{w_beh}</span>) = <span className="text-white font-bold">{score.toFixed(1)}</span> → <span className="font-bold" style={{ color: riskColor }}>{decisionWord}</span>
            </div>
          </div>
          <div className="mt-2 text-right font-mono text-[10px] text-text-muted/40">Scored in {latencyMs}ms</div>
        </div>

        {/* ═══ SECTION 2: FEATURE PARAMETERS TABLE ═══ */}
        <div>
          <h3 className="section-label mb-3 border-b border-border/50 pb-2">FEATURE PARAMETERS</h3>
          <div className="rounded-xl border border-border/30 overflow-hidden">
            {/* Table Header */}
            <div className="grid grid-cols-[1fr_80px_60px_36px] gap-1 px-3 py-2 bg-bg-300/30 font-mono text-[8px] text-text-muted/60 uppercase tracking-widest">
              <span>Parameter</span>
              <span>Value</span>
              <span>Threshold</span>
              <span className="text-center">Status</span>
            </div>
            {/* Table Rows */}
            {featureRows.map((row, i) => (
              <div
                key={row.param}
                className={clsx(
                  "grid grid-cols-[1fr_80px_60px_36px] gap-1 px-3 py-1.5 font-mono text-[10px] border-t border-border/10 transition-colors hover:bg-red-500/[0.03]",
                  i % 2 === 0 ? 'bg-transparent' : 'bg-bg-200/20'
                )}
              >
                <span className="text-text-muted truncate">{row.param}</span>
                <span className={clsx("font-bold", row.status === 'danger' ? 'text-red-400' : row.status === 'warning' ? 'text-amber-400' : 'text-text-secondary')}>
                  {row.value}
                </span>
                <span className="text-text-muted/50">{row.threshold}</span>
                <div className="flex justify-center items-center">
                  <StatusDot status={row.status} />
                </div>
              </div>
            ))}
          </div>
          {/* Footer */}
          <div className="mt-2 font-mono text-[10px] text-text-muted/50 text-right">
            {rulesFireCount} rules fired · Behavioral contribution: <span className={clsx("font-bold", behContribution === 'HIGH' ? 'text-red-400' : behContribution === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400')}>{behContribution}</span>
          </div>
        </div>

        {/* ═══ WHY THIS SCORE? ═══ */}
        <div>
          <h3 className="section-label mb-4 border-b border-border/50 pb-2">WHY THIS SCORE?</h3>
          <div className="flex flex-col bg-bg-200/30 p-4 rounded-xl border border-border/30">
            {finalReasons.map((r, i) => renderReasonBadge(r, i))}
          </div>
        </div>

        {/* ═══ SECTION 3: GROUND TRUTH REFERENCE ═══ */}
        <div>
          <h3 className="section-label mb-3 border-b border-border/50 pb-2">GROUND TRUTH REFERENCE</h3>
          <div
            className="rounded-xl p-4 bg-bg-200/30 border border-border/30"
            style={{ borderLeft: `3px solid ${isFraud ? '#ff4d6d' : '#00e5a0'}` }}
          >
            {/* Label source */}
            <div className="font-mono text-[9px] text-text-muted/50 uppercase tracking-widest mb-2">
              Label Source: <span className="text-text-muted">Simulation Rule Engine</span>
            </div>

            {/* True Label */}
            <div className="flex items-center gap-2 mb-3">
              <span className="font-mono text-[10px] text-text-muted/60 uppercase tracking-widest">True Label:</span>
              <span className={clsx("font-mono text-[12px] font-bold", isFraud ? 'text-red-400' : 'text-emerald-400')}>
                {isFraud ? '🔴 FRAUD' : '✅ LEGIT'}
              </span>
            </div>

            {/* Verdict Badge */}
            <div
              className="flex items-center gap-2 px-3 py-2 rounded-lg mb-4 border"
              style={{
                backgroundColor: `${verdictColor}10`,
                borderColor: `${verdictColor}30`,
              }}
            >
              <span className="text-[18px]">{verdictIcon}</span>
              <span className="font-mono text-[14px] font-bold tracking-wider" style={{ color: verdictColor }}>
                {verdict}
              </span>
            </div>

            {/* Verdict description */}
            <div className="font-mono text-[10px] text-text-muted/60 mb-3">
              {verdict === 'TRUE POSITIVE' && 'Fraud correctly blocked by model.'}
              {verdict === 'TRUE NEGATIVE' && 'Legitimate transaction correctly approved.'}
              {verdict === 'FALSE POSITIVE' && 'Legit transaction incorrectly blocked (over-flagged).'}
              {verdict === 'FALSE NEGATIVE' && 'Fraudulent transaction missed by model.'}
            </div>

            {/* Why bullets */}
            <div className="font-mono text-[9px] text-text-muted/50 uppercase tracking-widest mb-2">
              Why it should be {groundTruth}:
            </div>
            <ul className="space-y-1.5">
              {gtBullets.map((b, i) => (
                <li key={i} className="flex items-start gap-2 font-sans text-[11px] text-text-secondary leading-snug">
                  <div className={clsx("w-1.5 h-1.5 rounded-full shrink-0 mt-1.5", isFraud ? 'bg-red-400' : 'bg-emerald-400')} />
                  {b}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* ═══ PRIVACY BADGE ═══ */}
        <div className="flex gap-3 p-3 rounded-lg border border-border/50 bg-[#0d1117]">
          <Lock size={16} className="text-text-muted shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1.5 font-sans text-[11px]">
            <span className="text-text-secondary leading-snug">
              PII hashed with <span className="font-mono text-cyan-400">{hashAlgo}</span> — raw identities never reached the model
            </span>
            <span className={clsx("font-bold tracking-wide text-[10px] uppercase", dpApplied ? "text-emerald-400" : "text-text-muted/40")}>
              Differential privacy noise {dpApplied ? "APPLIED" : "NOT APPLIED"}
            </span>
          </div>
        </div>

        {/* ═══ ADMIN DECISION PANEL ═══ */}
        <div>
          <h3 className="section-label mb-4 border-b border-border/50 pb-2">ADMIN DECISION PANEL</h3>
          <div className="flex flex-col gap-2 mb-4">
            {score < approve_threshold ? (
              <>
                <div className="flex items-center gap-2 text-emerald-400 font-mono text-[12px] font-bold tracking-wide"><Lock size={14} /> DECISION LOCKED — HIGH CONFIDENCE</div>
                <span className="text-[11px] text-text-muted leading-relaxed max-w-[90%]">Model confidence is high. No override required.</span>
              </>
            ) : score <= flag_threshold ? (
              <>
                <div className="flex items-center gap-2 text-amber-500 font-mono text-[12px] font-bold tracking-wide"><AlertTriangle size={14} /> MANUAL REVIEW SUGGESTED</div>
                <span className="text-[11px] text-text-muted leading-relaxed max-w-[90%]">Model is uncertain. Human review recommended.</span>
              </>
            ) : (
              <>
                <div className="flex items-center gap-2 text-red-500 font-mono text-[12px] font-bold tracking-wide"><Lock size={14} /> DECISION LOCKED — HIGH CONFIDENCE</div>
                <span className="text-[11px] text-text-muted leading-relaxed max-w-[90%]">High fraud probability. Transaction blocked.</span>
              </>
            )}
          </div>
          <div className="grid grid-cols-3 gap-2">
            <button disabled={score > flag_threshold} className={clsx(
              "flex flex-col items-center justify-center gap-1.5 py-3 rounded-lg border font-mono text-[10px] font-bold tracking-widest transition-all",
              score < approve_threshold ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.15)] ring-1 ring-emerald-500/30"
                : score <= flag_threshold ? "bg-bg-200 text-text-secondary border-border hover:bg-emerald-500/10 hover:text-emerald-400 hover:border-emerald-500/30"
                : "bg-bg-200/30 text-text-muted/30 border-border/30 cursor-not-allowed"
            )}><Check size={16} />APPROVE</button>
            <button disabled={score < approve_threshold || score > flag_threshold} className={clsx(
              "flex flex-col items-center justify-center gap-1.5 py-3 rounded-lg border font-mono text-[10px] font-bold tracking-widest transition-all",
              score <= flag_threshold && score >= approve_threshold ? "bg-bg-200 text-text-secondary border-border hover:bg-amber-500/10 hover:text-amber-400 hover:border-amber-500/30"
                : "bg-bg-200/30 text-text-muted/30 border-border/30 cursor-not-allowed"
            )}><AlertTriangle size={16} />FLAG</button>
            <button disabled={score < flag_threshold} className={clsx(
              "flex flex-col items-center justify-center gap-1.5 py-3 rounded-lg border font-mono text-[10px] font-bold tracking-widest transition-all",
              score > flag_threshold ? "bg-red-500/20 text-red-400 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.15)] ring-1 ring-red-500/30"
                : score >= approve_threshold ? "bg-bg-200 text-text-secondary border-border hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/30"
                : "bg-bg-200/30 text-text-muted/30 border-border/30 cursor-not-allowed"
            )}><ShieldAlert size={16} />BLOCK</button>
          </div>
        </div>

      </div>
    </div>
  )
}
