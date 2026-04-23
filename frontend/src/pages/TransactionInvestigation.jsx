import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  Search as SearchIcon, AlertTriangle, ShieldAlert, Check, X,
  Lock, Loader2, Info, ChevronDown, ChevronUp, FileText,
  ArrowRight, Fingerprint, Globe, Zap
} from 'lucide-react'
import clsx from 'clsx'
import { formatCurrency } from '../utils/formatters'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts'

const API_BASE = 'http://localhost:8000'

// ─── Status Dot Component ─────────────────────────────
const StatusDot = ({ status }) => {
  const colors = {
    danger:  'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]',
    warning: 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]',
    safe:    'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]',
  }
  return <div className={clsx('w-2 h-2 rounded-full shrink-0', colors[status] || colors.safe)} />
}

// ─── Utility ──────────────────────────────────────────
const card = 'bg-bg-100 border border-border rounded-2xl p-4 relative overflow-hidden'
const sectionTitle = 'text-[10px] uppercase tracking-[0.15em] text-text-muted mb-3 font-bold'

const REASON_MAP = {
  'Account fully drained to new recipient':        { rule: 'Drain → Unknown',  weight: '35%', color: '#ef4444' },
  "Amount significantly exceeds user's average":   { rule: 'Amt Deviation',    weight: '25%', color: '#f59e0b' },
  'Amount significantly exceeds user average':     { rule: 'Amt Deviation',    weight: '25%', color: '#f59e0b' },
  'High context risk (Foreign/New Device/IP)':     { rule: 'Risky Context',    weight: '20%', color: '#a855f7' },
  'High urgency/velocity signals detected':        { rule: 'Rapid Session',    weight: '20%', color: '#00d4ff' },
  'Normal behavior pattern':                       { rule: 'No Rules Fired',   weight: '—',   color: 'oklch(0.40 0.02 260)' },
}

function matchReason(str) {
  if (REASON_MAP[str]) return REASON_MAP[str]
  const key = Object.keys(REASON_MAP).find(k => str.includes(k) || k.includes(str))
  return key ? REASON_MAP[key] : null
}

function decisionColor(d) {
  if (d === 'APPROVE') return '#10b981'
  if (d === 'FLAG')    return '#f59e0b'
  return '#ef4444'
}

function riskColor(level) {
  if (level === 'HIGH')   return '#ef4444'
  if (level === 'MEDIUM') return '#f59e0b'
  return '#10b981'
}

// Map mock engine transactions to backend TransactionRequest schema
function toBackendSchema(t) {
  return {
    transaction_id:   t.id,
    amount:           t.amount  || 0,
    sender_id:        t.userId  || 'USR-UNKNOWN',
    receiver_id:      t.receiverId || 'USR-UNKNOWN',
    transaction_type: 'transfer',
    timestamp:        t.timestamp || new Date().toISOString(),
    
    // Derived values to ensure scoring matches
    avg_transaction_amount_30d: t.amount * 0.6,
    amount_vs_avg_ratio:        t.amount > 0 ? parseFloat(((t.amount / (t.amount * 0.6)) || 1).toFixed(2)) : 1,
    transaction_hour: new Date(t.timestamp).getHours(),
    is_weekend:       [0,6].includes(new Date(t.timestamp).getDay()) ? 1 : 0,
    is_new_device:    t.newDevice ? 1 : 0,
    failed_login_attempts: 0,
    is_proxy_ip:      t.vpnDetected ? 1 : 0,
    ip_risk_score:    t.vpnDetected ? 0.8 : 0.1,
    sender_account_fully_drained: t.riskFactors?.includes('Account fully drained to new recipient') ? 1 : 0,
    account_age_days: 120,
    tx_count_24h:     t.velocityFlag ? 8 : 1,
    country_mismatch: 0,
    is_new_recipient: 1, // trigger behavior rule
    established_user_new_recipient: 0,
  }
}

// ─── Loading Skeleton ──────────────────────────────────────────────────────────
function Skeleton({ className = '' }) {
  return (
    <div className={clsx('animate-pulse rounded bg-white/[0.06]', className)} />
  )
}

// ─── Score Range SVG ──────────────────────────────────────────────────────────
function ScoreRangeSVG({ score }) {
  const pct = Math.min(100, Math.max(0, score))
  const x = (pct / 100) * 360
  return (
    <svg width="360" height="44" viewBox="0 0 360 44" className="opacity-85">
      <rect x="0"   y="4" width="144" height="12" fill="#10b981" fillOpacity="0.18" rx="3" />
      <rect x="144" y="4" width="126" height="12" fill="#f59e0b" fillOpacity="0.18" />
      <rect x="270" y="4" width="90"  height="12" fill="#ef4444" fillOpacity="0.18" rx="3" />
      <rect x="144" y="4" width="1"   height="12" fill="oklch(0.35 0.02 260)" />
      <rect x="270" y="4" width="1"   height="12" fill="oklch(0.35 0.02 260)" />
      <polygon
        points={`${x},1 ${x-5},18 ${x+5},18`}
        fill={pct <= 40 ? '#10b981' : pct <= 70 ? '#f59e0b' : '#ef4444'}
      />
      <text x={x} y="30" fontSize="9" textAnchor="middle" fontFamily="JetBrains Mono, monospace"
        fill={pct <= 40 ? '#10b981' : pct <= 70 ? '#f59e0b' : '#ef4444'}>[{Math.round(pct)}]</text>
      <text x="0"   y="44" fontSize="9" fill="oklch(0.40 0.02 260)" fontFamily="JetBrains Mono, monospace">0</text>
      <text x="72"  y="44" fontSize="9" fill="#10b981" textAnchor="middle" fontFamily="JetBrains Mono, monospace">APPROVE</text>
      <text x="144" y="44" fontSize="9" fill="oklch(0.40 0.02 260)" textAnchor="middle" fontFamily="JetBrains Mono, monospace">40</text>
      <text x="207" y="44" fontSize="9" fill="#f59e0b" textAnchor="middle" fontFamily="JetBrains Mono, monospace">FLAG</text>
      <text x="270" y="44" fontSize="9" fill="oklch(0.40 0.02 260)" textAnchor="middle" fontFamily="JetBrains Mono, monospace">70</text>
      <text x="315" y="44" fontSize="9" fill="#ef4444" textAnchor="middle" fontFamily="JetBrains Mono, monospace">BLOCK</text>
      <text x="360" y="44" fontSize="9" fill="oklch(0.40 0.02 260)" textAnchor="end" fontFamily="JetBrains Mono, monospace">100</text>
    </svg>
  )
}

// ─── SHAP Waterfall Chart ─────────────────────────────────────────────────────
function ShapWaterfall({ topFeatures }) {
  const data = topFeatures.map(f => ({
    name: f.feature.replace(/_/g, ' '),
    fullName: f.feature,
    value: f.contribution ?? f.shap_value ?? 0,
    actual: f.actual_value ?? 'N/A',
  }))

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0].payload
    return (
      <div className="bg-zinc-900 border border-white/10 rounded p-2 font-mono text-[11px]">
        <div className="text-zinc-300">{d.fullName}</div>
        <div style={{ color: d.value >= 0 ? '#ef4444' : '#10b981' }}>
          SHAP: {d.value >= 0 ? '+' : ''}{d.value.toFixed(3)}
        </div>
        <div className="text-zinc-500">value: {typeof d.actual === 'boolean' ? String(d.actual) : d.actual}</div>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 60, left: 130, bottom: 4 }}
      >
        <XAxis
          type="number"
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fill: 'oklch(0.50 0.02 260)' }}
          tickFormatter={v => (v >= 0 ? `+${v.toFixed(1)}` : v.toFixed(1))}
          label={{
            value: '← Decreases Risk · Increases Risk →',
            position: 'insideBottom',
            offset: -2,
            style: { fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fill: 'oklch(0.45 0.02 260)' }
          }}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: 'oklch(0.65 0.02 260)' }}
          width={128}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine x={0} stroke="oklch(0.30 0.02 260)" strokeDasharray="3 3" />
        <Bar dataKey="value" radius={[0, 2, 2, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value >= 0 ? '#ef4444' : '#10b981'} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function TransactionInvestigation({ engine }) {
  const { allTransactions } = engine

  // ── Left column state ──
  const [search, setSearch]     = useState('')
  const [filter, setFilter]     = useState('ALL')
  const [sort, setSort]         = useState('Highest Risk')

  // ── Search extensions ──
  const [dbSearchResults, setDbSearchResults] = useState([])
  const [isSearchingDb, setIsSearchingDb] = useState(false)

  // ── Right column state ──
  const [selectedId, setSelectedId]   = useState(null)
  const [riskResult, setRiskResult]   = useState(null)
  const [shapResult, setShapResult]   = useState(null)
  const [scoring, setScoring]         = useState(false)
  const [decisions, setDecisions]     = useState({})
  const [feedbackLog, setFeedbackLog] = useState([])
  const [resolvingState, setResolvingState] = useState(null)   // 'APPROVE' | 'BLOCK' | null
  const [adminReason, setAdminReason] = useState('')
  const [feedbackAnim, setFeedbackAnim] = useState(null)       // 'pulsing' | 'done' | null
  const [impactSim, setImpactSim]     = useState(null)

  // ── LLM Investigation ──
  const [llmQuery, setLlmQuery]       = useState('')
  const [llmResponse, setLlmResponse] = useState(null)
  const [llmLoading, setLlmLoading]   = useState(false)
  const [llmStatus, setLlmStatus]     = useState(null)  // { available, model, fallback }
  const [llmExpanded, setLlmExpanded]  = useState(true)

  // ── Engine health ──
  const [engineOnline, setEngineOnline] = useState(true)
  const [healthLoading, setHealthLoading] = useState(true)

  useEffect(() => {
    let interval
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/health`)
        const d = await res.json()
        setEngineOnline(d.status === 'ok' && d.engine_loaded)
      } catch {
        setEngineOnline(false)
      } finally {
        setHealthLoading(false)
      }
    }
    check()
    interval = setInterval(check, 10000)
    return () => clearInterval(interval)
  }, [])

  // ── LLM Status Check ──
  useEffect(() => {
    const checkLlm = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/llm/status`)
        if (res.ok) setLlmStatus(await res.json())
      } catch { setLlmStatus(null) }
    }
    checkLlm()
  }, [])

  // ── Debounced DB Search ──
  useEffect(() => {
    if (search.length < 3) {
      setDbSearchResults([])
      setIsSearchingDb(false)
      return
    }

    const abortController = new AbortController()
    const timer = setTimeout(async () => {
      setIsSearchingDb(true)
      try {
        const res = await fetch(`${API_BASE}/api/v1/transactions/search?q=${encodeURIComponent(search)}`, {
          signal: abortController.signal
        })
        if (res.ok) {
          const data = await res.json()
          const mapped = data.map(dbTx => ({
            ...dbTx,
            id: dbTx.transaction_id,
            userId: dbTx.user_hash,
            receiverId: dbTx.recipient_hash,
            amount: dbTx.amount,
            timestamp: new Date().toISOString(),
            decision: dbTx.action_taken,
            ensembleScore: dbTx.ml_risk_score,
            isFraud: dbTx.is_fraud === 1,
            scoredByBackend: true,
          }))
          setDbSearchResults(mapped)
        }
      } catch (err) {
        if (err.name !== 'AbortError') console.error('DB search fail:', err)
      } finally {
        setIsSearchingDb(false)
      }
    }, 300)

    return () => {
      clearTimeout(timer)
      abortController.abort()
    }
  }, [search])

  // ── Derived left list ──
  const mappedTxns = useMemo(() => {
    const mergedMap = new Map()
    allTransactions.forEach(t => mergedMap.set(t.id, t))
    dbSearchResults.forEach(t => {
      if (!mergedMap.has(t.id)) mergedMap.set(t.id, t)
    })
    const mergedList = Array.from(mergedMap.values())

    return mergedList.map(t => {
      const ov = decisions[t.id]
      return {
        ...t,
        currentDecision: ov ? ov.decision : t.decision,
        resolved: !!ov,
      }
    })
  }, [allTransactions, dbSearchResults, decisions])

  const filteredTxns = useMemo(() => {
    let d = mappedTxns
    if (search) {
      const q = search.toLowerCase()
      d = d.filter(t =>
        t.id.toLowerCase().includes(q) ||
        (t.userId || '').toLowerCase().includes(q) ||
        (t.receiverId || '').toLowerCase().includes(q)
      )
    }
    if (filter !== 'ALL') {
      const m = { FLAGGED: 'FLAG', BLOCKED: 'BLOCK', APPROVED: 'APPROVE' }
      d = d.filter(t => t.decision === (m[filter] || filter))
    }
    d = [...d].sort((a, b) => {
      if (sort === 'Highest Risk')  return b.ensembleScore - a.ensembleScore
      if (sort === 'Lowest Risk')   return a.ensembleScore - b.ensembleScore
      if (sort === 'Highest Amount') return b.amount - a.amount
      return b.timestamp - a.timestamp
    })
    return d
  }, [mappedTxns, search, filter, sort])

  const flaggedCount    = useMemo(() => allTransactions.filter(t => t.decision === 'FLAG').length, [allTransactions])
  const resolvedCount   = useMemo(() => Object.keys(decisions).length, [decisions])
  const feedbackBufCount = feedbackLog.length

  const selected = useMemo(
    () => mappedTxns.find(t => t.id === selectedId) || null,
    [mappedTxns, selectedId]
  )

  // ── On transaction select: fetch /predict + /explain in parallel ──
  const handleSelect = useCallback(async (txn) => {
    setSelectedId(txn.id)
    setRiskResult(null)
    setShapResult(null)
    setResolvingState(null)
    setAdminReason('')
    setFeedbackAnim(null)
    setImpactSim(null)
    setLlmResponse(null)
    setLlmQuery('')

    if (!engineOnline) return

    setScoring(true)
    const body = toBackendSchema(txn)
    try {
      const [riskRes, shapRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/score-transaction`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        }).then(r => r.ok ? r.json() : null).catch(() => null),

        fetch(`${API_BASE}/api/v1/explain/${txn.id}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        }).then(r => r.ok ? r.json() : null).catch(() => null),
      ])
      setRiskResult(riskRes)
      setShapResult(shapRes)
    } catch (e) {
      console.error(e)
    } finally {
      setScoring(false)
    }
  }, [engineOnline])

  // ── Admin decision ──
  const handleResolve = (action) => {
    if (!adminReason.trim() || !selected) return
    const original = selected.decision
    setFeedbackAnim('pulsing')
    setTimeout(() => {
      setDecisions(prev => ({
        ...prev,
        [selected.id]: { decision: action, reason: adminReason, timestamp: new Date(), original }
      }))
      const simCount   = Object.values(decisions).filter(d => d.original === 'FLAG').length + 1
      const delta      = parseFloat((Math.random() * 2.9 + 0.1).toFixed(1)) * (action === 'BLOCK' ? -1 : 1)
      setImpactSim({ similarCount: simCount, thresholdDelta: delta })
      setFeedbackLog(prev => [...prev, { tx_id: selected.id, label: action, timestamp: new Date() }])
      setFeedbackAnim('done')
      setAdminReason('')
    }, 1500)
  }

  // ── LLM Investigate Handler (must be after `selected` and `riskResult` are defined) ──
  const handleInvestigate = useCallback(async (quickQuery) => {
    const query = quickQuery || llmQuery
    if (!query.trim() || !selected) return
    setLlmLoading(true)
    setLlmResponse(null)
    try {
      const body = {
        query,
        transaction_id: selected.id || selected.transaction_id,
        context: riskResult ? {
          ...riskResult,
          feature_snapshot: riskResult.feature_snapshot || {},
          rule_breakdown: riskResult.rule_breakdown || {},
        } : null,
      }
      const res = await fetch(`${API_BASE}/api/v1/investigate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (res.ok) {
        setLlmResponse(await res.json())
      } else {
        setLlmResponse({ response: 'Failed to get analysis. Please try again.', status: 'error' })
      }
    } catch (e) {
      console.error('LLM investigate error:', e)
      setLlmResponse({ response: 'Connection error. Is the backend running?', status: 'error' })
    } finally {
      setLlmLoading(false)
    }
  }, [llmQuery, selected, riskResult])

  // ── Derived risk values from backend OR mock fallback ──
  const riskScore = riskResult
    ? Math.round(riskResult.risk_score * 100)
    : selected ? Math.round(selected.ensembleScore * 100) : 0

  const riskLevel = riskResult?.decision === 'BLOCK' ? 'HIGH'
    : riskResult?.decision === 'FLAG' ? 'MEDIUM'
    : riskResult?.decision === 'APPROVE' ? 'LOW'
    : selected?.risk_level || (selected?.decision === 'BLOCK' ? 'HIGH' : selected?.decision === 'FLAG' ? 'MEDIUM' : 'LOW')

  // ── Decision for admin panel (use override if any, else backend, else mock) ──
  const adminDecision = selected
    ? (decisions[selected.id]?.decision || riskResult?.decision || selected.decision)
    : 'APPROVE'
  const isResolved = selected ? !!decisions[selected.id] : false
  const isFlag     = adminDecision === 'FLAG' && !isResolved

  const lgbScore  = riskResult ? riskResult.supervised_score     : selected?.lgbScore  || 0
  const isoScore  = riskResult ? riskResult.unsupervised_score   : selected?.xgbScore || 0
  const behScore  = riskResult ? riskResult.behavioral_score     : 0
  const reasons   = riskResult?.reasons?.length ? riskResult.reasons : (selected?.riskFactors || [])
  const privacy   = riskResult?.privacy

  // --- Feature Snapshot & Detailed Metrics ---
  const fs = riskResult?.feature_snapshot || {}
  const amtRatio = fs.amount_vs_avg_ratio ?? selected?.amountVsAvgRatio ?? selected?.amount_vs_avg_ratio ?? 1
  const ipRisk = fs.ip_risk_score ?? selected?.ipRiskScore ?? selected?.ip_risk_score ?? 0
  const txCount = fs.tx_count_24h ?? selected?.txCount24h ?? selected?.tx_count_24h ?? 0
  const sessionDur = fs.session_duration_seconds ?? selected?.sessionDurationSeconds ?? selected?.session_duration_seconds ?? 0
  const isNewDev = fs.is_new_device ?? selected?.isNewDevice ?? selected?.is_new_device ?? 0
  const countryMM = fs.country_mismatch ?? selected?.countryMismatch ?? selected?.country_mismatch ?? 0
  const senderDrained = fs.sender_fully_drained ?? selected?.senderFullyDrained ?? selected?.sender_account_fully_drained ?? 0
  const isNewRecip = fs.is_new_recipient ?? selected?.isNewRecipient ?? selected?.is_new_recipient ?? 0
  const accountAge = fs.account_age_days ?? selected?.accountAgeDays ?? selected?.account_age_days ?? 0
  const failedLogins = selected?.failed_login_attempts ?? 0

  // Balances
  const balBefore = selected?.sender_balance_before ?? 10000.00
  const balAfter = selected?.sender_balance_after ?? (balBefore - (selected?.amount || 0))
  const balChangePct = balBefore > 0 ? ((balAfter - balBefore) / balBefore * 100) : 0

  const isFraud = selected?.isFraud ?? false
  const groundTruth = isFraud ? "FRAUD" : "LEGIT"
  const modelPredictedFraud = adminDecision !== 'APPROVE'

  let verdict, verdictColor, verdictIcon
  if (modelPredictedFraud && isFraud) { verdict = "TRUE POSITIVE"; verdictColor = "#10b981"; verdictIcon = "✅" }
  else if (!modelPredictedFraud && !isFraud) { verdict = "TRUE NEGATIVE"; verdictColor = "#10b981"; verdictIcon = "✅" }
  else if (modelPredictedFraud && !isFraud) { verdict = "FALSE POSITIVE"; verdictColor = "#f59e0b"; verdictIcon = "⚠" }
  else { verdict = "FALSE NEGATIVE"; verdictColor = "#ef4444"; verdictIcon = "✗" }

  const channelMap = {
    'cash_out': 'ATM / Withdrawal', 'CASH_OUT': 'ATM / Withdrawal',
    'transfer': 'P2P Transfer',      'TRANSFER': 'P2P Transfer',
    'payment': 'Merchant Payment',   'PAYMENT': 'Merchant Payment',
  }
  const txType = selected?.transfer_type || selected?.transaction_type || 'TRANSFER'
  const channel = channelMap[txType] || 'P2P Transfer'

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

  const featureRows = [
    { param: 'amount_vs_avg_ratio',   value: `${amtRatio.toFixed(2)}×`, threshold: '> 1.5×', status: amtRatio > 5 ? 'danger' : amtRatio > 1.5 ? 'warning' : 'safe' },
    { param: 'sender_account_drained', value: senderDrained ? 'YES' : 'NO', threshold: '—', status: senderDrained ? 'danger' : 'safe' },
    { param: 'ip_risk_score',         value: ipRisk.toFixed(3), threshold: '> 0.5', status: ipRisk > 0.7 ? 'danger' : ipRisk > 0.5 ? 'warning' : 'safe' },
    { param: 'country_mismatch',      value: countryMM ? 'YES (foreign)' : 'NO (domestic)', threshold: '—', status: countryMM ? 'danger' : 'safe' },
    { param: 'is_new_device',         value: isNewDev ? 'YES' : 'NO', threshold: '—', status: isNewDev ? 'warning' : 'safe' },
    { param: 'tx_count_24h',          value: txCount.toString(), threshold: '> 5', status: txCount > 5 ? 'danger' : 'safe' },
    { param: 'session_duration_secs', value: `${Math.round(sessionDur)}s`, threshold: '< 60s', status: sessionDur < 60 ? 'danger' : 'safe' },
  ]
  const rulesFireCount = featureRows.filter(r => r.status !== 'safe').length
  const behContribution = behScore > 0.5 ? 'HIGH' : behScore > 0.2 ? 'MEDIUM' : 'LOW'


  return (
    <div className="flex gap-4 h-[calc(100vh-112px)] max-w-[1500px] mx-auto relative font-mono bg-[#06090e]">

      {/* ── Offline Banner ── */}
      {!healthLoading && !engineOnline && (
        <div className="absolute top-[-52px] left-0 right-0 z-50 bg-[#0d1a12] border border-emerald-900/60 text-emerald-400 px-4 py-2 rounded-xl flex items-center justify-center gap-2 text-[12px] uppercase tracking-wider">
          <AlertTriangle size={14} />
          ⚠ Risk Engine offline — live scoring unavailable
        </div>
      )}

      {/* ══════════════════ LEFT COLUMN ══════════════════ */}
      <div className="w-[300px] shrink-0 flex flex-col gap-3">
        <div className="bg-bg-100 border border-border rounded-2xl flex flex-col gap-3 h-full overflow-hidden p-4">

          {/* Search */}
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-cyan-500" size={15} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search TX ID or Sender/Receiver..."
              className="w-full bg-bg-50 border border-border rounded-xl py-2 pl-9 pr-9 text-[12px] text-text-primary focus:outline-none focus:ring-1 focus:ring-cyan-500/50 placeholder:text-text-muted"
            />
            {isSearchingDb && <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 text-cyan-500 animate-spin" size={14} />}
          </div>

          {/* Filter Pills */}
          <div className="flex gap-1.5 flex-wrap">
            {[
              { k: 'ALL',      label: 'All' },
              { k: 'FLAGGED',  label: `Flagged (${flaggedCount})` },
              { k: 'BLOCKED',  label: 'Blocked' },
              { k: 'APPROVED', label: 'Approved' },
            ].map(({ k, label }) => (
              <button
                key={k}
                onClick={() => setFilter(k)}
                className={clsx(
                  'px-2.5 py-1 rounded-full border text-[10px] uppercase tracking-wider transition-all',
                  filter === k
                    ? 'bg-cyan-500/10 border-cyan-500/50 text-cyan-400'
                    : 'border-border text-text-muted hover:text-text-primary hover:border-border-md'
                )}
              >{label}</button>
            ))}
          </div>

          {/* Sort */}
          <select
            value={sort}
            onChange={e => setSort(e.target.value)}
            className="bg-bg-50 border border-border rounded-lg px-3 py-1.5 text-[11px] text-text-secondary focus:outline-none uppercase"
          >
            <option>Highest Risk</option>
            <option>Lowest Risk</option>
            <option>Most Recent</option>
            <option>Highest Amount</option>
          </select>

          {/* Transaction Rows */}
          <div className="flex-1 overflow-y-auto -mx-1 px-1 custom-scrollbar space-y-1">
            {allTransactions.length === 0 ? (
              <div className="text-center text-zinc-600 text-[12px] py-8">
                No transactions scored yet — start the simulator
              </div>
            ) : filteredTxns.length === 0 ? (
              <div className="text-center text-zinc-600 text-[12px] py-8">No matching transactions</div>
            ) : filteredTxns.map(t => {
              const isSel    = selectedId === t.id
              const isFlag_  = t.decision === 'FLAG' && !t.resolved
              return (
                <button
                  key={t.id}
                  onClick={() => handleSelect(t)}
                  className={clsx(
                    'w-full text-left px-3 py-2.5 rounded-xl border transition-all',
                    isSel
                      ? 'bg-bg-200 border-cyan-500/60 shadow-[inset_2px_0_0_0_#00e5ff]'
                      : isFlag_
                        ? 'bg-[#f59e0b08] border-border shadow-[inset_2px_0_0_0_#f59e0b]'
                        : 'bg-bg-50 border-border hover:bg-bg-100'
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: decisionColor(t.currentDecision) }} />
                      <span className="text-[12px] font-bold text-zinc-200">{t.id}</span>
                    </div>
                    <span className="text-[11px] text-zinc-400">{formatCurrency(t.amount)}</span>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-zinc-500">
                    <span className="truncate max-w-[130px]">{t.userId} → {t.receiverId || 'USR-UNK'}</span>
                    <span className="flex items-center gap-1.5 shrink-0">
                      <span style={{ color: decisionColor(t.currentDecision) }} className="font-bold">
                        {Math.round(t.ensembleScore * 100)}
                      </span>
                      {t.resolved && <span className="text-cyan-600 bg-cyan-500/10 px-1 rounded">✏ RES</span>}
                    </span>
                  </div>
                </button>
              )
            })}
          </div>

          {/* Footer */}
          <div className="pt-2 border-t border-border text-[10px] text-text-muted space-y-0.5">
            <div>{allTransactions.length} transactions · {flaggedCount} pending review · {resolvedCount} resolved</div>
            <div>Model feedback buffer: {feedbackBufCount} label{feedbackBufCount !== 1 ? 's' : ''} queued</div>
          </div>
        </div>
      </div>

      {/* ══════════════════ RIGHT COLUMN ══════════════════ */}
      <div className="flex-1 overflow-y-auto custom-scrollbar pr-1">
        {!selected ? (
          <div className="h-full flex flex-col items-center justify-center text-text-muted rounded-2xl border border-border bg-bg-100">
            <FileText size={48} className="mb-4 opacity-20" />
            <p className="text-[16px] font-bold text-text-secondary">Select a transaction to investigate</p>
            <p className="text-[12px] text-text-muted mt-1">FLAG transactions require admin review</p>
          </div>
        ) : (
          <div className="space-y-3 pb-6">

            {/* ═══ SECTION 1: TRANSACTION PROFILE ═══ */}
            <div className={card}>
              <div className={sectionTitle}>Transaction Profile</div>
              <div className="flex items-center gap-2 mb-3">
                <div className="flex items-center gap-1.5 px-2 py-0.5 bg-[#10b981]/10 border border-[#10b981]/20 rounded text-[#10b981] font-mono text-[9px] font-bold tracking-widest">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#10b981] animate-pulse" />
                  LIVE
                </div>
                <span className="font-mono text-[11px] text-text-primary font-bold">
                  {(selected.transaction_id || selected.id || 'UNKNOWN').toUpperCase()}
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
                {new Date(selected.timestamp).toISOString().split('T')[0]} · {new Date(selected.timestamp).toTimeString().split(' ')[0]}
              </div>

              {/* Amount */}
              <div className="bg-bg-200/50 p-4 rounded-xl border border-border/50 mb-4">
                <div className="font-mono text-[32px] font-bold text-text-primary leading-none tracking-tight mb-4">
                  {formatCurrency(selected.amount)}
                </div>

                {/* SENDER Block */}
                <div className="mb-3 p-3 rounded-lg bg-bg-300/30 border border-border/30">
                  <div className="font-mono text-[9px] text-text-muted/60 uppercase tracking-widest mb-1">SENDER</div>
                  <div className="font-mono text-[12px] text-cyan-400 font-bold mb-1">{selected.userId}</div>
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
                  <div className="font-mono text-[12px] text-cyan-400 font-bold mb-1">{selected.receiverId || 'USR-UNKNOWN'}</div>
                  {isNewRecip ? (
                    <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/15 border border-amber-500/30 text-amber-400 font-mono text-[9px] font-bold tracking-widest">
                      ⚠ FIRST-TIME RECIPIENT
                    </div>
                  ) : (
                    <div className="font-mono text-[10px] text-text-muted/50">Known recipient</div>
                  )}
                </div>
              </div>
            </div>

            {/* ── Card 2: Risk Summary (Gauge & Ensemble) ── */}
            <div className={card}>
                <div className={sectionTitle}>Score & Ensemble Breakdown</div>
                {scoring ? (
                   <div className="space-y-3">
                     <Skeleton className="h-8 w-[280px]" />
                     <Skeleton className="h-20 w-full" />
                   </div>
                ) : (
                  <>
                    <div className="flex flex-col items-center mb-6">
                        <ScoreRangeSVG score={riskScore} />
                    </div>

                    <div className="flex flex-col gap-3 mb-6">
                        {[
                          { label: 'LIGHTGBM', pct: '55%', val: lgbScore, color: '#4FC3F7' },
                          { label: 'ISOFOREST', pct: '25%', val: isoScore, color: '#ce93d8' },
                          { label: 'BEHAVIORAL', pct: '20%', val: behScore, color: '#ffb74d' }
                        ].map((bar, idx) => (
                          <div key={idx} className="flex items-center gap-3">
                            <div className="flex flex-col w-[110px] shrink-0">
                              <span className="font-mono text-[10px] tracking-widest text-text-secondary">{bar.label}</span>
                            </div>
                            <div className="flex-1 h-[6px] bg-white/5 rounded-full relative overflow-hidden">
                              <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ backgroundColor: bar.color, width: `${bar.val * 100}%` }} />
                            </div>
                            <div className="w-[30px] text-right font-mono text-[11px] font-bold" style={{ color: bar.color }}>{(bar.val * 100).toFixed(0)}</div>
                          </div>
                        ))}
                    </div>

                  </>
                )}
            </div>

            {/* ═══ SECTION 2: FEATURE PARAMETERS TABLE ═══ */}
            <div className={card}>
              <div className={sectionTitle}>Feature Parameters</div>
              <div className="rounded-xl border border-border/30 overflow-hidden">
                <div className="grid grid-cols-[1fr_80px_60px_36px] gap-1 px-3 py-2 bg-bg-300/30 font-mono text-[8px] text-text-muted/60 uppercase tracking-widest">
                  <span>Parameter</span>
                  <span>Value</span>
                  <span>Threshold</span>
                  <span className="text-center">Status</span>
                </div>
                {featureRows.map((row, i) => (
                  <div key={row.param} className={clsx("grid grid-cols-[1fr_80px_60px_36px] gap-1 px-3 py-1.5 font-mono text-[10px] border-t border-border/10 transition-colors hover:bg-red-500/[0.03]", i % 2 === 1 && 'bg-bg-200/20')}>
                    <span className="text-text-muted truncate">{row.param}</span>
                    <span className={clsx("font-bold", row.status === 'danger' ? 'text-red-400' : row.status === 'warning' ? 'text-amber-400' : 'text-text-secondary')}>{row.value}</span>
                    <span className="text-text-muted/50">{row.threshold}</span>
                    <div className="flex justify-center items-center"><StatusDot status={row.status} /></div>
                  </div>
                ))}
              </div>
              <div className="mt-2 font-mono text-[10px] text-text-muted/50 text-right">
                {rulesFireCount} rules fired · Behavioral: <span className={clsx("font-bold", behContribution === 'HIGH' ? 'text-red-400' : behContribution === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400')}>{behContribution}</span>
              </div>
            </div>

            {/* ── Card 3: Triggered Rules ── */}
            <div className={card}>
              <div className={sectionTitle}>Triggered Rules</div>
              {scoring ? (
                <div className="space-y-2">
                  {[1,2].map(i => <Skeleton key={i} className="h-14 w-full" />)}
                </div>
              ) : (
                <>
                  <div className="space-y-2 mb-3">
                    {(reasons.length > 0 ? reasons : ['Normal behavior pattern']).map((r, i) => {
                      const mapped = matchReason(r) || { rule: r, weight: '—', color: 'oklch(0.40 0.02 260)' }
                      return (
                        <div key={i} className="border-l-[3px] pl-3 py-1" style={{ borderColor: mapped.color }}>
                          <div className="flex justify-between items-center">
                            <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: mapped.color }}>{mapped.rule}</span>
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full" style={{ color: mapped.color, backgroundColor: `${mapped.color}20` }}>{mapped.weight}</span>
                          </div>
                          <div className="text-[11px] text-zinc-500 mt-0.5">{r}</div>
                        </div>
                      )
                    })}
                  </div>
                  <div className="pt-2 border-t border-[#1a2a3a] text-[10px] text-zinc-600 space-y-1">
                    <div className="flex items-center gap-2"><Lock size={9} />PII hashed with SHA-256</div>
                    {privacy?.dp_applied && <div className="flex items-center gap-2"><Lock size={9} />Differential privacy applied</div>}
                  </div>
                </>
              )}
            </div>

            {/* ── Card 4: SHAP Explainability ── */}
            <div className={card}>
              <div className={sectionTitle}>Ensemble SHAP Explanation</div>
              {scoring ? (
                <div className="space-y-2">
                  {[1,2,3,4,5].map(i => <Skeleton key={i} className="h-6 w-full" />)}
                </div>
              ) : !engineOnline ? (
                <div className="text-[12px] text-zinc-600 py-4 text-center">SHAP unavailable — start backend</div>
              ) : !shapResult ? (
                <div className="text-[12px] text-zinc-600 py-4 text-center">Loading explanation...</div>
              ) : (
                <>
                  <ShapWaterfall topFeatures={shapResult.top_features} />
                  <div className="mt-4 pt-4 border-t border-white/[0.06]">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-zinc-600 mb-3">Behavioral Rule Decomposition</div>
                    <div className="space-y-2">
                      {[
                        { key: 'drain_score',      label: 'Drain → Unknown',  color: '#ef4444' },
                        { key: 'deviation_score',  label: 'Amt Deviation',    color: '#f59e0b' },
                        { key: 'context_score',          label: 'Risky Context',    color: '#a855f7' },
                        { key: 'velocity_score',          label: 'Rapid Session',    color: '#00d4ff' },
                      ].map(({ key, label, color }) => {
                        const val = riskResult?.rule_breakdown?.[key] ?? 0
                        const pct = Math.min(100, val * 100)
                        return (
                          <div key={key} className="flex items-center gap-3 text-[11px]">
                            <span className="w-[110px] shrink-0" style={{ color }}>{label}</span>
                            <div className="flex-1 bg-white/[0.06] h-1.5 rounded-full overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.7 }} />
                            </div>
                            <span className="text-zinc-500 w-16 text-right">{val.toFixed(2)} pts</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* ═══ SECTION 3: GROUND TRUTH REFERENCE ═══ */}
            <div className={card} style={{ borderLeft: `3px solid ${isFraud ? '#ff4d6d' : '#00e5a0'}` }}>
              <div className={sectionTitle}>Ground Truth Reference</div>
              <div className="flex items-center gap-2 mb-3">
                <span className="font-mono text-[10px] text-text-muted/60 uppercase tracking-widest">True Label:</span>
                <span className={clsx("font-mono text-[12px] font-bold", isFraud ? 'text-red-400' : 'text-emerald-400')}>
                  {isFraud ? '🔴 FRAUD' : '✅ LEGIT'}
                </span>
              </div>

              <div className="flex items-center gap-2 px-3 py-2 rounded-lg mb-4 border" style={{ backgroundColor: `${verdictColor}10`, borderColor: `${verdictColor}30` }}>
                <span className="text-[18px]">{verdictIcon}</span>
                <span className="font-mono text-[14px] font-bold tracking-wider" style={{ color: verdictColor }}>{verdict}</span>
              </div>

              <div className="font-mono text-[10px] text-text-muted/60 mb-3">
                {verdict === 'TRUE POSITIVE'  ? 'Fraud correctly blocked by model.' :
                 verdict === 'TRUE NEGATIVE'  ? 'Legitimate transaction correctly approved.' :
                 verdict === 'FALSE POSITIVE' ? 'Legit transaction incorrectly flagged.' :
                 'Fraudulent transaction missed by model.'}
              </div>

              <div className="font-mono text-[9px] text-text-muted/50 uppercase tracking-widest mb-2">Evidence Bullets:</div>
              <ul className="space-y-1.5">
                {gtBullets.map((b, i) => (
                  <li key={i} className="flex items-start gap-2 font-sans text-[11px] text-text-secondary leading-snug">
                    <div className={clsx("w-1.5 h-1.5 rounded-full shrink-0 mt-1.5", isFraud ? 'bg-red-400' : 'bg-emerald-400')} />
                    {b}
                  </li>
                ))}
              </ul>
            </div>

            {/* ── Card 5: Admin Decision Panel ── */}
            <div className={card}>
              <div className={sectionTitle}>Admin Decision Panel</div>
              {!isFlag && !isResolved && (
                <div className="opacity-60">
                   <div className="flex items-center gap-2 text-zinc-300 font-bold text-[13px] mb-3"><Lock size={15} />DECISION LOCKED</div>
                   <p className="text-[11px] text-zinc-500 mb-4 leading-relaxed">Only pending FLAG transactions require review.</p>
                </div>
              )}
              {isResolved && (
                <div className="border border-cyan-500/20 rounded-xl p-3 bg-cyan-500/[0.04] flex items-start gap-3">
                  <Check size={16} className="text-cyan-400 shrink-0 mt-0.5" />
                  <div className="font-mono">
                    <div className="text-cyan-400 font-bold text-[11px] uppercase mb-1">Resolved by Admin</div>
                    <div className="text-[11px] text-zinc-300 italic">"{decisions[selected.id].reason}"</div>
                  </div>
                </div>
              )}
              {isFlag && (
                <div className="border border-amber-500/30 bg-amber-500/[0.04] rounded-xl p-4">
                  {!resolvingState ? (
                    <div className="flex gap-3">
                      <button onClick={() => setResolvingState('APPROVE')} className="flex-1 bg-emerald-500/10 text-emerald-400 font-bold text-[11px] py-2 rounded-lg border border-emerald-500/30">APPROVE</button>
                      <button onClick={() => setResolvingState('BLOCK')} className="flex-1 bg-red-500/10 text-red-400 font-bold text-[11px] py-2 rounded-lg border border-red-500/30">BLOCK</button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <input autoFocus value={adminReason} onChange={e => setAdminReason(e.target.value)} placeholder="Reason for decision..." className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg p-2 text-[12px] text-zinc-200 focus:outline-none" />
                      <div className="flex gap-3">
                        <button onClick={() => handleResolve(resolvingState)} disabled={!adminReason.trim()} className="flex-1 bg-cyan-500 text-zinc-900 font-bold text-[11px] py-2 rounded-lg">Confirm</button>
                        <button onClick={() => setResolvingState(null)} className="px-4 bg-white/[0.06] text-zinc-300 text-[11px] py-2 rounded-lg">Cancel</button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Card 6: AI Investigation Assistant ── */}
            <div className={clsx(card, 'border-purple-500/20')}>
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={sectionTitle + ' mb-0'}>AI Investigation Assistant</div>
                  {llmStatus?.available ? (
                    <span className="text-[8px] font-mono font-bold uppercase tracking-widest px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                      {llmStatus.model}
                    </span>
                  ) : (
                    <span className="text-[8px] font-mono font-bold uppercase tracking-widest px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400">
                      Fallback
                    </span>
                  )}
                </div>
                <button
                  onClick={() => setLlmExpanded(v => !v)}
                  className="text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  {llmExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                </button>
              </div>

              {llmExpanded && (
                <div className="space-y-3">
                  {/* Quick Actions */}
                  <div className="flex flex-wrap gap-1.5">
                    {[
                      { label: 'Why this score?', q: 'Why did this transaction receive this risk score? Explain the key contributing factors.' },
                      { label: 'Fraud pattern?', q: 'Does this transaction match any known fraud patterns like account takeover, phishing, or money mule operations?' },
                      { label: 'Next steps', q: 'What concrete investigation steps should I take for this transaction?' },
                      { label: 'False positive?', q: 'Could this be a false positive? What evidence supports or contradicts the fraud classification?' },
                    ].map(({ label, q }) => (
                      <button
                        key={label}
                        onClick={() => { setLlmQuery(q); handleInvestigate(q) }}
                        disabled={llmLoading || !engineOnline}
                        className="px-2 py-1 rounded-lg border text-[9px] uppercase tracking-wider font-bold transition-all
                          bg-purple-500/5 border-purple-500/20 text-purple-300 hover:bg-purple-500/15 hover:border-purple-500/40
                          disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  {/* Free-form Query */}
                  <div className="flex gap-2">
                    <input
                      value={llmQuery}
                      onChange={e => setLlmQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleInvestigate()}
                      placeholder="Ask about this transaction..."
                      disabled={llmLoading || !engineOnline}
                      className="flex-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-[12px] text-zinc-200
                        placeholder:text-zinc-600 focus:outline-none focus:border-purple-500/40 disabled:opacity-40"
                    />
                    <button
                      onClick={() => handleInvestigate()}
                      disabled={llmLoading || !llmQuery.trim() || !engineOnline}
                      className="px-4 py-2 rounded-lg font-mono text-[10px] font-bold uppercase tracking-wider transition-all
                        bg-gradient-to-r from-purple-600/80 to-cyan-600/80 text-white shadow-[0_0_12px_rgba(168,85,247,0.2)]
                        hover:shadow-[0_0_20px_rgba(168,85,247,0.4)]
                        disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none"
                    >
                      {llmLoading ? <Loader2 size={14} className="animate-spin" /> : 'Ask AI'}
                    </button>
                  </div>

                  {/* Response */}
                  {llmLoading && (
                    <div className="space-y-2 py-2">
                      <div className="flex items-center gap-2 text-purple-400 text-[11px] font-bold">
                        <Loader2 size={12} className="animate-spin" />
                        Analyzing transaction with {llmStatus?.model || 'AI'}...
                      </div>
                      <Skeleton className="h-4 w-full" />
                      <Skeleton className="h-4 w-4/5" />
                      <Skeleton className="h-4 w-3/5" />
                    </div>
                  )}

                  {llmResponse && !llmLoading && (
                    <div className="space-y-2">
                      <div className={clsx(
                        'rounded-xl p-4 border text-[12px] leading-relaxed font-sans whitespace-pre-wrap',
                        llmResponse.status === 'success'
                          ? 'bg-purple-500/[0.04] border-purple-500/20 text-zinc-300'
                          : llmResponse.status === 'unavailable'
                          ? 'bg-amber-500/[0.04] border-amber-500/20 text-zinc-300'
                          : 'bg-red-500/[0.04] border-red-500/20 text-zinc-400'
                      )}>
                        {/* Simple markdown-like rendering */}
                        {llmResponse.response.split('\n').map((line, i) => {
                          if (line.startsWith('## ')) return <div key={i} className="text-[13px] font-bold text-zinc-200 mt-3 mb-1">{line.slice(3)}</div>
                          if (line.startsWith('**') && line.endsWith('**')) return <div key={i} className="font-bold text-zinc-200 mt-2">{line.slice(2, -2)}</div>
                          if (line.startsWith('- ')) return <div key={i} className="flex gap-2 ml-2"><span className="text-purple-400 shrink-0">•</span><span>{line.slice(2)}</span></div>
                          if (line.trim() === '') return <div key={i} className="h-2" />
                          return <div key={i}>{line}</div>
                        })}
                      </div>

                      {/* Meta */}
                      <div className="flex items-center justify-between text-[9px] font-mono text-zinc-600 uppercase tracking-widest">
                        <div className="flex items-center gap-3">
                          <span>Model: {llmResponse.model_used}</span>
                          {llmResponse.tokens_used && <span>· {llmResponse.tokens_used} tokens</span>}
                        </div>
                        <span className={clsx(
                          'px-1.5 py-0.5 rounded',
                          llmResponse.status === 'success' ? 'bg-emerald-500/10 text-emerald-400' :
                          llmResponse.status === 'unavailable' ? 'bg-amber-500/10 text-amber-400' :
                          'bg-red-500/10 text-red-400'
                        )}>
                          {llmResponse.status}
                        </span>
                      </div>
                    </div>
                  )}

                  {!llmResponse && !llmLoading && (
                    <div className="text-center py-4 text-[11px] text-zinc-600">
                      <Info size={16} className="inline mr-1.5 opacity-50" />
                      Select a quick action or type a question to investigate this transaction
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Card 7: Model Feedback Loop ── */}
            {(feedbackAnim === 'pulsing' || (feedbackAnim === 'done' && impactSim)) && (
              <div className={clsx(card, 'transition-all duration-300', feedbackAnim === 'pulsing' ? 'border-amber-500/50 animate-pulse' : 'border-emerald-500/30')}>
                {feedbackAnim === 'pulsing' ? (
                   <div className="flex items-center gap-2 text-amber-400 font-bold text-[11px]"><Loader2 size={14} className="animate-spin" /> Retraining Simulation...</div>
                ) : (
                  <div className="space-y-3 text-[11px]">
                    <div className="text-emerald-400 font-bold">Feedback Received</div>
                    <div className="flex justify-between border-t border-white/5 pt-2">
                       <span className="text-zinc-500">Threshold Adjustment:</span>
                       <span className={clsx('font-bold', impactSim.thresholdDelta < 0 ? 'text-red-400' : 'text-emerald-400')}>{impactSim.thresholdDelta > 0 ? '+' : ''}{impactSim.thresholdDelta} pts</span>
                    </div>
                  </div>
                )}
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  )
}
