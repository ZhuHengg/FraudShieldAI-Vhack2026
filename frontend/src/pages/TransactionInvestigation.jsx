import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  Search as SearchIcon, AlertTriangle, ShieldAlert, Check, X,
  Lock, Loader2, Info, ChevronDown, ChevronUp, FileText
} from 'lucide-react'
import clsx from 'clsx'
import { formatCurrency } from '../utils/formatters'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts'

const API_BASE = 'http://localhost:8000'

// ─── Utility ──────────────────────────────────────────────────────────────────
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
    value: f.shap_value,
    actual: f.actual_value,
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

  // ── Derived left list ──
  const mappedTxns = useMemo(() => {
    return allTransactions.map(t => {
      const ov = decisions[t.id]
      return {
        ...t,
        currentDecision: ov ? ov.decision : t.decision,
        resolved: !!ov,
      }
    })
  }, [allTransactions, decisions])

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

  // ── Derived risk values from backend OR mock fallback ──
  const riskScore = riskResult
    ? Math.round(riskResult.risk_score * 100)
    : selected ? Math.round(selected.ensembleScore * 100) : 0

  const riskLevel = riskResult?.decision === 'BLOCK' ? 'HIGH'
    : riskResult?.decision === 'FLAG' ? 'MEDIUM'
    : riskResult?.decision === 'APPROVE' ? 'LOW'
    : selected?.risk_level || (selected?.decision === 'BLOCK' ? 'HIGH' : selected?.decision === 'FLAG' ? 'MEDIUM' : 'LOW')

  const lgbScore  = riskResult ? riskResult.layer_scores.lightgbm     : selected?.lgbScore  || 0
  const isoScore  = riskResult ? riskResult.layer_scores.isolation_forest : selected?.xgbScore || 0
  const behScore  = riskResult ? riskResult.layer_scores.behavioral    : 0
  const reasons   = riskResult?.reasons?.length ? riskResult.reasons : (selected?.riskFactors || [])
  const privacy   = riskResult?.privacy

  // ── Decision for admin panel (use override if any, else backend, else mock) ──
  const adminDecision = selected
    ? (decisions[selected.id]?.decision || riskResult?.decision || selected.decision)
    : null
  const isResolved = selected ? !!decisions[selected.id] : false
  const isFlag     = adminDecision === 'FLAG' && !isResolved

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
              className="w-full bg-bg-50 border border-border rounded-xl py-2 pl-9 pr-3 text-[12px] text-text-primary focus:outline-none focus:ring-1 focus:ring-cyan-500/50 placeholder:text-text-muted"
            />
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

            {/* ── Card 1: Transaction Header ── */}
            <div className={card}>
              <div className={sectionTitle}>Transaction</div>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-[16px] font-bold text-zinc-100">{selected.id}</span>
                    <div className="text-[10px] border border-[#1e3a2a] text-emerald-700/80 px-2 py-0.5 rounded uppercase tracking-widest">Transfer</div>
                  </div>
                  <div className="text-[28px] font-bold text-white mb-2">{formatCurrency(selected.amount)}</div>
                  <div className="flex items-center gap-3 text-[12px]">
                    <span className="text-cyan-400 font-bold">{selected.userId}</span>
                    <span className="text-zinc-700">──────────→</span>
                    <span className="text-cyan-400 font-bold">{selected.receiverId || 'USR-UNKNOWN'}</span>
                  </div>
                </div>
                <div className="text-right text-[11px] text-zinc-500 shrink-0">
                  <div>{new Date(selected.timestamp).toISOString().split('T')[0]}</div>
                  <div>{new Date(selected.timestamp).toTimeString().split(' ')[0]}</div>
                </div>
              </div>
              {selected.amount && (
                <div className="mt-3 text-[11px] text-zinc-600 pt-3 border-t border-[#1a2a3a]">
                  Sender balance: RM 10,000.00 → RM {Math.max(0, 10000 - selected.amount).toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  <span className="text-red-400 ml-2 font-bold">(-{((selected.amount / 10000) * 100).toFixed(1)}%)</span>
                </div>
              )}
            </div>

            {/* ── Card 2: Risk Summary ── */}
            <div className={card}>
              <div className={sectionTitle}>Risk Summary</div>
              {scoring ? (
                <div className="space-y-3">
                  <Skeleton className="h-8 w-[280px]" />
                  <Skeleton className="h-4 w-[200px]" />
                  <Skeleton className="h-4 w-[320px]" />
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-3 gap-4 mb-3">
                    <div>
                      <div className="text-[10px] text-zinc-600 uppercase tracking-wider mb-1">Score</div>
                      <div className="text-[26px] font-bold" style={{ color: riskColor(riskLevel) }}>
                        {riskScore} <span className="text-[14px] text-zinc-500">/ 100</span>
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-zinc-600 uppercase tracking-wider mb-1">Decision</div>
                      <div className="text-[14px] font-bold px-3 py-1 rounded-lg inline-block mt-1"
                        style={{ color: decisionColor(adminDecision), backgroundColor: `${decisionColor(adminDecision)}18` }}>
                        [{adminDecision}]
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-zinc-600 uppercase tracking-wider mb-1">Latency</div>
                      <div className="text-[18px] font-bold text-zinc-300">{selected.latencyMs || '—'}ms</div>
                    </div>
                  </div>
                  <div className="text-[11px] text-zinc-500 flex gap-4 mb-2">
                    <span className="text-cyan-400">LGB: {lgbScore.toFixed(2)}</span>
                    <span className="text-zinc-600">·</span>
                    <span className="text-purple-400">ISO: {isoScore.toFixed(2)}</span>
                    <span className="text-zinc-600">·</span>
                    <span className="text-amber-400">BEH: {behScore.toFixed(2)}</span>
                  </div>
                  <div className="text-[10px] text-zinc-600">
                    <span className="text-cyan-400">({Math.round(lgbScore*100)}×0.55)</span>
                    {' + '}
                    <span className="text-purple-400">({Math.round(isoScore*100)}×0.25)</span>
                    {' + '}
                    <span className="text-amber-400">({Math.round(behScore*100)}×0.20)</span>
                    {' = '}
                    <span style={{ color: riskColor(riskLevel) }} className="font-bold">
                      {(lgbScore*0.55*100 + isoScore*0.25*100 + behScore*0.20*100).toFixed(2)} → {adminDecision}
                    </span>
                  </div>
                </>
              )}
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
                            <span className="text-[11px] font-bold uppercase tracking-wider" style={{ color: mapped.color }}>
                              {mapped.rule}
                            </span>
                            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                              style={{ color: mapped.color, backgroundColor: `${mapped.color}20` }}>
                              {mapped.weight}
                            </span>
                          </div>
                          <div className="text-[11px] text-zinc-500 mt-0.5">{r}</div>
                        </div>
                      )
                    })}
                  </div>
                  <div className="pt-2 border-t border-[#1a2a3a] text-[10px] text-zinc-600 space-y-1">
                    <div className="flex items-center gap-2">
                      <Lock size={9} />PII hashed with SHA-256 — raw identities never reached the model
                    </div>
                    {privacy?.dp_applied && (
                      <div className="flex items-center gap-2"><Lock size={9} />Differential privacy noise applied</div>
                    )}
                  </div>
                </>
              )}
            </div>

            {/* ── Card 4: SHAP Explainability ── */}
            <div className={card}>
              <div className={sectionTitle}>Ensemble SHAP Explanation</div>
              <div className="text-[10px] text-zinc-600 mb-3">
                Weighted feature attribution · LGB×0.55 + ISO×0.25 + BEH×0.20
              </div>
              {scoring ? (
                <div className="space-y-2">
                  {[1,2,3,4,5].map(i => <Skeleton key={i} className="h-6 w-full" />)}
                </div>
              ) : !engineOnline ? (
                <div className="text-[12px] text-zinc-600 py-4 text-center">
                  SHAP unavailable — start backend to enable
                </div>
              ) : !shapResult ? (
                <div className="text-[12px] text-zinc-600 py-4 text-center">
                  {selectedId ? 'Select a transaction to load SHAP explanation' : 'SHAP unavailable — start backend to enable'}
                </div>
              ) : (
                <>
                  {/* 4a — Waterfall */}
                  <ShapWaterfall topFeatures={shapResult.top_features} />

                  {/* 4b — Behavioral Decomposition */}
                  <div className="mt-4 pt-4 border-t border-white/[0.06]">
                    <div className="text-[10px] uppercase tracking-[0.12em] text-zinc-600 mb-3">
                      Behavioral Rule Decomposition
                    </div>
                    <div className="space-y-2">
                      {[
                        { key: 'drain_to_unknown',      label: 'Drain → Unknown',  color: '#ef4444' },
                        { key: 'high_amount_deviation',  label: 'Amt Deviation',    color: '#f59e0b' },
                        { key: 'risky_context',          label: 'Risky Context',    color: '#a855f7' },
                        { key: 'rapid_session',          label: 'Rapid Session',    color: '#00d4ff' },
                      ].map(({ key, label, color }) => {
                        const val = shapResult.beh_contributions?.[key] ?? 0
                        const pct = Math.min(100, val * 100)
                        return (
                          <div key={key} className="flex items-center gap-3 text-[11px]">
                            <span className="w-[110px] shrink-0" style={{ color }}>{label}</span>
                            <div className="flex-1 bg-white/[0.06] h-1.5 rounded-full overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.7 }} />
                            </div>
                            <span className="text-zinc-500 w-16 text-right">{val.toFixed(2)} × score</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  {/* 4c — Model Agreement */}
                  <div className="mt-3 pt-3 border-t border-[#1a2a3a] text-[11px]">
                    {(() => {
                      const delta = Math.abs(lgbScore - isoScore)
                      const disagree = delta > 0.3
                      return (
                        <span className={disagree ? 'text-amber-400' : 'text-zinc-600'}>
                          LGB says: {lgbScore.toFixed(2)} ({lgbScore > 0.7 ? 'HIGH' : lgbScore > 0.35 ? 'MED' : 'LOW'})
                          {' · '}
                          ISO says: {isoScore.toFixed(2)} ({isoScore > 0.7 ? 'HIGH' : isoScore > 0.35 ? 'MED' : 'LOW'})
                          {' · '}
                          Delta: ±{delta.toFixed(2)}
                          {disagree && ' ⚠ Models disagree on this transaction'}
                        </span>
                      )
                    })()}
                  </div>
                </>
              )}
            </div>

            {/* ── Card 5: Admin Decision Panel ── */}
            <div className={card}>
              <div className={sectionTitle}>Admin Decision Panel</div>

              {/* Locked — APPROVE or BLOCK */}
              {!isFlag && !isResolved && (
                <div className="opacity-60">
                  <div className="flex items-center gap-2 text-zinc-300 font-bold text-[13px] mb-3">
                    <Lock size={15} />DECISION LOCKED — HIGH CONFIDENCE
                  </div>
                  <p className="text-[11px] text-zinc-500 mb-4 leading-relaxed">
                    This transaction was automatically{' '}
                    <span style={{ color: decisionColor(adminDecision) }} className="font-bold">[{adminDecision}]</span>{' '}
                    by the model with a risk score of <span className="text-zinc-300">{riskScore}/100</span>.<br />
                    Auto-decisions are made when the score falls outside the FLAG review threshold range.<br />
                    Only FLAGGED transactions require admin review.
                  </p>
                  <div className="text-[10px] text-zinc-700 uppercase tracking-wider mb-2">Score Range</div>
                  <ScoreRangeSVG score={riskScore} />
                </div>
              )}

              {/* Resolved */}
              {isResolved && (
                <div className="space-y-3">
                  <div className="border border-cyan-500/20 rounded-xl p-3 bg-cyan-500/[0.04] flex items-start gap-3">
                    <Check size={16} className="text-cyan-400 shrink-0 mt-0.5" />
                    <div className="font-mono">
                      <div className="text-cyan-400 font-bold text-[11px] uppercase tracking-wider mb-1">Resolved by Admin</div>
                      <div className="text-[10px] text-zinc-500">
                        {new Date(decisions[selected.id].timestamp).toTimeString().split(' ')[0]}
                        {' · '}FLAG → {decisions[selected.id].decision}
                      </div>
                      <div className="text-[11px] text-zinc-300 mt-1 italic">
                        "{decisions[selected.id].reason}"
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* FLAG — actionable */}
              {isFlag && (
                <div className="border border-amber-500/30 bg-amber-500/[0.04] rounded-xl p-4">
                  <div className="flex items-center gap-2 text-amber-400 mb-2">
                    <AlertTriangle size={16} />
                    <span className="font-bold text-[13px] uppercase tracking-wider">Pending Admin Review</span>
                  </div>
                  <p className="text-[11px] text-zinc-500 mb-4">
                    Model confidence is insufficient to auto-decide. Your review is required.
                  </p>

                  {!resolvingState ? (
                    <div className="flex gap-3">
                      <button
                        onClick={() => setResolvingState('APPROVE')}
                        className="flex-1 border border-emerald-500/40 bg-emerald-500/10 text-emerald-400 font-bold text-[12px] uppercase py-2.5 rounded-xl hover:bg-emerald-500/20 transition-colors flex items-center justify-center gap-2"
                      >
                        <Check size={14} /> Approve Transaction
                      </button>
                      <button
                        onClick={() => setResolvingState('BLOCK')}
                        className="flex-1 border border-red-500/40 bg-red-500/10 text-red-400 font-bold text-[12px] uppercase py-2.5 rounded-xl hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"
                      >
                        <X size={14} /> Block Transaction
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div>
                        <label className="block text-[10px] text-zinc-500 uppercase tracking-wider mb-1">
                          Reason for {resolvingState}ing (Required)
                        </label>
                        <input
                          autoFocus
                          value={adminReason}
                          onChange={e => setAdminReason(e.target.value)}
                          placeholder="E.g., Suspicious receiver — new recipient draining pattern"
                          className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg p-2.5 text-[12px] text-zinc-200 focus:outline-none focus:border-cyan-500/50"
                        />
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={() => handleResolve(resolvingState)}
                          disabled={!adminReason.trim()}
                          className="flex-1 bg-cyan-500 text-zinc-900 font-bold text-[12px] uppercase py-2.5 rounded-xl hover:bg-cyan-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                        >Confirm</button>
                        <button
                          onClick={() => { setResolvingState(null); setAdminReason('') }}
                          className="px-5 bg-white/[0.06] text-zinc-300 font-bold text-[12px] uppercase py-2.5 rounded-xl hover:bg-white/[0.10] transition-colors"
                        >Cancel</button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* ── Card 6: Model Feedback Loop ── */}
            {(feedbackAnim === 'pulsing' || (feedbackAnim === 'done' && impactSim)) && (
              <div className={clsx(
                card, 'transition-all duration-300',
                feedbackAnim === 'pulsing'
                  ? 'border-amber-500/50 animate-pulse'
                  : 'border-emerald-500/30'
              )}>
                {feedbackAnim === 'pulsing' && (
                  <>
                    <div className="flex items-center gap-2 text-amber-400 font-bold text-[12px] uppercase tracking-wider mb-3">
                      <Loader2 size={14} className="animate-spin" />
                      Sending Feedback to Model...
                    </div>
                    <div className="h-1.5 w-full bg-white/[0.06] rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500 w-2/3 animate-pulse" />
                    </div>
                  </>
                )}

                {feedbackAnim === 'done' && impactSim && (
                  <>
                    <div className="flex items-center gap-2 text-emerald-400 font-bold text-[12px] uppercase tracking-wider mb-3">
                      <Check size={14} />Model Feedback Received
                    </div>
                    <div className="space-y-3 text-[11px]">
                      <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                        <span className="text-zinc-500">Label added to training buffer:</span>
                        <span className={clsx(
                          'font-bold px-2 py-0.5 rounded text-[10px]',
                          decisions[selected.id]?.decision === 'BLOCK'
                            ? 'bg-red-500/10 text-red-400'
                            : 'bg-emerald-500/10 text-emerald-400'
                        )}>
                          {selected.id} → {decisions[selected.id]?.decision === 'BLOCK' ? 'CONFIRMED FRAUD' : 'CONFIRMED LEGITIMATE'}
                        </span>
                      </div>
                      <div>
                        <div className="text-cyan-400 font-bold text-[10px] uppercase tracking-wider mb-2">Impact Simulation</div>
                        <div className="grid grid-cols-3 gap-3 bg-white/[0.03] border border-white/[0.06] rounded-xl p-3">
                          <div>
                            <div className="text-[9px] text-zinc-600 uppercase mb-1">Similar in Buffer</div>
                            <div className="text-[18px] font-bold text-zinc-200">{impactSim.similarCount}</div>
                          </div>
                          <div>
                            <div className="text-[9px] text-zinc-600 uppercase mb-1">Threshold Adj.</div>
                            <div className={clsx(
                              'text-[18px] font-bold',
                              impactSim.thresholdDelta < 0 ? 'text-red-400' : 'text-emerald-400'
                            )}>
                              {impactSim.thresholdDelta > 0 ? '+' : ''}{impactSim.thresholdDelta} pts
                            </div>
                          </div>
                          <div>
                            <div className="text-[9px] text-zinc-600 uppercase mb-1">Next Retrain</div>
                            <div className="text-[12px] font-bold text-zinc-400">Scheduled</div>
                          </div>
                        </div>
                      </div>
                      <div className="text-zinc-600 text-[10px] italic text-center leading-relaxed">
                        "Your feedback helps the model learn to auto-decide similar cases in future"<br />
                        <span className="text-amber-600">⚠ Simulation only — actual retraining requires full pipeline</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  )
}
