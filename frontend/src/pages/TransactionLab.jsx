import React, { useState, useCallback } from 'react'
import {
  Zap, FlaskConical, AlertTriangle, ShieldCheck, ShieldAlert,
  Loader2, RotateCcw, ChevronDown
} from 'lucide-react'
import clsx from 'clsx'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine,
  ResponsiveContainer, Cell
} from 'recharts'

const API_BASE = 'http://localhost:8000'

// ── Default form values ──────────────────────────────────
const DEFAULTS = {
  transaction_type: 'CASH_OUT',
  amount: 500,
  avg_transaction_amount_30d: 100,
  transaction_hour: 3,
  is_weekend: 0,
  is_new_device: 1,
  failed_login_attempts: 2,
  ip_risk_score: 0.7,
  sender_account_fully_drained: 1,
  account_age_days: 15,
  tx_count_24h: 8,
  is_new_recipient: 1,
  country_mismatch: 1,
  is_proxy_ip: 1,
  session_duration_seconds: 30,
}

// ── Field definitions with labels/types ─────────────────
const FIELDS = [
  { key: 'transaction_type', label: 'Transfer Type', type: 'select', options: ['CASH_OUT', 'TRANSFER'], col: 1 },
  { key: 'amount', label: 'Amount ($)', type: 'number', step: 1, col: 2 },
  { key: 'avg_transaction_amount_30d', label: '30-Day Avg ($)', type: 'number', step: 1, col: 3 },
  { key: 'transaction_hour', label: 'Hour (0-23)', type: 'number', min: 0, max: 23, col: 1 },
  { key: 'is_weekend', label: 'Weekend', type: 'toggle', col: 2 },
  { key: 'is_new_device', label: 'New Device', type: 'toggle', col: 3 },
  { key: 'failed_login_attempts', label: 'Failed Logins', type: 'number', min: 0, col: 1 },
  { key: 'ip_risk_score', label: 'IP Risk', type: 'number', step: 0.01, min: 0, max: 1, col: 2 },
  { key: 'sender_account_fully_drained', label: 'Account Drained', type: 'toggle', col: 3 },
  { key: 'account_age_days', label: 'Account Age (Days)', type: 'number', min: 0, col: 1 },
  { key: 'tx_count_24h', label: 'Txns Last 24H', type: 'number', min: 0, col: 2 },
  { key: 'is_new_recipient', label: 'New Recipient', type: 'toggle', col: 3 },
  { key: 'country_mismatch', label: 'Country Mismatch', type: 'toggle', col: 1 },
  { key: 'is_proxy_ip', label: 'Proxy IP', type: 'toggle', col: 2 },
  { key: 'session_duration_seconds', label: 'Session Duration (s)', type: 'number', min: 0, col: 3 },
]

// ── Result cards ─────────────────────────────────────────
function ScoreGauge({ score, size = 120 }) {
  const r = (size - 12) / 2
  const circumference = Math.PI * r
  const pct = Math.min(100, Math.max(0, score))
  const offset = circumference - (pct / 100) * circumference
  const color = pct >= 70 ? '#ef4444' : pct >= 35 ? '#f59e0b' : '#10b981'

  return (
    <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
      <path
        d={`M 6,${size / 2} A ${r},${r} 0 0 1 ${size - 6},${size / 2}`}
        fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" strokeLinecap="round"
      />
      <path
        d={`M 6,${size / 2} A ${r},${r} 0 0 1 ${size - 6},${size / 2}`}
        fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={offset}
        className="transition-all duration-1000 ease-out"
      />
      <text x={size / 2} y={size / 2 - 4} textAnchor="middle" fill={color}
        fontSize="28" fontWeight="bold" fontFamily="JetBrains Mono, monospace">
        {Math.round(pct)}
      </text>
      <text x={size / 2} y={size / 2 + 12} textAnchor="middle" fill="rgba(255,255,255,0.4)"
        fontSize="9" fontFamily="JetBrains Mono, monospace" letterSpacing="0.12em">
        {pct >= 70 ? 'HIGH RISK' : pct >= 35 ? 'MEDIUM RISK' : 'LOW RISK'}
      </text>
    </svg>
  )
}

function LayerBar({ label, weight, value, color }) {
  const pct = Math.min(100, Math.max(0, value * 100))
  return (
    <div className="flex items-center gap-3">
      <div className="w-[130px] shrink-0">
        <div className="font-mono text-[10px] tracking-widest text-text-secondary">{label}</div>
        <div className="font-mono text-[9px] text-text-muted/50">({weight})</div>
      </div>
      <div className="flex-1 h-[6px] bg-white/5 rounded-full relative overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{ backgroundColor: color, width: `${pct}%` }}
        />
      </div>
      <div className="w-[36px] text-right font-mono text-[12px] font-bold" style={{ color }}>
        {Math.round(pct)}
      </div>
    </div>
  )
}

// ── SHAP Waterfall ───────────────────────────────────────
function ShapWaterfall({ topFeatures }) {
  if (!topFeatures || topFeatures.length === 0) return null
  const data = topFeatures.map(f => ({
    name: f.feature.replace(/_/g, ' ').substring(0, 22),
    fullName: f.feature,
    value: f.contribution ?? f.shap_value ?? 0,
    actual: f.actual_value ?? 'N/A',
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 60, left: 110, bottom: 4 }}>
        <XAxis
          type="number"
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fill: 'rgba(255,255,255,0.35)' }}
          tickFormatter={v => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
        />
        <YAxis
          type="category" dataKey="name"
          tick={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fill: 'rgba(255,255,255,0.5)' }}
          width={108}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const d = payload[0].payload
            return (
              <div className="bg-zinc-900 border border-white/10 rounded p-2 font-mono text-[11px]">
                <div className="text-zinc-300">{d.fullName}</div>
                <div style={{ color: d.value >= 0 ? '#ef4444' : '#10b981' }}>
                  SHAP: {d.value >= 0 ? '+' : ''}{d.value.toFixed(4)}
                </div>
                <div className="text-zinc-500">value: {typeof d.actual === 'boolean' ? String(d.actual) : d.actual}</div>
              </div>
            )
          }}
        />
        <ReferenceLine x={0} stroke="rgba(255,255,255,0.1)" strokeDasharray="3 3" />
        <Bar dataKey="value" radius={[0, 2, 2, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.value >= 0 ? '#ef4444' : '#10b981'} fillOpacity={0.8} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Main Component ───────────────────────────────────────
export default function TransactionLab() {
  const [form, setForm] = useState({ ...DEFAULTS })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [shapResult, setShapResult] = useState(null)
  const [history, setHistory] = useState([])

  const updateField = useCallback((key, val) => {
    setForm(prev => ({ ...prev, [key]: val }))
  }, [])

  const resetForm = useCallback(() => {
    setForm({ ...DEFAULTS })
    setResult(null)
    setShapResult(null)
  }, [])

  const scoreTransaction = useCallback(async () => {
    setLoading(true)
    setResult(null)
    setShapResult(null)

    const txnId = `MANUAL-${Date.now().toString(36).toUpperCase()}`
    const body = {
      transaction_id: txnId,
      amount: parseFloat(form.amount) || 0,
      sender_id: 'MANUAL-SENDER',
      receiver_id: 'MANUAL-RECEIVER',
      transaction_type: form.transaction_type.toLowerCase(),
      timestamp: new Date().toISOString(),
      avg_transaction_amount_30d: parseFloat(form.avg_transaction_amount_30d) || 100,
      amount_vs_avg_ratio: parseFloat(form.amount) / Math.max(parseFloat(form.avg_transaction_amount_30d) || 100, 1),
      transaction_hour: parseInt(form.transaction_hour) || 0,
      is_weekend: parseInt(form.is_weekend) || 0,
      is_new_device: parseInt(form.is_new_device) || 0,
      failed_login_attempts: parseInt(form.failed_login_attempts) || 0,
      is_proxy_ip: parseInt(form.is_proxy_ip) || 0,
      ip_risk_score: parseFloat(form.ip_risk_score) || 0,
      country_mismatch: parseInt(form.country_mismatch) || 0,
      sender_account_fully_drained: parseInt(form.sender_account_fully_drained) || 0,
      account_age_days: parseInt(form.account_age_days) || 0,
      tx_count_24h: parseInt(form.tx_count_24h) || 0,
      is_new_recipient: parseInt(form.is_new_recipient) || 0,
      established_user_new_recipient: 0,
      session_duration_seconds: parseInt(form.session_duration_seconds) || 0,
      recipient_risk_profile_score: 0.0,
    }

    try {
      const [riskRes, shapRes] = await Promise.all([
        fetch(`${API_BASE}/api/v1/score-transaction`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }).then(r => r.ok ? r.json() : null).catch(() => null),

        fetch(`${API_BASE}/api/v1/explain/${txnId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }).then(r => r.ok ? r.json() : null).catch(() => null),
      ])

      setResult(riskRes)
      setShapResult(shapRes)

      if (riskRes) {
        setHistory(prev => [{
          time: new Date().toLocaleTimeString(),
          type: form.transaction_type,
          amount: parseFloat(form.amount),
          score: riskRes.risk_score,
          decision: riskRes.decision,
          reason: riskRes.reasons?.[0] || '—',
        }, ...prev].slice(0, 20))
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [form])

  const card = 'bg-bg-100 border border-border rounded-2xl p-5 relative overflow-hidden'

  return (
    <div className="max-w-[1500px] mx-auto space-y-5 pb-8 font-mono">

      {/* ══ FORM + RESULT SIDE BY SIDE ══ */}
      <div className="flex gap-5">

        {/* ── FORM PANEL ── */}
        <div className={clsx(card, 'flex-1')}>
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <FlaskConical size={16} className="text-cyan-400" />
              <span className="text-[14px] font-bold text-text-primary tracking-wide">Score a Transaction</span>
            </div>
            <button
              onClick={resetForm}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-border text-[10px] text-text-muted hover:text-text-primary hover:border-border-md transition-all uppercase tracking-widest"
            >
              <RotateCcw size={11} /> Reset
            </button>
          </div>

          <div className="grid grid-cols-3 gap-x-5 gap-y-4">
            {FIELDS.map(field => (
              <div key={field.key}>
                <label className="block text-[9px] uppercase tracking-[0.15em] text-text-muted/60 font-bold mb-1.5">
                  {field.label}
                </label>
                {field.type === 'select' ? (
                  <div className="relative">
                    <select
                      value={form[field.key]}
                      onChange={e => updateField(field.key, e.target.value)}
                      className="w-full bg-bg-50 border border-border rounded-lg px-3 py-2.5 text-[13px] text-text-primary focus:outline-none focus:ring-1 focus:ring-cyan-500/30 appearance-none cursor-pointer"
                    >
                      {field.options.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                    <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
                  </div>
                ) : field.type === 'toggle' ? (
                  <div className="relative">
                    <select
                      value={form[field.key]}
                      onChange={e => updateField(field.key, parseInt(e.target.value))}
                      className="w-full bg-bg-50 border border-border rounded-lg px-3 py-2.5 text-[13px] text-text-primary focus:outline-none focus:ring-1 focus:ring-cyan-500/30 appearance-none cursor-pointer"
                    >
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                    <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted pointer-events-none" />
                  </div>
                ) : (
                  <input
                    type="number"
                    value={form[field.key]}
                    onChange={e => updateField(field.key, e.target.value)}
                    step={field.step || 1}
                    min={field.min}
                    max={field.max}
                    className="w-full bg-bg-50 border border-border rounded-lg px-3 py-2.5 text-[13px] text-text-primary focus:outline-none focus:ring-1 focus:ring-cyan-500/30 tabular-nums"
                  />
                )}
              </div>
            ))}
          </div>

          {/* Score Button */}
          <div className="mt-5 flex justify-end">
            <button
              onClick={scoreTransaction}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold text-[12px] uppercase tracking-widest shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/40 transition-all disabled:opacity-50"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
              Score Now
            </button>
          </div>
        </div>

        {/* ── RESULT PANEL ── */}
        <div className={clsx(card, 'w-[320px] shrink-0')}>
          <div className="text-[10px] uppercase tracking-[0.15em] text-text-muted/60 font-bold mb-4">Result</div>

          {loading ? (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <Loader2 size={32} className="animate-spin text-cyan-500/50" />
              <span className="text-[11px] text-text-muted">Scoring...</span>
            </div>
          ) : !result ? (
            <div className="flex flex-col items-center justify-center py-12 text-text-muted/40 text-[12px]">
              Submit a transaction to see results
            </div>
          ) : (
            <div className="space-y-5">
              {/* Gauge */}
              <div className="flex flex-col items-center">
                <ScoreGauge score={result.risk_score} size={160} />
              </div>

              {/* Layer Breakdown */}
              <div className="space-y-3">
                <LayerBar label="LIGHTGBM" weight="60%" value={result.supervised_score} color="#4FC3F7" />
                <LayerBar label="ISOFOREST" weight="10%" value={result.unsupervised_score} color="#ce93d8" />
                <LayerBar label="BEHAVIORAL" weight="30%" value={result.behavioral_score} color="#ffb74d" />
              </div>

              {/* Decision Badge */}
              <div className="flex items-center justify-center gap-2 py-2">
                {result.decision === 'BLOCK' ? (
                  <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30">
                    <ShieldAlert size={16} className="text-red-400" />
                    <span className="text-red-400 font-bold text-[12px] uppercase tracking-widest">BLOCKED</span>
                  </div>
                ) : result.decision === 'FLAG' ? (
                  <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-amber-500/10 border border-amber-500/30">
                    <AlertTriangle size={16} className="text-amber-400" />
                    <span className="text-amber-400 font-bold text-[12px] uppercase tracking-widest">FLAGGED</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500/10 border border-emerald-500/30">
                    <ShieldCheck size={16} className="text-emerald-400" />
                    <span className="text-emerald-400 font-bold text-[12px] uppercase tracking-widest">APPROVED</span>
                  </div>
                )}
              </div>

              {/* Reasons */}
              {result.reasons?.length > 0 && (
                <div>
                  <div className="text-[9px] uppercase tracking-[0.15em] text-text-muted/40 font-bold mb-2">Top Reasons</div>
                  <div className="space-y-1">
                    {result.reasons.slice(0, 3).map((r, i) => (
                      <div key={i} className="text-[10px] text-amber-400/80 flex items-start gap-1.5">
                        <span className="text-amber-400 mt-0.5">•</span>
                        <span className="leading-snug">{r}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Latency */}
              {result.latency_ms != null && (
                <div className="text-[10px] text-text-muted/30 text-center font-mono">
                  Scored in {result.latency_ms.toFixed(0)}ms
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ══ SHAP EXPLANATION ══ */}
      {shapResult?.top_features?.length > 0 && (
        <div className={card}>
          <div className="text-[10px] uppercase tracking-[0.15em] text-text-muted/60 font-bold mb-3">
            SHAP Explanation — Why This Score?
          </div>
          <ShapWaterfall topFeatures={shapResult.top_features} />
        </div>
      )}

      {/* ══ INVESTIGATION LOG ══ */}
      <div className={card}>
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[14px]">📋</span>
          <span className="text-[12px] font-bold text-text-primary tracking-wide">Investigation Log</span>
          <span className="text-[10px] text-text-muted/40 ml-auto">{history.length} entries</span>
        </div>

        {history.length === 0 ? (
          <div className="text-center text-text-muted/30 text-[11px] py-4">
            No scored transactions yet
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-border/30">
            <div className="grid grid-cols-[80px_80px_100px_100px_100px_1fr] gap-1 px-3 py-2 bg-bg-300/30 text-[8px] text-text-muted/50 uppercase tracking-widest font-bold">
              <span>Time</span>
              <span>Type</span>
              <span>Amount</span>
              <span>Score</span>
              <span>Decision</span>
              <span>Top Reason</span>
            </div>
            {history.map((h, i) => (
              <div
                key={i}
                className={clsx(
                  'grid grid-cols-[80px_80px_100px_100px_100px_1fr] gap-1 px-3 py-2 text-[11px] border-t border-border/10',
                  i % 2 === 1 && 'bg-bg-200/20'
                )}
              >
                <span className="text-text-muted/50">{h.time}</span>
                <span className="text-cyan-400">{h.type}</span>
                <span className="text-text-secondary">${h.amount.toLocaleString()}</span>
                <span
                  className="font-bold"
                  style={{ color: h.score >= 70 ? '#ef4444' : h.score >= 35 ? '#f59e0b' : '#10b981' }}
                >
                  {h.score.toFixed(1)}%
                </span>
                <span
                  className="font-bold text-[10px] uppercase"
                  style={{ color: h.decision === 'BLOCK' ? '#ef4444' : h.decision === 'FLAG' ? '#f59e0b' : '#10b981' }}
                >
                  {h.decision}
                </span>
                <span className="text-text-muted/40 truncate">{h.reason}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
