import React, { useState, useEffect, useCallback } from 'react'
import {
  RefreshCw, CheckCircle2, XCircle, AlertTriangle, Loader2,
  ArrowUpRight, ArrowDownRight, Minus, Tag, Brain, Database
} from 'lucide-react'
import clsx from 'clsx'
import Panel from '../components/shared/Panel'
import { formatCurrency } from '../utils/formatters'

const API_BASE = 'http://localhost:8000'

export default function RetrainPanel() {
  // ── State ──────────────────────────────────────────────────
  const [stats, setStats] = useState(null)
  const [transactions, setTransactions] = useState([])
  const [retrainResult, setRetrainResult] = useState(null)
  const [isRetraining, setIsRetraining] = useState(false)
  const [isLoadingTxns, setIsLoadingTxns] = useState(false)
  const [labelingInProgress, setLabelingInProgress] = useState({})
  const [activeFilter, setActiveFilter] = useState('unlabeled') // 'all' | 'unlabeled' | 'labeled'
  const [toastMsg, setToastMsg] = useState(null)

  // ── Fetch stats ────────────────────────────────────────────
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/feedback/stats`)
      if (res.ok) setStats(await res.json())
    } catch { }
  }, [])

  // ── Fetch transactions ─────────────────────────────────────
  const fetchTransactions = useCallback(async () => {
    setIsLoadingTxns(true)
    try {
      const endpoint = activeFilter === 'unlabeled'
        ? '/api/v1/transactions/unlabeled?limit=100'
        : '/api/v1/transactions?limit=100'
      const res = await fetch(`${API_BASE}${endpoint}`)
      if (res.ok) {
        let data = await res.json()
        if (activeFilter === 'labeled') {
          data = data.filter(t => t.analyst_label)
        }
        setTransactions(data)
      }
    } catch { }
    setIsLoadingTxns(false)
  }, [activeFilter])

  useEffect(() => { fetchStats() }, [fetchStats])
  useEffect(() => { fetchTransactions() }, [fetchTransactions])

  // ── Label a transaction ────────────────────────────────────
  const labelTransaction = async (txnId, label) => {
    setLabelingInProgress(prev => ({ ...prev, [txnId]: label }))
    try {
      const res = await fetch(`${API_BASE}/api/v1/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_id: txnId, analyst_label: label }),
      })
      if (res.ok) {
        setToastMsg(`✓ ${txnId.slice(0, 12)} labeled as ${label}`)
        setTimeout(() => setToastMsg(null), 2000)
        // Remove from list if viewing unlabeled
        if (activeFilter === 'unlabeled') {
          setTransactions(prev => prev.filter(t => t.transaction_id !== txnId))
        } else {
          setTransactions(prev => prev.map(t =>
            t.transaction_id === txnId ? { ...t, analyst_label: label } : t
          ))
        }
        fetchStats()
      }
    } catch (e) {
      console.error('Labeling failed:', e)
    }
    setLabelingInProgress(prev => {
      const next = { ...prev }
      delete next[txnId]
      return next
    })
  }

  // ── Trigger retrain ────────────────────────────────────────
  const triggerRetrain = async () => {
    setIsRetraining(true)
    setRetrainResult(null)
    try {
      const res = await fetch(`${API_BASE}/api/v1/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_labeled_samples: 10 }),
      })
      if (res.ok) {
        setRetrainResult(await res.json())
        fetchStats()
      } else {
        const err = await res.json()
        setRetrainResult({ status: 'error', message: err.detail || 'Retrain failed' })
      }
    } catch (e) {
      setRetrainResult({ status: 'error', message: e.message })
    }
    setIsRetraining(false)
  }

  // ── Helpers ────────────────────────────────────────────────
  const riskColor = (score) => {
    if (score >= 0.7) return 'text-red-400'
    if (score >= 0.4) return 'text-amber-400'
    return 'text-emerald-400'
  }

  const riskBg = (score) => {
    if (score >= 0.7) return 'bg-red-500/10 border-red-500/20'
    if (score >= 0.4) return 'bg-amber-500/10 border-amber-500/20'
    return 'bg-emerald-500/10 border-emerald-500/20'
  }

  return (
    <div className="max-w-[1400px] mx-auto space-y-4 font-mono pb-8">
      {/* Toast notification */}
      {toastMsg && (
        <div className="fixed top-4 right-4 z-50 bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 px-4 py-2 rounded-xl text-xs font-mono animate-in fade-in slide-in-from-top-2 duration-200">
          {toastMsg}
        </div>
      )}

      {/* PAGE HEADER */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-wider text-foreground">CLOSED-LOOP RETRAINING</h1>
          <p className="text-xs text-muted-foreground tracking-widest uppercase mt-1">
            Label Transactions · Re-Optimize Ensemble · Deploy
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={fetchTransactions}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg font-mono text-[10px] font-bold tracking-widest uppercase transition-all border bg-transparent text-text-secondary border-border hover:bg-bg-200"
          >
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-3">
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#06b6d4' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Total in DB</p>
          <p className="text-2xl font-bold mt-1 text-cyan-400">
            {stats?.total_transactions?.toLocaleString() ?? '—'}
          </p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#10b981' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Labeled</p>
          <p className="text-2xl font-bold mt-1 text-emerald-400">
            {stats?.labeled_count?.toLocaleString() ?? '—'}
          </p>
          <p className="text-[9px] text-muted-foreground mt-1">
            {stats ? `${((stats.labeled_count / Math.max(stats.total_transactions, 1)) * 100).toFixed(1)}% coverage` : ''}
          </p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#ef4444' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Fraud Labels</p>
          <p className="text-2xl font-bold mt-1 text-red-400">{stats?.fraud_labels ?? '—'}</p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: '#3b82f6' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Legit Labels</p>
          <p className="text-2xl font-bold mt-1 text-blue-400">{stats?.legit_labels ?? '—'}</p>
        </Panel>
        <Panel className="!p-4 border-l-2" style={{ borderLeftColor: stats?.ready_to_retrain ? '#10b981' : '#f59e0b' }}>
          <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Retrain Ready</p>
          <p className={clsx("text-2xl font-bold mt-1", stats?.ready_to_retrain ? 'text-emerald-400' : 'text-amber-400')}>
            {stats?.ready_to_retrain ? 'YES' : 'NO'}
          </p>
          <p className="text-[9px] text-muted-foreground mt-1">
            Need ≥{stats?.min_samples_needed ?? 50} labels
          </p>
        </Panel>
      </div>

      <div className="grid grid-cols-12 gap-4">
        {/* LEFT: Labeling Interface */}
        <div className="col-span-8 space-y-4">
          <Panel title="TRANSACTION LABELING" className="flex flex-col" style={{ minHeight: '500px' }}>
            {/* Filter tabs */}
            <div className="flex gap-1 mb-4">
              {[
                { id: 'unlabeled', label: 'Unlabeled', count: stats?.unlabeled_count },
                { id: 'labeled', label: 'Labeled', count: stats?.labeled_count },
                { id: 'all', label: 'All', count: stats?.total_transactions },
              ].map(f => (
                <button
                  key={f.id}
                  onClick={() => setActiveFilter(f.id)}
                  className={clsx(
                    "px-3 py-1.5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all border",
                    activeFilter === f.id
                      ? "bg-[rgba(0,229,255,0.1)] text-[#00e5ff] border-[rgba(0,229,255,0.3)]"
                      : "bg-transparent text-muted-foreground border-border hover:bg-bg-200"
                  )}
                >
                  {f.label} {f.count != null ? `(${f.count.toLocaleString()})` : ''}
                </button>
              ))}
            </div>

            {/* Transaction table */}
            <div className="flex-1 overflow-y-auto custom-scrollbar -mr-2 pr-2" style={{ maxHeight: '480px' }}>
              {isLoadingTxns ? (
                <div className="flex items-center justify-center py-16 text-muted-foreground">
                  <Loader2 size={20} className="animate-spin mr-2" /> Loading...
                </div>
              ) : transactions.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
                  <Database size={32} className="mb-2 opacity-40" />
                  <p className="text-xs">No transactions found</p>
                  <p className="text-[10px] mt-1">Run the simulator to generate and save transactions</p>
                </div>
              ) : (
                <table className="w-full text-left border-collapse">
                  <thead className="sticky top-0 bg-bg-100 z-10 border-b border-border">
                    <tr>
                      <th className="py-2 pl-3 text-[9px] uppercase tracking-widest text-muted-foreground w-28">TX ID</th>
                      <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-right w-20">Amount</th>
                      <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-center w-16">Risk</th>
                      <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-center w-16">ML Label</th>
                      <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-center w-24">Status</th>
                      <th className="py-2 text-[9px] uppercase tracking-widest text-muted-foreground text-center pr-3">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {transactions.map(t => {
                      const isLabeling = !!labelingInProgress[t.transaction_id]
                      const riskScore = t.ml_risk_score || 0

                      return (
                        <tr key={t.transaction_id} className="hover:bg-bg-200/50 transition-colors">
                          <td className="py-2.5 pl-3 text-[10px] text-text-primary">
                            <span className="font-mono bg-white/5 px-1 rounded">{t.transaction_id?.slice(0, 12)}...</span>
                          </td>
                          <td className="py-2.5 text-[10px] text-right text-text-primary">
                            {formatCurrency(t.amount || 0).replace('MYR', 'RM')}
                          </td>
                          <td className="py-2.5 text-center">
                            <span className={clsx("text-[10px] font-bold px-1.5 py-0.5 rounded border", riskBg(riskScore), riskColor(riskScore))}>
                              {(riskScore * 100).toFixed(0)}%
                            </span>
                          </td>
                          <td className="py-2.5 text-center">
                            <span className={clsx(
                              "text-[9px] uppercase font-bold px-1.5 py-0.5 rounded",
                              t.is_fraud ? "text-red-400 bg-red-500/10" : "text-emerald-400 bg-emerald-500/10"
                            )}>
                              {t.is_fraud ? 'FRAUD' : 'LEGIT'}
                            </span>
                          </td>
                          <td className="py-2.5 text-center">
                            {t.analyst_label ? (
                              <span className={clsx(
                                "text-[9px] uppercase font-bold px-2 py-0.5 rounded-full inline-flex items-center gap-1",
                                t.analyst_label === 'FRAUD'
                                  ? "text-red-400 bg-red-500/10 border border-red-500/20"
                                  : "text-emerald-400 bg-emerald-500/10 border border-emerald-500/20"
                              )}>
                                <Tag size={8} />
                                {t.analyst_label}
                              </span>
                            ) : (
                              <span className="text-[9px] text-muted-foreground">Unlabeled</span>
                            )}
                          </td>
                          <td className="py-2.5 text-center pr-3">
                            {!t.analyst_label && (
                              <div className="flex gap-1 justify-center">
                                <button
                                  onClick={() => labelTransaction(t.transaction_id, 'FRAUD')}
                                  disabled={isLabeling}
                                  className="flex items-center gap-1 px-2 py-1 rounded-lg text-[9px] font-bold uppercase bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-all disabled:opacity-50"
                                >
                                  {labelingInProgress[t.transaction_id] === 'FRAUD' ? <Loader2 size={10} className="animate-spin" /> : <XCircle size={10} />}
                                  Fraud
                                </button>
                                <button
                                  onClick={() => labelTransaction(t.transaction_id, 'LEGIT')}
                                  disabled={isLabeling}
                                  className="flex items-center gap-1 px-2 py-1 rounded-lg text-[9px] font-bold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all disabled:opacity-50"
                                >
                                  {labelingInProgress[t.transaction_id] === 'LEGIT' ? <Loader2 size={10} className="animate-spin" /> : <CheckCircle2 size={10} />}
                                  Legit
                                </button>
                              </div>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </Panel>
        </div>

        {/* RIGHT: Retrain Controls */}
        <div className="col-span-4 space-y-4">
          {/* Retrain Button */}
          <Panel title="MODEL RETRAINING">
            <div className="space-y-4 pt-2">
              <p className="text-[9px] text-muted-foreground leading-relaxed">
                Re-optimizes ensemble weights and decision thresholds using your labeled data.
                Base models (LightGBM, IsoForest) are preserved — only the fusion layer is updated.
              </p>

              {/* Progress bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-[9px] text-muted-foreground">
                  <span>Labeling Progress</span>
                  <span>{stats?.labeled_count ?? 0}/{stats?.min_samples_needed ?? 50} min</span>
                </div>
                <div className="h-2 bg-bg-200 rounded-full overflow-hidden">
                  <div
                    className={clsx("h-full rounded-full transition-all duration-500",
                      stats?.ready_to_retrain ? "bg-emerald-400" : "bg-amber-400"
                    )}
                    style={{ width: `${Math.min(((stats?.labeled_count || 0) / (stats?.min_samples_needed || 50)) * 100, 100)}%` }}
                  />
                </div>
              </div>

              <button
                onClick={triggerRetrain}
                disabled={isRetraining || !stats?.ready_to_retrain}
                className={clsx(
                  "w-full flex items-center justify-center gap-2 py-3 rounded-xl font-mono text-xs font-bold uppercase tracking-widest transition-all border",
                  stats?.ready_to_retrain && !isRetraining
                    ? "bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-400 border-cyan-500/30 hover:from-cyan-500/30 hover:to-blue-500/30 shadow-[0_0_20px_rgba(6,182,212,0.15)]"
                    : "bg-bg-200/50 text-muted-foreground border-border cursor-not-allowed opacity-60"
                )}
              >
                {isRetraining ? (
                  <><Loader2 size={14} className="animate-spin" /> Retraining...</>
                ) : (
                  <><Brain size={14} /> Retrain Ensemble</>
                )}
              </button>

              {!stats?.ready_to_retrain && stats && (
                <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-3">
                  <div className="flex items-start gap-2">
                    <AlertTriangle size={12} className="text-amber-400 mt-0.5 shrink-0" />
                    <p className="text-[9px] text-amber-400 leading-relaxed">
                      Need at least {stats.min_samples_needed} labeled samples (with ≥5 FRAUD and ≥5 LEGIT) before retraining.
                      Currently: {stats.labeled_count} labeled ({stats.fraud_labels} FRAUD, {stats.legit_labels} LEGIT).
                    </p>
                  </div>
                </div>
              )}
            </div>
          </Panel>

          {/* Retrain Results */}
          {retrainResult && (
            <Panel title="RETRAIN RESULTS">
              <div className="space-y-3 pt-2">
                {/* Status Badge */}
                <div className={clsx(
                  "flex items-center gap-2 px-3 py-2 rounded-xl border",
                  retrainResult.status === 'success'
                    ? "bg-emerald-500/10 border-emerald-500/20"
                    : retrainResult.status === 'insufficient_data'
                    ? "bg-amber-500/10 border-amber-500/20"
                    : "bg-red-500/10 border-red-500/20"
                )}>
                  {retrainResult.status === 'success' ? (
                    <CheckCircle2 size={14} className="text-emerald-400" />
                  ) : (
                    <AlertTriangle size={14} className="text-amber-400" />
                  )}
                  <span className={clsx("text-[10px] font-bold uppercase tracking-wider",
                    retrainResult.status === 'success' ? "text-emerald-400" : "text-amber-400"
                  )}>
                    {retrainResult.status}
                  </span>
                </div>

                {retrainResult.status === 'success' && (
                  <>
                    {/* F1 Comparison */}
                    <div className="bg-bg-200 border border-border rounded-xl p-3">
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wider mb-2">F1 Score Change</p>
                      <div className="flex items-end justify-between">
                        <div>
                          <span className="text-muted-foreground text-[10px]">Old: </span>
                          <span className="text-lg font-bold text-text-primary">{(retrainResult.old_f1 * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/5">
                          {retrainResult.improvement_pct > 0 ? (
                            <ArrowUpRight size={12} className="text-emerald-400" />
                          ) : retrainResult.improvement_pct < 0 ? (
                            <ArrowDownRight size={12} className="text-red-400" />
                          ) : (
                            <Minus size={12} className="text-muted-foreground" />
                          )}
                          <span className={clsx("text-[10px] font-bold",
                            retrainResult.improvement_pct > 0 ? "text-emerald-400" :
                            retrainResult.improvement_pct < 0 ? "text-red-400" : "text-muted-foreground"
                          )}>
                            {retrainResult.improvement_pct > 0 ? '+' : ''}{retrainResult.improvement_pct.toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-muted-foreground text-[10px]">New: </span>
                          <span className="text-lg font-bold text-cyan-400">{(retrainResult.new_f1 * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </div>

                    {/* Weights Comparison */}
                    <div className="bg-bg-200 border border-border rounded-xl p-3">
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wider mb-2">Weight Changes</p>
                      <div className="space-y-2">
                        {['lgb', 'iso', 'beh'].map(key => {
                          const oldW = retrainResult.old_weights[key]
                          const newW = retrainResult.new_weights[key]
                          const diff = newW - oldW
                          const label = { lgb: 'LightGBM', iso: 'IsoForest', beh: 'Behavioral' }[key]
                          const color = { lgb: '#00e5ff', iso: '#a855f7', beh: '#f59e0b' }[key]

                          return (
                            <div key={key} className="flex items-center justify-between">
                              <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color }}>{label}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] text-muted-foreground">{oldW.toFixed(2)}</span>
                                <span className="text-[10px] text-muted-foreground">→</span>
                                <span className="text-[10px] font-bold" style={{ color }}>{newW.toFixed(2)}</span>
                                {diff !== 0 && (
                                  <span className={clsx("text-[9px]",
                                    diff > 0 ? "text-emerald-400" : "text-red-400"
                                  )}>
                                    ({diff > 0 ? '+' : ''}{(diff * 100).toFixed(0)}%)
                                  </span>
                                )}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>

                    {/* Samples info */}
                    <p className="text-[9px] text-muted-foreground text-center">
                      Trained on {retrainResult.samples_used} analyst-labeled samples
                    </p>
                  </>
                )}

                <p className="text-[9px] text-muted-foreground leading-relaxed bg-white/[0.02] border border-white/[0.05] p-2 rounded-lg">
                  {retrainResult.message}
                </p>
              </div>
            </Panel>
          )}

          {/* Architecture Diagram */}
          <Panel title="HOW IT WORKS">
            <div className="space-y-3 pt-2">
              <div className="space-y-2">
                {[
                  { step: '1', label: 'Label', desc: 'Analysts mark flagged transactions as FRAUD or LEGIT', color: '#f59e0b' },
                  { step: '2', label: 'Score', desc: 'All labeled txns are re-scored through the full engine', color: '#00e5ff' },
                  { step: '3', label: 'Optimize', desc: 'Grid search finds optimal weights & thresholds on labeled data', color: '#a855f7' },
                  { step: '4', label: 'Deploy', desc: 'New config is saved and hot-reloaded (no restart needed)', color: '#10b981' },
                ].map(s => (
                  <div key={s.step} className="flex gap-3 items-start">
                    <div
                      className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 mt-0.5"
                      style={{ backgroundColor: `${s.color}20`, color: s.color, border: `1px solid ${s.color}40` }}
                    >
                      {s.step}
                    </div>
                    <div>
                      <p className="text-[10px] font-bold uppercase tracking-wider" style={{ color: s.color }}>{s.label}</p>
                      <p className="text-[9px] text-muted-foreground leading-relaxed">{s.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  )
}
