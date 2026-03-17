import { useState, useEffect, useCallback, useMemo } from 'react'
import { generateTransaction } from '../utils/fraudScoring'

/**
 * useTransactionEngine
 *
 * Generates synthetic transactions and sends them to the real backend
 * at POST /predict for scoring.  NO frontend scoring fallback.
 * If the backend is offline, transactions are silently discarded.
 */

export function useTransactionEngine() {
  const [params, setParams] = useState({
    simulationSpeed: 1,   // transactions per second
    smoteLevel: 0.30,     // augments probability of high-risk features
  })

  const [allTransactions, setAllTransactions] = useState([])
  const [isRunning, setIsRunning] = useState(true)
  const [attackQueue, setAttackQueue] = useState(0)

  // ─── Fix 4: Engine health polling ──────────────────────────────────────────
  const [engineOnline, setEngineOnline] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/v1/health')
        if (res.ok) {
          const data = await res.json()
          setEngineOnline(data.engine_loaded === true)
        } else {
          setEngineOnline(false)
        }
      } catch {
        setEngineOnline(false)
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  // ─── Fix 5: Backend stats polling ─────────────────────────────────────────
  const [backendStats, setBackendStats] = useState(null)

  useEffect(() => {
    const fetchStats = async () => {
      // Only poll stats every 2s while simulation is running (per requirement)
      if (!isRunning && attackQueue === 0) return

      try {
        const res = await fetch('http://localhost:8000/api/v1/stats')
        if (res.ok) setBackendStats(await res.json())
        else setBackendStats(null)
      } catch {
        setBackendStats(null)
      }
    }
    fetchStats()
    const interval = setInterval(fetchStats, 2000) // Poll every 2s
    return () => clearInterval(interval)
  }, [isRunning, attackQueue])

  // ─── Fix 7: Fetch config (thresholds + weights) from backend ──────────────
  const [config, setConfig] = useState({
    approve_threshold: 0.35,
    flag_threshold: 0.70,
    weights: { lgb: 0.55, iso: 0.25, beh: 0.20 }
  })

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/v1/config')
        if (res.ok) setConfig(await res.json())
      } catch { }
    }
    fetchConfig()
  }, [])

  // ─── Fix 3: Backend-only transaction scoring loop ─────────────────────────
  useEffect(() => {
    if (!isRunning && attackQueue === 0) return

    let timeoutId
    const tick = async () => {
      const isAttackMode = attackQueue > 0

      // Select template
      let template = 'normal'
      if (isAttackMode) {
        template = Math.random() > 0.4 ? 'attack' : 'suspicious'
      }

      const raw = generateTransaction(template, params.smoteLevel)

      // Schedule next tick IMMEDIATELY for non-blocking simulation
      const interval = isAttackMode ? 500 : (1000 / Math.max(params.simulationSpeed, 0.1))
      timeoutId = setTimeout(tick, interval)
      if (isAttackMode) setAttackQueue(q => q - 1)

      const startTick = Date.now()
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 3000)

      try {
        const res = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          body: JSON.stringify({
            transaction_id: raw.transaction_id,
            amount: raw.amount,
            sender_id: raw.name_sender,
            receiver_id: raw.name_recipient,
            transaction_type: raw.transfer_type.toLowerCase(),
            timestamp: raw.timestamp,
            // Balanced fields (matched to TransactionRequest schema)
            sender_balance_before: raw.sender_balance_before,
            sender_balance_after: raw.sender_balance_after,
            receiver_balance_before: raw.receiver_balance_before,
            receiver_balance_after: raw.receiver_balance_after,
            // Contextual features
            amount_vs_avg_ratio: raw.amount_vs_avg_ratio,
            avg_transaction_amount_30d: raw.avg_transaction_amount_30d,
            session_duration_seconds: raw.session_duration_seconds,
            failed_login_attempts: raw.failed_login_attempts,
            tx_count_24h: raw.tx_count_24h,
            transaction_hour: raw.transaction_hour,
            is_weekend: raw.is_weekend,
            sender_account_fully_drained: raw.sender_account_fully_drained,
            is_new_recipient: raw.is_new_recipient,
            established_user_new_recipient: raw.established_user_new_recipient,
            account_age_days: raw.account_age_days,
            recipient_risk_profile_score: raw.recipient_risk_profile_score,
            is_new_device: raw.is_new_device,
            is_proxy_ip: raw.is_proxy_ip,
            ip_risk_score: raw.ip_risk_score,
            country_mismatch: raw.country_mismatch
          })
        })

        clearTimeout(timeout)

        if (res.ok) {
          const scored = await res.json()
          const processed = {
            ...raw,
            ensembleScore: scored.risk_score / 100,
            lgbScore: scored.supervised_score,
            isoScore: scored.unsupervised_score,
            behScore: scored.behavioral_score,
            decision: scored.risk_level === 'LOW' ? 'APPROVE'
              : scored.risk_level === 'MEDIUM' ? 'FLAG' : 'BLOCK',
            riskLevel: scored.risk_level,
            reasons: scored.reasons,
            riskFactors: scored.reasons,
            scoredByBackend: true,
            latencyMs: Date.now() - startTick,
          }
          setAllTransactions(prev => [processed, ...prev].slice(0, 500))
        } else if (res.status === 503) {
          const errorTx = {
            ...raw,
            decision: 'BLOCK',
            riskLevel: 'HIGH',
            reasons: ['Engine Unavailable (503)'],
            scoredByBackend: false,
          }
          setAllTransactions(prev => [errorTx, ...prev].slice(0, 500))
        }
      } catch (err) {
        if (err.name !== 'AbortError') console.error('Prediction failed:', err)
      }
    }

    timeoutId = setTimeout(tick, 1000 / Math.max(params.simulationSpeed, 0.1))
    return () => clearTimeout(timeoutId)
  }, [isRunning, attackQueue, params.smoteLevel, params.simulationSpeed])

  const triggerAttackBurst = useCallback(() => setAttackQueue(20), []) // Generate 20 burst transactions
  const updateParam = useCallback((key, value) => setParams(p => ({ ...p, [key]: value })), [])
  const resetParams = useCallback(() => setParams({ simulationSpeed: 1, smoteLevel: 0.30 }), [])
  const clearData = useCallback(async () => {
    setAllTransactions([])
    try {
      await fetch('http://localhost:8000/api/v1/reset-stats', { method: 'POST' })
    } catch (err) {
      console.error('Failed to reset backend stats:', err)
    }
  }, [])

  // ─── Derived stats ────────────────────────────────────────────────────────
  const transactions = allTransactions.slice(0, 50)

  // Use backend stats when available for global consistency, fallback to local for session context
  const displayTotal = backendStats?.total_transactions ?? allTransactions.length
  const displayApproved = backendStats?.approved ?? allTransactions.filter(t => t.decision === 'APPROVE').length
  const displayFlagged = backendStats?.flagged ?? allTransactions.filter(t => t.decision === 'FLAG').length
  const displayBlocked = backendStats?.blocked ?? allTransactions.filter(t => t.decision === 'BLOCK').length

  const total = displayTotal
  const approved = displayApproved
  const flagged = displayFlagged
  const blocked = displayBlocked

  const avgLatency = useMemo(() => {
    if (total === 0) return 0
    return Math.round(allTransactions.reduce((s, t) => s + (t.latencyMs || 0), 0) / total)
  }, [allTransactions, total])

  // ─── Fix 6: Confusion matrix using real ground truth ──────────────────────
  const matrix = useMemo(() => {
    let tp = 0, fp = 0, fn = 0, tn = 0
    allTransactions.forEach(t => {
      const predicted = t.decision !== 'APPROVE'
      if (predicted && t.isFraud) tp++
      else if (predicted && !t.isFraud) fp++
      else if (!predicted && t.isFraud) fn++
      else tn++
    })
    const precision = tp + fp > 0 ? tp / (tp + fp) : 1
    const recall = tp + fn > 0 ? tp / (tp + fn) : 1
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 1
    const accuracy = tp + fp + fn + tn > 0 ? (tp + tn) / (tp + fp + fn + tn) : 1
    return { tp, fp, fn, tn, precision, recall, f1, accuracy }
  }, [allTransactions])

  const trends = useMemo(() => {
    const now = Date.now()
    const t30 = allTransactions.filter(t => (now - new Date(t.timestamp)) <= 30000)
    const t60 = allTransactions.filter(t => {
      const age = now - new Date(t.timestamp)
      return age > 30000 && age <= 60000
    })
    const rate = v => v.length ? v.filter(t => t.decision !== 'APPROVE').length / v.length : 0
    return {
      blockedRate: t30.length ? t30.filter(t => t.decision === 'BLOCK').length / t30.length : 0,
      flaggedRate: t30.length ? t30.filter(t => t.decision === 'FLAG').length / t30.length : 0,
      volume: t30.length,
      fraudRateNow: rate(t30),
      fraudRatePrev: rate(t60),
    }
  }, [allTransactions])

  // ─── Extra derived metrics ────────────────────────────────────────────────
  const lgbAvgScore = useMemo(() => {
    const scored = allTransactions.filter(t => t.lgbScore != null)
    if (scored.length === 0) return 0
    return scored.reduce((s, t) => s + t.lgbScore, 0) / scored.length
  }, [allTransactions])

  const isoAnomalyRate = useMemo(() => {
    const scored = allTransactions.filter(t => t.isoScore != null)
    if (scored.length === 0) return 0
    return (scored.filter(t => t.isoScore > 0.5).length / scored.length) * 100
  }, [allTransactions])

  const behHitRate = useMemo(() => {
    const scored = allTransactions.filter(t => t.reasons)
    if (scored.length === 0) return 0
    return (scored.filter(t => !t.reasons.includes('Normal behavior pattern')).length / scored.length) * 100
  }, [allTransactions])

  const modelDisagreements = useMemo(() => {
    return allTransactions
      .filter(t => t.lgbScore != null && t.isoScore != null)
      .filter(t => Math.abs(t.lgbScore - t.isoScore) > 0.3)
      .map(t => ({
        id: t.transaction_id || t.step,
        amount: t.amount,
        lgb: (t.lgbScore * 100).toFixed(1),
        iso: (t.isoScore * 100).toFixed(1),
        beh: (t.behScore * 100).toFixed(1),
        final: (t.ensembleScore * 100).toFixed(1),
        delta: Math.abs(t.lgbScore - t.isoScore).toFixed(2),
        reason: t.reasons?.[0] || 'Unknown'
      }))
  }, [allTransactions])

  const disagreementRate = useMemo(() => {
    const scored = allTransactions.filter(t => t.lgbScore != null && t.isoScore != null)
    if (scored.length === 0) return 0
    return (modelDisagreements.length / scored.length) * 100
  }, [allTransactions, modelDisagreements])

  const sparkline = useMemo(() => {
    const now = Date.now()
    const buckets = Array.from({ length: 12 }, () => ({ count: 0, fraud: 0 }))
    allTransactions.forEach(t => {
      const ageSecs = (now - new Date(t.timestamp)) / 1000
      if (ageSecs <= 60) {
        const idx = Math.min(11, Math.floor(ageSecs / 5))
        buckets[11 - idx].count++
        if (t.decision !== 'APPROVE') buckets[11 - idx].fraud++
      }
    })
    return buckets.map((b, i) => ({ t: i * 5, rate: b.count ? b.fraud / b.count : 0 }))
  }, [allTransactions])

  const histogram = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({
      bin: i * 0.1,
      label: `${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`,
      shortLabel: `${(i * 0.1).toFixed(1)}`,
      count: 0,
    }))
    allTransactions.forEach(t => {
      bins[Math.min(9, Math.floor(t.ensembleScore / 0.1))].count++
    })
    return bins
  }, [allTransactions])

  return {
    params, transactions, allTransactions,
    isRunning,
    matrix, trends, sparkline, histogram,
    total, blocked, flagged, approved, avgLatency,
    setIsRunning, updateParam, resetParams, triggerAttackBurst, clearData,
    // Backend state
    engineOnline,
    backendStats,
    config,
    // Derived Metrics
    lgbAvgScore,
    isoAnomalyRate,
    behHitRate,
    disagreementRate,
    modelDisagreements,
    // expose weights/thresholds from config
    weights: config.weights,
    thresholds: { approve: config.approve_threshold, block: config.flag_threshold },
  }
}
