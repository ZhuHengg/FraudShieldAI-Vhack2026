import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { generateTransaction } from '../utils/fraudScoring'

/**
 * useTransactionEngine
 *
 * Generates synthetic transactions and sends them to the real backend
 * at POST /predict for scoring.  NO frontend scoring fallback.
 * If the backend is offline, transactions are silently discarded.
 */

// ── Global shared store — survives tab switches ──────────────────────────────
if (!window.__fraudShieldStore) {
  window.__fraudShieldStore = {
    txnHistory: [],
    activeWeights: { lgb: 0.55, iso: 0.25, beh: 0.20 },
    activeThresholds: { approve: 35, flag: 60 },
  }
}

function pushToGlobalStore(txn) {
  const store = window.__fraudShieldStore
  store.txnHistory.push(txn)
  if (store.txnHistory.length > 500) store.txnHistory.shift()
  window.dispatchEvent(new CustomEvent('fraudshield:newtxn', { detail: txn }))
}

export function useTransactionEngine() {
  const [params, setParams] = useState({
    simulationSpeed: 1,   // transactions per second
    smoteLevel: 0.30,     // augments probability of high-risk features
  })

  const [allTransactions, setAllTransactions] = useState([])
  const [isRunning, setIsRunning] = useState(true)
  const [attackQueue, setAttackQueue] = useState(0)
  const [selectedTxn, setSelectedTxn] = useState(null)

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

  // ─── Initial Fetch of Transactions ──────────────────────────────────────────
  useEffect(() => {
    const fetchInitialTransactions = async () => {
      try {
        const res = await fetch('http://localhost:8000/api/v1/transactions');
        if (res.ok) {
          const data = await res.json();
          const mappedTransactions = data.map(dbTx => {
            const riskScore = dbTx.ml_risk_score * 100;
            let riskLevel = 'LOW';
            if (dbTx.action_taken === 'BLOCK') riskLevel = 'HIGH';
            else if (dbTx.action_taken === 'FLAG') riskLevel = 'MEDIUM';

            return {
              transaction_id: dbTx.transaction_id,
              id: dbTx.transaction_id,
              name_sender: dbTx.user_hash,
              userId: dbTx.user_hash,
              name_recipient: dbTx.recipient_hash,
              receiverId: dbTx.recipient_hash,
              transfer_type: dbTx.transfer_type,
              amount: dbTx.amount,
              avg_transaction_amount_30d: dbTx.avg_transaction_amount_30d,
              amount_vs_avg_ratio: dbTx.amount_vs_avg_ratio,
              transaction_hour: dbTx.transaction_hour,
              is_weekend: dbTx.is_weekend,
              sender_account_fully_drained: dbTx.sender_account_fully_drained,
              is_new_device: dbTx.is_new_device,
              isNewDevice: dbTx.is_new_device,
              session_duration_seconds: dbTx.session_duration_seconds,
              sessionDurationSeconds: dbTx.session_duration_seconds,
              failed_login_attempts: dbTx.failed_login_attempts,
              is_proxy_ip: dbTx.is_proxy_ip,
              isProxyIp: dbTx.is_proxy_ip,
              ip_risk_score: dbTx.ip_risk_score,
              ipRiskScore: dbTx.ip_risk_score,
              country_mismatch: dbTx.country_mismatch,
              countryMismatch: dbTx.country_mismatch,
              account_age_days: dbTx.account_age_days,
              accountAgeDays: dbTx.account_age_days,
              tx_count_24h: dbTx.tx_count_24h,
              txCount24h: dbTx.tx_count_24h,
              is_new_recipient: dbTx.is_new_recipient,
              isNewRecipient: dbTx.is_new_recipient,
              established_user_new_recipient: dbTx.established_user_new_recipient,
              recipient_risk_profile_score: dbTx.recipient_risk_profile_score,
              isFraud: dbTx.is_fraud === 1,
              decision: dbTx.action_taken,
              riskLevel: riskLevel,
              riskScore: riskScore,
              ensembleScore: dbTx.ml_risk_score,
              ground_truth: dbTx.is_fraud === 1 ? 'FRAUD' : 'LEGIT',
              template: dbTx.is_fraud === 1 ? 'attack' : 'normal',
              timestamp: new Date().toISOString(),
              scoredByBackend: true,
              sender_balance_before: dbTx.sender_balance_before,
              sender_balance_after: dbTx.sender_balance_after,
              receiver_balance_before: dbTx.receiver_balance_before,
              receiver_balance_after: dbTx.receiver_balance_after,
              currency: dbTx.currency,
              country: dbTx.country,
              deviceType: dbTx.device_type,
            };
          });
          
          setAllTransactions(mappedTransactions);
          // Optional: also push to global store if needed for other tabs, though fetching on each tab works too.
          window.__fraudShieldStore.txnHistory = [...mappedTransactions, ...window.__fraudShieldStore.txnHistory].slice(0, 500);
        }
      } catch (err) {
        console.error("Failed to fetch initial transactions:", err);
      }
    };
    
    fetchInitialTransactions();
  }, []);

  // ─── Fix 3: Backend-only transaction scoring loop ─────────────────────────
  useEffect(() => {
    if (!isRunning && attackQueue === 0) return

    let timeoutId
    const tick = async () => {
      const isAttackMode = attackQueue > 0

      // Select template naturally based on smoteLevel fraud rate, or override during an attack
      let template = 'normal'
      if (isAttackMode) {
        template = Math.random() > 0.4 ? 'attack' : 'suspicious'
      } else {
        // Natural mix based on slider
        const rand = Math.random()
        if (rand < params.smoteLevel) {
          template = Math.random() > 0.6 ? 'attack' : 'suspicious'
        }
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
          
          // --- Model Tuning Lab Integration ---
          // Only re-score if the user has explicitly applied custom tuning;
          // otherwise passthrough the backend's own risk_score / risk_level.
          const activeTuning = window.__activeTuning;

          const lgbScore = (scored.supervised_score || 0) * 100;
          const isoScore = (scored.unsupervised_score || 0) * 100;
          const behScore = (scored.behavioral_score || 0) * 100;

          let tunedScore, tunedDecision;
          if (activeTuning) {
            tunedScore = Math.min(100, Math.max(0,
              lgbScore * activeTuning.weights.lgb +
              isoScore * activeTuning.weights.iso +
              behScore * activeTuning.weights.beh
            ));
            tunedDecision =
              tunedScore < activeTuning.thresholds.approve ? 'APPROVE' :
              tunedScore < activeTuning.thresholds.flag    ? 'FLAG' : 'BLOCK';
          } else {
            // Use backend's authoritative score and decision directly
            tunedScore = scored.risk_score;
            tunedDecision =
              scored.risk_level === 'LOW' ? 'APPROVE' :
              scored.risk_level === 'MEDIUM' ? 'FLAG' : 'BLOCK';
          }

          const processed = {
            ...raw,
            // Core scores for tuning lab
            supervised_score: scored.supervised_score,
            unsupervised_score: scored.unsupervised_score,
            behavioral_score: scored.behavioral_score,
            rule_breakdown: scored.rule_breakdown ?? null,
            feature_snapshot: scored.feature_snapshot ?? null,
            
            // Re-scoring logic applied to live stream
            ensembleScore: (tunedScore / 100),
            riskScore: tunedScore,
            decision: tunedDecision,
            riskLevel: tunedDecision === 'BLOCK' ? 'HIGH' : tunedDecision === 'FLAG' ? 'MEDIUM' : 'LOW',
            
            // Ground truth for performance metrics
            ground_truth: raw.isFraud ? 'FRAUD' : 'LEGIT',

            // Legacy field support
            lgbScore: scored.supervised_score,
            isoScore: scored.unsupervised_score,
            behScore: scored.behavioral_score,
            reasons: scored.reasons,
            riskFactors: scored.reasons,
            scoredByBackend: true,
            latencyMs: Date.now() - startTick,

            // Extended backend fields for dashboard analytics
            rule_breakdown: scored.rule_breakdown ?? null,
            feature_snapshot: scored.feature_snapshot ?? null,

            // CamelCase aliases for RiskRadar.jsx data binding
            // (these override the raw snake_case originals with backend-confirmed values)
            isProxyIp: scored.feature_snapshot?.is_proxy_ip ?? raw.is_proxy_ip ?? 0,
            countryMismatch: scored.feature_snapshot?.country_mismatch ?? raw.country_mismatch ?? 0,
            isNewDevice: scored.feature_snapshot?.is_new_device ?? raw.is_new_device ?? 0,
            ipRiskScore: scored.feature_snapshot?.ip_risk_score ?? raw.ip_risk_score ?? 0,
            senderFullyDrained: scored.feature_snapshot?.sender_fully_drained ?? raw.sender_account_fully_drained ?? 0,
            isNewRecipient: scored.feature_snapshot?.is_new_recipient ?? raw.is_new_recipient ?? 0,
            amountVsAvgRatio: scored.feature_snapshot?.amount_vs_avg_ratio ?? raw.amount_vs_avg_ratio ?? 1,
            txCount24h: scored.feature_snapshot?.tx_count_24h ?? raw.tx_count_24h ?? 0,
            sessionDurationSeconds: scored.feature_snapshot?.session_duration_seconds ?? raw.session_duration_seconds ?? 0,
            accountAgeDays: scored.feature_snapshot?.account_age_days ?? raw.account_age_days ?? 0,
          }
          pushToGlobalStore(processed)
          setAllTransactions(prev => [processed, ...prev].slice(0, 500))

          // Fire-and-forget: save to Neon DB for closed-loop retraining
          try {
            fetch('http://localhost:8000/api/v1/transactions', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                transaction_id: raw.transaction_id || raw.id,
                user_hash: raw.name_sender || raw.userId || '',
                recipient_hash: raw.name_recipient || raw.receiverId || '',
                transfer_type: raw.transfer_type || raw.transaction_type || 'TRANSFER',
                amount: raw.amount || 0,
                avg_transaction_amount_30d: raw.avg_transaction_amount_30d || 0,
                amount_vs_avg_ratio: raw.amount_vs_avg_ratio || 1,
                transaction_hour: raw.transaction_hour || new Date().getHours(),
                is_weekend: raw.is_weekend || 0,
                sender_account_fully_drained: raw.sender_account_fully_drained || 0,
                is_new_device: raw.is_new_device || 0,
                session_duration_seconds: raw.session_duration_seconds || 0,
                failed_login_attempts: raw.failed_login_attempts || 0,
                is_proxy_ip: raw.is_proxy_ip || 0,
                ip_risk_score: raw.ip_risk_score || 0,
                country_mismatch: raw.country_mismatch || 0,
                account_age_days: raw.account_age_days || 0,
                tx_count_24h: raw.tx_count_24h || 0,
                is_new_recipient: raw.is_new_recipient || 0,
                established_user_new_recipient: raw.established_user_new_recipient || 0,
                recipient_risk_profile_score: raw.recipient_risk_profile_score || 0,
                is_fraud: raw.isFraud ? 1 : 0,
                action_taken: tunedDecision,
                ml_risk_score: (tunedScore / 100),
                sender_balance_before: raw.sender_balance_before || 0,
                sender_balance_after: raw.sender_balance_after || 0,
                receiver_balance_before: raw.receiver_balance_before || 0,
                receiver_balance_after: raw.receiver_balance_after || 0,
                currency: raw.currency || 'MYR',
                country: raw.country || 'MY',
                device_type: raw.deviceType || 'Mobile',
              }),
            }).catch(() => {}) // silently ignore save errors
          } catch {}

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
        if (err.name !== 'AbortError') {
          console.error('Prediction failed:', err)
          // No mock data allowed on frontend. Transaction dropped if engine offline.
        }
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
    if (allTransactions.length === 0) return 0
    return Math.round(allTransactions.reduce((s, t) => s + (t.latencyMs || 0), 0) / allTransactions.length)
  }, [allTransactions])

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
        id: t.transaction_id || t.id,
        amount: t.amount,
        lgbScore: t.lgbScore,
        isoScore: t.isoScore,
        behScore: t.behScore,
        ensembleScore: t.ensembleScore,
        delta: Math.abs((t.lgbScore || 0) - (t.isoScore || 0)),
        reason: t.reasons?.[0] || 'Unknown',
        _raw: t
      }))
  }, [allTransactions])

  const disagreementRate = useMemo(() => {
    const scored = allTransactions.filter(t => t.lgbScore != null && t.isoScore != null)
    if (scored.length === 0) return 0
    return (modelDisagreements.length / scored.length) * 100
  }, [allTransactions, modelDisagreements])

  // ─── Fix: Decoupled Rolling Sparkline Chart (60 seconds) ────────────────
  const [sparkline, setSparkline] = useState(Array.from({ length: 60 }, (_, i) => ({ time: i, rate: 0 })))
  
  // Keep an up-to-date ref for the interval to read from without re-triggering
  const txRef = useRef(allTransactions)
  useEffect(() => { txRef.current = allTransactions }, [allTransactions])

  useEffect(() => {
    if (!isRunning) return
    
    const intervalId = setInterval(() => {
      const now = Date.now()
      // Find transactions occurring strictly in the last 1 second
      const recentTxns = txRef.current.filter(t => {
          const tTime = new Date(t.timestamp).getTime()
          return (now - tTime) <= 1000 && (now - tTime) >= 0
      })
      
      // Calculate instantaneous fraud rate for this 1-second interval
      const newRate = recentTxns.length > 0 
        ? recentTxns.filter(t => t.decision !== 'APPROVE').length / recentTxns.length 
        : 0
        
      setSparkline(prev => {
        const next = [...prev, { time: now, rate: newRate }]
        if (next.length > 60) next.shift()
        return next
      })
    }, 1000)
    
    return () => clearInterval(intervalId)
  }, [isRunning])

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
    selectedTxn, setSelectedTxn,
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
