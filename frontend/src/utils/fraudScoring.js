/**
 * fraudScoring.js
 *
 * Generates synthetic transactions whose field names match the real backend
 * TransactionInput schema exactly.  Scoring is done entirely by the backend.
 */

// ─── Helpers ──────────────────────────────────────────────────────────────────
const randFloat = (min, max) => Math.random() * (max - min) + min
const randInt   = (min, max) => Math.floor(randFloat(min, max + 1))
const pick      = arr => arr[Math.floor(Math.random() * arr.length)]
const clamp     = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const uid       = pfx => `${pfx}-${Math.random().toString(36).substring(2, 7).toUpperCase()}`

// ─── Main generator ───────────────────────────────────────────────────────────
export function generateTransaction(template = 'normal', smoteLevel = 0.3) {
  const now    = new Date()
  const hour   = now.getHours()
  const dow    = now.getDay()

  const isAttack = template === 'attack'
  const isSuspicious = template === 'suspicious'
  const isHighRisk = isAttack || isSuspicious

  // Identity
  const transaction_id  = uid('TXN')
  const name_sender     = Math.floor(Math.random() * 9000000000 + 1000000000).toString()
  const name_recipient  = Math.floor(Math.random() * 9000000000 + 1000000000).toString()

  // Amount & Type per template
  let amount, transfer_type
  if (isSuspicious) {
    amount = parseFloat(randFloat(5000, 50000).toFixed(2))
    transfer_type = 'CASH_OUT'
  } else if (isAttack) {
    amount = parseFloat(randFloat(10000, 25000).toFixed(2))
    transfer_type = 'CASH_OUT'
  } else {
    // Normal
    amount = parseFloat(randFloat(100, 500).toFixed(2))
    transfer_type = 'TRANSFER'
  }

  // 30-day average
  const avg_transaction_amount_30d = parseFloat(
    isHighRisk
      ? randFloat(200, 800).toFixed(2)
      : randFloat(100, amount * 1.5 + 100).toFixed(2)
  )
  const amount_vs_avg_ratio = parseFloat(
    (amount / Math.max(avg_transaction_amount_30d, 1)).toFixed(3)
  )

  // Temporal
  const transaction_hour = hour
  const is_weekend       = [0, 6].includes(dow) ? 1 : 0

  // Security signals
  const is_new_device    = isHighRisk ? (Math.random() > 0.4 ? 1 : 0) : (Math.random() > 0.95 ? 1 : 0)
  const is_proxy_ip      = isHighRisk ? (Math.random() > 0.5 ? 1 : 0) : (Math.random() > 0.98 ? 1 : 0)
  const ip_risk_score    = parseFloat(
    isHighRisk
      ? clamp(randFloat(0.5, 1.0) + Math.random() * smoteLevel * 0.4, 0, 1).toFixed(3)
      : clamp(randFloat(0.0, 0.2), 0, 1).toFixed(3)
  )
  const failed_login_attempts = isHighRisk ? randInt(1, 5) : (Math.random() > 0.95 ? randInt(1, 2) : 0)

  // Account status
  const sender_account_fully_drained = isSuspicious || (isAttack && Math.random() > 0.3) ? 1 : (Math.random() > 0.98 ? 1 : 0)
  const account_age_days             = isHighRisk ? randInt(1, 30)  : randInt(180, 2000)
  const tx_count_24h                 = isHighRisk ? randInt(8, 25)  : randInt(1, 3)

  // Trust profile
  const country_mismatch              = isHighRisk ? (Math.random() > 0.6 ? 1 : 0) : (Math.random() > 0.95 ? 1 : 0)
  const is_new_recipient              = isHighRisk ? (Math.random() > 0.7 ? 1 : 0) : (Math.random() > 0.90 ? 1 : 0)
  const established_user_new_recipient = (account_age_days > 180 && is_new_recipient) ? 1 : 0

  // Missing fields from real dataset
  const session_duration_seconds     = isHighRisk ? randInt(5, 45)      : randInt(120, 2400)
  const recipient_risk_profile_score = isHighRisk ? randFloat(0.7, 1.0) : randFloat(0.0, 0.2)

  // Sender balance simulation
  const sender_balance_before = parseFloat(randFloat(amount, amount * 1.5 + 500).toFixed(2))
  // For suspicious/attack, balance after is ~0 if drained
  const sender_balance_after  = sender_account_fully_drained ? randFloat(0, 10) : parseFloat((sender_balance_before - amount).toFixed(2))

  // Receiver balance simulation
  const receiver_balance_before = parseFloat(randFloat(100, 10000).toFixed(2))
  const receiver_balance_after  = parseFloat((receiver_balance_before + amount).toFixed(2))

  const txn = {
    transaction_id, name_sender, name_recipient, transfer_type, amount,
    avg_transaction_amount_30d, amount_vs_avg_ratio,
    transaction_hour, is_weekend,
    is_new_device, failed_login_attempts, is_proxy_ip, ip_risk_score,
    sender_account_fully_drained, account_age_days, tx_count_24h,
    country_mismatch, is_new_recipient, established_user_new_recipient,
    session_duration_seconds, recipient_risk_profile_score,
    sender_balance_before, sender_balance_after,
    receiver_balance_before, receiver_balance_after,

    // UI and Context fields
    id:         transaction_id,
    userId:     name_sender,
    receiverId: name_recipient,
    timestamp:  now.toISOString(),
    currency:   'MYR',
    country: pick(['MY', 'SG', 'TH', 'ID', 'PH', 'VN', 'MM', 'KH']),
    deviceType: pick(['Mobile', 'Desktop', 'Tablet', 'API-Direct']),

    // Ground truth
    isFraud: isHighRisk,
    template,
  }

  return txn
}
