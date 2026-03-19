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

  // Helper to generate realistic amounts based on type and attack statys
  const generateAmount = (type, isAttackEvent = false) => {
    if (isAttackEvent) {
      const userAvg = 500; // Boosted average
      // Generate amounts that are strictly high out-of-bounds to trigger model (e.g. 10,000 to 45,000)
      return userAvg * (20 + Math.random() * 70);
    }
    
    const ranges = {
      'PAYMENT':  [10, 300],    // daily purchases
      'TRANSFER': [50, 2000],   // P2P transfers  
      'CASH_OUT': [100, 1000],  // ATM-equivalent
    };
    const [min, max] = ranges[type] || [50, 500];
    return parseFloat((min + Math.random() * (max - min)).toFixed(2));
  }

  if (isSuspicious) {
    transfer_type = 'CASH_OUT'
    amount = parseFloat(randFloat(5000, 20000).toFixed(2)) // Keep suspicious high but bounded
  } else if (isAttack) {
    transfer_type = 'CASH_OUT'
    amount = generateAmount(transfer_type, true)
  } else {
    // Normal
    transfer_type = pick(['TRANSFER', 'PAYMENT', 'CASH_OUT'])
    amount = generateAmount(transfer_type, false)
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

  // --- Introduce Stochastic Overlap (Noise) ---
  // ~15% chance a normal user exhibits a high-risk trait (e.g., using a VPN, fast session)
  // ~10% chance a fraudster exhibits a normal trait (e.g., slow session, not drained)
  const applyNoise = (highRiskVal, normalVal) => {
      if (isHighRisk && Math.random() < 0.10) return normalVal();
      if (!isHighRisk && Math.random() < 0.15) return highRiskVal();
      return isHighRisk ? highRiskVal() : normalVal();
  }

  // Security signals
  const is_new_device    = applyNoise(() => (Math.random() > 0.3 ? 1 : 0), () => (Math.random() > 0.95 ? 1 : 0));
  const is_proxy_ip      = applyNoise(() => (Math.random() > 0.4 ? 1 : 0), () => (Math.random() > 0.90 ? 1 : 0)); // Normal users use VPNs too (~10%)
  const ip_risk_score    = applyNoise(
      () => parseFloat(clamp(randFloat(0.5, 1.0) + Math.random() * smoteLevel * 0.4, 0, 1).toFixed(3)),
      () => parseFloat(clamp(randFloat(0.0, 0.4), 0, 1).toFixed(3))
  );
  const failed_login_attempts = applyNoise(() => randInt(1, 4), () => (Math.random() > 0.9 ? randInt(1, 2) : 0));

  // Account status
  const sender_account_fully_drained = applyNoise(() => (Math.random() > 0.4 ? 1 : 0), () => (Math.random() > 0.98 ? 1 : 0));
  const account_age_days             = applyNoise(() => randInt(1, 30), () => randInt(90, 2000));
  const tx_count_24h                 = applyNoise(() => randInt(8, 20), () => randInt(1, 4));

  // Trust profile
  const country_mismatch              = applyNoise(() => (Math.random() > 0.5 ? 1 : 0), () => (Math.random() > 0.90 ? 1 : 0));
  const is_new_recipient              = applyNoise(() => (Math.random() > 0.6 ? 1 : 0), () => (Math.random() > 0.80 ? 1 : 0));
  const established_user_new_recipient = (account_age_days > 180 && is_new_recipient) ? 1 : 0

  // Missing fields from real dataset
  const session_duration_seconds     = applyNoise(() => randInt(5, 60), () => randInt(120, 2400));
  const recipient_risk_profile_score = applyNoise(() => randFloat(0.6, 1.0), () => randFloat(0.0, 0.3));

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
