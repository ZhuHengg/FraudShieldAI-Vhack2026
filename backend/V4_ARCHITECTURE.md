# FraudShield V4 Architecture — Implementation Guide

## Overview

This document maps the 8-step **Real-Time Fraud Shield Architecture (V4)** against the actual FraudShield codebase. For each step, it explains what was implemented, what was intentionally omitted, and why.

---

## Architecture Map

| Step | Name | Status | Implementation |
|------|------|--------|----------------|
| 1 | Session-Level Behavioral Biometrics | **Partial** | Proxy signals only |
| 2 | Cold-Start Gate | **Partial** | Account-age rules in BehavioralProfiler |
| 3 | Privacy-Preserving Data Enrichment | **Implemented** | `privacy.py` + SHA-256 hashing |
| 4 | Network-Level Velocity Check | **Implemented** | `velocity.py` — Anti-Mule Layer |
| 5 | Resilient Hybrid AI Engine | **Implemented** | HA Fallback cascade in `inference.py` |
| 6 | Calibrated Logic Matrix | **Implemented** | PR-curve thresholds + `/api/v1/calibration` |
| 7 | 3-Tier Action Execution | **Partial** | APPROVE/FLAG/BLOCK in UI, no SMS/USSD |
| 8 | Quarantine Retraining Loop | **Implemented** | `quarantine.py` + auto-quarantine pipeline |

---

## Step-by-Step Breakdown

### Step 1: Session-Level Behavioral Biometrics — `PARTIAL`

**What the V4 spec says:**
> Measure typing cadence, swipe pressure, navigation speed to verify the human, not just the device.

**What we implemented:**
- `session_duration_seconds` — proxy for navigation speed
- `failed_login_attempts` — proxy for unauthorized access attempts
- `is_new_device` — device fingerprint change detection
- These signals feed into the Behavioral Profiler's `rapid_session` and `risky_context` rules

**What we did NOT implement and why:**
- **Touch pressure / typing cadence / accelerometer data** — These require a native mobile SDK (Android `MotionEvent.getPressure()`, iOS `UITouch.force`). Our system is a web-based API; there is no mobile instrumentation layer to capture these signals. Implementing this would require shipping a companion mobile SDK, which is outside the scope of a backend fraud detection engine.
- **Continuous biometric verification** — Same reason. This is a device-side capability, not a server-side capability.

**Files involved:**
- [`behavioural.py`](api/behavioural.py) — BehavioralProfiler rules
- [`inference.py`](api/inference.py) — `_score_behavioral()` method (lines 202-243)

---

### Step 2: Cold-Start Gate — `PARTIAL`

**What the V4 spec says:**
> For accounts < 30 days old, AI operates in "listen-only" mode with a static rules sandbox. Once a statistically significant profile is built, AI seamlessly takes over.

**What we implemented:**
- `account_age_days` is a feature used by both LightGBM and the Behavioral Profiler
- New accounts with high-risk signals (drain + new recipient + new device) are scored more aggressively by the ensemble
- The EMA cache in `inference.py` (line 130-146) builds per-user baselines progressively, serving as a lightweight cold-start handler

**What we did NOT implement and why:**
- **Formal "listen-only" sandbox mode** — Would require a separate execution path where the AI scores but doesn't act, with a hard cutover at N transactions. This adds complexity that isn't justified for a hackathon demo — the current system already handles new accounts conservatively through the `account_age_days` feature and the `TYPE_DEFAULTS` fallback for unknown users.
- **Graduated trust levels** (e.g., daily transfer caps) — These are business-logic constraints that belong in the wallet application layer, not the fraud detection engine.

**Files involved:**
- [`inference.py`](api/inference.py) — EMA cache and `TYPE_DEFAULTS` (lines 131-146)

---

### Step 3: Privacy-Preserving Data Enrichment — `IMPLEMENTED`

**What the V4 spec says:**
> Raw identifiable data undergoes one-way cryptographic hashing at the edge. Use Aggregated Statistical Profiles instead of raw transaction history.

**What we implemented:**
- `PrivacyProtector` in [`privacy.py`](api/privacy.py) — SHA-256 salted hashing of `sender_id` and `receiver_id` before inference
- PII is hashed at the API boundary (`prepare_for_inference()`) before any model sees it
- `user_hash` and `recipient_hash` stored in DB (never raw IDs)
- Aggregated Statistical Profile via in-memory EMA cache (`user_avg_cache`) — computes rolling averages without storing raw transaction history
- `PrivacyInfo` metadata attached to every `RiskResponse` confirming PII was hashed

**What we did NOT implement and why:**
- **90-day data purge** — This is an operational policy, not an engine feature. Would be a simple cron job (`DELETE FROM transaction_logs WHERE created_at < NOW() - INTERVAL '90 days'`).
- **Differential privacy on model outputs** — `add_dp_noise()` exists in `privacy.py` but is not applied to live scores (`dp_applied=False`). Adding Laplace noise to real-time fraud scores would degrade detection accuracy. It's useful only for aggregate reporting exports.

**Files involved:**
- [`privacy.py`](api/privacy.py) — PrivacyProtector class
- [`inference.py`](api/inference.py) — `preprocess()` calls `prepare_for_inference()` (line 94)

---

### Step 4: Network-Level Velocity Check (Anti-Mule Layer) — `IMPLEMENTED`

**What the V4 spec says:**
> If a destination wallet receives funds from 10+ unlinked, brand-new senders within 60 minutes, trigger automatic BLOCK override.

**What we implemented:**
- `RecipientVelocityTracker` in [`velocity.py`](api/velocity.py)
- **In-memory sliding-window graph**: `dict[recipient_hash] → {sender_set, timestamps}`
- **Detection**: 10+ unique senders within 3600-second (60-min) window → `mule_detected = True`
- **Auto-decay**: Entries older than the window are pruned on each check (no unbounded memory growth)
- **Thread-safe**: Uses `threading.Lock` for concurrent async workers
- **Short-circuit**: Called in `predict()` BEFORE model scoring. If mule detected → immediate `RiskResponse(risk_score=100, risk_level="HIGH", mule_flag=True, engine_mode="mule_override")`
- **Monitoring**: `GET /api/v1/velocity/stats` returns total checks, detections, tracked recipients

**What we did NOT implement and why:**
- **Graph-based community detection** (e.g., PageRank on the sender-recipient graph) — Would require a graph database like Neo4j or a real-time graph engine. The sliding-window approach achieves the core detection goal with O(1) per-check amortized cost using only in-memory dicts.
- **Persistent velocity state** — Current state is in-memory and resets on server restart. For production, this should be backed by Redis with TTL keys.

**Files involved:**
- [`velocity.py`](api/velocity.py) — RecipientVelocityTracker (new file)
- [`inference.py`](api/inference.py) — Integration in `predict()` (lines 301-324)
- [`main.py`](api/main.py) — `GET /api/v1/velocity/stats` endpoint

---

### Step 5: Resilient Hybrid AI Engine (HA Fallback) — `IMPLEMENTED`

**What the V4 spec says:**
> If Model B times out, Model A takes over but caps transfer limits. If both crash, default to lightweight rules-based bypass.

**What we implemented — 5-tier degradation cascade:**

```
Level 0: mule_override     → Mule detected, skip all models, BLOCK
Level 1: full              → LGB + ISO + BEH (normal operation)
Level 2: degraded_iso      → LGB + BEH, re-normalized weights
Level 3: degraded_lgb      → ISO + BEH, re-normalized weights
Level 4: behavioral_only   → BEH only (pure rules, no ML)
Level 5: static_rules      → All models down → amount-based rules
```

- Each model is wrapped in individual `try/except` blocks (lines 331-356)
- **Weight re-normalization**: When one model fails, remaining weights are divided by their sum to maintain proper probability distribution
- **Static rules fallback** (`_static_rules_score`): Conservative amount-based scoring as absolute last resort — `<1K + low risk → APPROVE`, `>5K or high risk → BLOCK`, else `FLAG`
- `engine_mode` and `active_models` fields in every `RiskResponse` expose which tier was used
- All degradation events are logged at WARNING/ERROR level

**What we did NOT implement and why:**
- **Transfer limit capping in degraded mode** — This is a business-logic constraint that belongs in the wallet app. The fraud engine's job is to score and classify; the wallet enforces limits based on the score.
- **Timeout-based model switching** (e.g., 50ms deadline per model) — Would require async model invocation with `asyncio.wait_for()`. Current models run in <15ms so timeouts aren't a practical concern. For production with remote model serving, this would be essential.

**Files involved:**
- [`inference.py`](api/inference.py) — `predict()` method (lines 294-460), `_static_rules_score()` (lines 256-292)
- [`schemas.py`](api/schemas.py) — `engine_mode`, `active_models` fields on `RiskResponse`

---

### Step 6: Calibrated Logic Matrix — `IMPLEMENTED`

**What the V4 spec says:**
> Thresholds for Approval, Flagging, or Blocking are derived mathematically from the Precision-Recall curve of the validation dataset to guarantee near-zero False Positive Rate.

**What we implemented:**
- **Thresholds derived from PR curve**: `score_fusion.py` grid search finds the optimal threshold that maximizes F1 on the validation set
- **Calibration provenance**: Every `RiskResponse` includes `calibration_source` field (e.g., `"PR curve argmax F1 on val set"`)
- **Calibration endpoint**: `GET /api/v1/calibration` returns:
  - Current approve/flag thresholds
  - Their exact mathematical origin
  - Validation metrics at those thresholds (precision, recall, F1, FPR)
  - Ensemble weights and retrain metadata
- **Retrain recalibration**: When `POST /api/v1/retrain` is triggered, new thresholds are derived from the labeled data's PR curve, and full calibration metrics (precision, recall, FPR) are computed and stored in `ensemble_config.json`

**What we did NOT implement and why:**
- **Dynamic FPR-constrained optimization** (e.g., "find threshold where FPR < 0.1%") — Current implementation maximizes F1, which implicitly controls FPR. A separate FPR constraint could be added as a parameter to the grid search if regulators require a specific FPR guarantee.

**Files involved:**
- [`retrain.py`](api/retrain.py) — Grid search + calibration metric computation (lines 230-240)
- [`inference.py`](api/inference.py) — Threshold application + calibration source tracking (lines 398-414)
- [`main.py`](api/main.py) — `GET /api/v1/calibration` endpoint
- [`schemas.py`](api/schemas.py) — `CalibrationResponse` schema

---

### Step 7: 3-Tier Action Execution — `PARTIAL`

**What the V4 spec says:**
> Approve (instant clear), Block (halt + dispute button), Flag (delayed settlement + push/SMS/USSD with accessible UI).

**What we implemented:**
- **3-tier decision logic**: `APPROVE` (LOW risk), `FLAG` (MEDIUM), `BLOCK` (HIGH) — fully operational in both backend and frontend
- **Frontend visualization**: Live transaction stream shows color-coded decisions, Model Tuning Lab shows confusion matrix and precision/recall impact of threshold changes
- **Investigation & SHAP**: `POST /api/v1/explain/{id}` provides per-feature SHAP explanations for any blocked/flagged transaction

**What we did NOT implement and why:**
- **"Dispute This" button / human agent routing** — This is a CRM/ticketing integration (e.g., Zendesk, Freshdesk). Outside the scope of the detection engine.
- **Delayed Settlement (escrow)** — Requires integration with the payment processing layer. The fraud engine flags the transaction; the payment system decides whether to hold funds.
- **SMS / USSD / push notification degradation** — Requires a mobile messaging gateway (Twilio, Firebase Cloud Messaging). These are delivery-channel concerns, not detection concerns.
- **Local-language audio playback / universal color-coding** — Frontend accessibility features that would be part of the wallet app's UX layer.

**Files involved:**
- [`inference.py`](api/inference.py) — Decision logic (lines 398-406)
- Frontend: `FraudSimulator.jsx`, `FraudAnalysis.jsx`, `Dashboard.jsx`

---

### Step 8: Quarantine Retraining Loop — `IMPLEMENTED`

**What the V4 spec says:**
> If a user taps "YES" to approve an anomalous transaction, the data is quarantined. An offline script validates it against biometric continuity and macro-regional spending trends before merging into the AI.

**What we implemented:**
- **Auto-quarantine trigger**: When an analyst labels a FLAGGED transaction as LEGIT (the exact grooming pattern), it automatically enters `QUARANTINED` status instead of being directly trusted
- **3-check validation pipeline** in [`quarantine.py`](api/quarantine.py):
  1. **Biometric Continuity** — Same device type as user's last 5 transactions + proxy IP consistency
  2. **Amount Plausibility** — Transaction amount within 3 standard deviations of user's rolling mean
  3. **Regional Trend Check** — Amount within 5x the regional average for the same hour/country
- **Graduation rule**: Must pass ≥ 2 of 3 checks to move from `QUARANTINED` → `VALIDATED`
- **Retraining filter**: `retrain.py` excludes `QUARANTINED` and `REJECTED` labels — only `NULL` (direct) and `VALIDATED` labels enter the training pool
- **Quarantine DB columns**: `quarantine_status` (NULL/QUARANTINED/VALIDATED/REJECTED) + `quarantine_reason`
- **API endpoints**:
  - `GET /api/v1/quarantine/stats` — Breakdown of pending/validated/rejected/direct
  - `POST /api/v1/quarantine/validate` — Run validation on all pending quarantined labels

**What we did NOT implement and why:**
- **Weekly cron-based validation** — The V4 spec mentions "an offline script runs weekly." We implemented something better: on-demand validation via API that can be triggered at any time. This is more responsive and testable. A cron wrapper could easily call this endpoint.
- **Biometric continuity via accelerometer/touch data** — Same limitation as Step 1. We use device fingerprint + proxy IP as a proxy.

**Files involved:**
- [`quarantine.py`](api/quarantine.py) — Validation logic (new file)
- [`main.py`](api/main.py) — Quarantine endpoints + auto-quarantine in feedback handler
- [`retrain.py`](api/retrain.py) — Quarantine-aware data filtering
- [`models.py`](api/models.py) — `quarantine_status`, `quarantine_reason` columns

---

### Step 9: LLM-Powered Investigation Assistant — `IMPLEMENTED`

**What the V4 spec says:**
> Integrate Gemini API to provide natural-language explanations of why a transaction was blocked. Input: SHAP values + rule breakdown + transaction features. Output: "This transaction was blocked because the sender's account was fully drained in a 15-second session from a Nigerian IP, which matches the profile of an account takeover attack."

**What we implemented:**
- `POST /api/v1/investigate` endpoint gathers SHAP, quarantine info, mule flags, and behavioral reasons, sending them to Gemini 2.0 for a natural-language report.
- Includes a deterministic fallback template if the API key is missing or the external API is unreachable, ensuring the UI never breaks.
- The synchronous Gemini SDK calls are wrapped in `asyncio.run_in_executor` to prevent blocking the high-throughput FastAPI event loop.

**What we did NOT implement and why:**
- **Streaming responses** — While LLMs support streaming, the UI currently expects a single JSON response payload. Streaming would require a WebSocket or SSE integration on the frontend.

**Files involved:**
- [`llm.py`](api/llm.py) — LLM integration and fallback logic
- [`main.py`](api/main.py) — Investigation endpoint

---

### Step 10: Model Version Rollback — `IMPLEMENTED`

**What the V4 spec says:**
> Before overwriting `ensemble_config.json` during retrain, save a timestamped backup. Add a `POST /api/v1/rollback` endpoint that restores the previous config if the new weights perform worse in production.

**What we implemented:**
- **Auto-Backups**: During the retraining cycle (`POST /api/v1/retrain`), the existing `ensemble_config.json` is automatically backed up with a UTC timestamp.
- **History Tracking**: `GET /api/v1/model/history` surfaces all past configurations, including their Validation F1 scores, allowing data scientists to compare model drift.
- **Hot-Reloading Rollback**: `POST /api/v1/model/rollback` safely swaps out the active ensemble weights and thresholds in memory. It also creates a "pre-rollback" snapshot to prevent accidental data loss.

**What we did NOT implement and why:**
- **Automated rollback triggers** — Rollbacks are currently manual via the analyst UI. Automated rollbacks based on live model performance degradation would require a complex live evaluation pipeline.

**Files involved:**
- [`retrain.py`](api/retrain.py) — Backup logic during retraining
- [`main.py`](api/main.py) — Rollback and history endpoints

---

### Step 11: Dynamic Recipient Risk Scoring — `IMPLEMENTED`

**What the V4 spec says:**
> Currently `recipient_risk_profile_score` is a static input field. It could be dynamically computed from the velocity tracker's data: recipients with higher unique-sender counts get higher risk scores, creating a feedback loop between Step 4 and Step 5.

**What we implemented:**
- **Velocity-Driven Feedback**: The `RecipientVelocityTracker` calculates a dynamic risk score between `0.0` and `1.0` based on the volume of unique senders in the sliding window.
- **Feature Injection**: During inference, if the dynamic risk exceeds the static baseline, the higher risk score is injected into the feature payload.
- **Result**: LightGBM and Isolation Forest models receive real-time signals about early-stage mule behaviour, allowing them to flag suspicious network growth before it reaches the hard 10-sender override threshold.

**What we did NOT implement and why:**
- **Decay factor for older velocities** — The dynamic risk score is purely linear based on the active sliding window. Adding an exponential decay factor would require more complex state tracking but could be a future enhancement.

**Files involved:**
- [`velocity.py`](api/velocity.py) — Dynamic scoring logic
- [`inference.py`](api/inference.py) — Feature injection before inference

---

## API Endpoints Summary

### Core
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/health` | Engine status |
| GET | `/api/v1/config` | Current weights & thresholds |
| POST | `/predict` | Score a transaction |
| POST | `/api/v1/explain/{id}` | SHAP explanation |

### V4 Step 4 — Anti-Mule
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/velocity/stats` | Mule tracker statistics |

### V4 Step 6 — Calibration
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/calibration` | Threshold provenance + metrics |

### V4 Step 8 — Quarantine
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/quarantine/stats` | Quarantine status breakdown |
| POST | `/api/v1/quarantine/validate` | Run validation on pending labels |

### V4 Step 9 — LLM Investigation (Improvement #9)
| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/llm/status` | Check Gemini LLM availability |
| POST | `/api/v1/investigate` | Get natural-language explanation of blocked transaction |

### Closed-Loop Retraining
| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/feedback` | Submit analyst label (auto-quarantines FLAG+LEGIT) |
| GET | `/api/v1/feedback/stats` | Labeling progress |
| POST | `/api/v1/retrain` | Trigger weight re-optimization |
| GET | `/api/v1/model/history` | List all saved ensemble config versions |
| POST | `/api/v1/model/rollback` | Rollback to previous config version |

---

## Suggested Improvements

### High Impact / Low Effort

1. **Redis-backed velocity state** — Replace the in-memory `RecipientVelocityTracker` with Redis sorted sets + TTL. This makes mule detection persistent across server restarts and enables horizontal scaling.

2. **FPR-constrained threshold optimization** — Add a `max_fpr` parameter to the grid search in `retrain.py`. Instead of pure F1 maximization, find the best F1 where `FPR ≤ max_fpr`. This gives regulators a hard guarantee.

4. **90-day data purge job** — Add a simple endpoint `POST /api/v1/admin/purge` that deletes transaction logs older than 90 days to comply with PDPA data retention requirements.

### Medium Impact / Medium Effort

5. **Shadow scoring (A/B testing)** — After retraining, score incoming transactions with BOTH old and new weights simultaneously. Log the delta. Only commit to the new weights after N transactions show improvement. This prevents deploying a regression.

7. **Async model timeout** — Wrap each model scoring call in `asyncio.wait_for(timeout=0.05)`. If a model takes >50ms, treat it as failed and trigger the HA fallback cascade. This protects against model inference hangs under load.

8. **Quarantine escalation alerts** — When a quarantined label is REJECTED, emit a webhook/alert to the security team. A rejected quarantine means someone tried to groom the model — this is a security incident, not just a data quality issue.

### High Impact / High Effort

10. **Real-time graph analytics** — Replace the simple velocity tracker with a proper temporal graph engine (e.g., Apache Flink + Neo4j). This enables detecting complex multi-hop mule chains (A→B→C→D) rather than just single-recipient velocity spikes.

11. **Federated learning** — For multi-bank deployments, use federated learning to train a shared fraud model without sharing raw transaction data between institutions. Each bank trains locally; only gradients are aggregated.

12. **Adversarial robustness testing** — Implement automated red-team simulation that generates adversarial transactions designed to evade each model individually and the ensemble collectively. Run this as a CI/CD gate before deploying retrained models.
