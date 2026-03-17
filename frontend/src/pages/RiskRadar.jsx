/**
 * Risk Radar — Aligned to behavioural.py model rules
 * Axes and indicators now map directly to the 4 behavioral scoring rules:
 *   - drain_to_unknown      (35% weight)
 *   - high_amount_deviation (25% weight)
 *   - risky_context         (20% weight) → IP/Proxy + Geo + Device
 *   - rapid_session         (20% weight)
 */

import React, { useMemo, useState } from 'react'
import {
  Radar as RadarIcon,
  WifiOff,
  Smartphone,
  Globe,
  MapPin,
  Clock,
  AlertTriangle,
  Shield,
  Activity,
  Fingerprint,
  Zap,
  TrendingUp,
  Server,
} from 'lucide-react'
import clsx from 'clsx'
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  BarChart, Bar,
} from 'recharts'

/* ─── City mapping (SE Asia focus) ──────────────────── */
const COUNTRY_CITIES = {
  MY: ['Kuala Lumpur', 'Penang', 'Johor Bahru'],
  SG: ['Singapore'],
  TH: ['Bangkok', 'Chiang Mai', 'Phuket'],
  ID: ['Jakarta', 'Surabaya', 'Bali'],
  PH: ['Manila', 'Cebu', 'Davao'],
  VN: ['Ho Chi Minh', 'Hanoi', 'Da Nang'],
  MM: ['Yangon', 'Mandalay'],
  KH: ['Phnom Penh', 'Siem Reap'],
}

/* ─── Severity helpers ──────────────────────────────── */
const severityBg = {
  LOW: 'border-emerald-400/30 bg-emerald-400/5',
  MEDIUM: 'border-amber-400/30 bg-amber-400/5',
  HIGH: 'border-red-400/30 bg-red-400/5',
  CRITICAL: 'border-red-500/40 bg-red-500/10',
}
const severityText = {
  LOW: 'text-emerald-400',
  MEDIUM: 'text-amber-400',
  HIGH: 'text-red-400',
  CRITICAL: 'text-red-500',
}
const severityBadge = {
  LOW: 'text-emerald-400 bg-emerald-400/10',
  MEDIUM: 'text-amber-400 bg-amber-400/10',
  HIGH: 'text-red-400 bg-red-400/10',
  CRITICAL: 'text-red-500 bg-red-500/15',
}

/* ─── Rule weight labels ────────────────────────────── */
const RULE_WEIGHTS = {
  drain: '35%',
  amount: '25%',
  context: '20%',
  velocity: '20%',
}

const tooltipStyle = {
  backgroundColor: '#f8fafc', // Light slate
  border: '1px solid #cbd5e1',
  borderRadius: '8px',
  fontSize: '11px',
  fontFamily: 'JetBrains Mono',
  color: '#0f172a', // Dark slate
  boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)',
}
const tooltipItemStyle = { color: '#0f172a' }

export default function RiskRadar({ engine }) {
  const { allTransactions: transactions } = engine
  const [selectedRisk, setSelectedRisk] = useState(null)

  /* ── IP / Context Analysis (replaces VPN-only view) ─ */
  const contextStats = useMemo(() => {
    const proxyTxns = transactions.filter(t => t.isProxyIp)
    const nonProxyTxns = transactions.filter(t => !t.isProxyIp)
    const geoMismatch = transactions.filter(t => t.countryMismatch)
    const newDevice = transactions.filter(t => t.isNewDevice)

    const avgIpRisk = transactions.length > 0
      ? transactions.reduce((s, t) => s + (t.ipRiskScore || 0), 0) / transactions.length
      : 0

    const proxyFraudRate = proxyTxns.length > 0
      ? proxyTxns.filter(t => t.isFraud).length / proxyTxns.length : 0
    const nonProxyFraudRate = nonProxyTxns.length > 0
      ? nonProxyTxns.filter(t => t.isFraud).length / nonProxyTxns.length : 0

    return {
      proxyCount: proxyTxns.length,
      proxyFraudRate,
      nonProxyFraudRate,
      proxyBlockedRate: proxyTxns.length > 0
        ? proxyTxns.filter(t => t.decision === 'BLOCK').length / proxyTxns.length : 0,
      avgIpRisk,
      geoMismatchCount: geoMismatch.length,
      newDeviceCount: newDevice.length,
    }
  }, [transactions])

  /* ── City distribution ────────────────────────────── */
  const cityData = useMemo(() => {
    const counts = {}
    transactions.forEach(t => {
      const cities = COUNTRY_CITIES[t.country] || [t.country]
      const idx = t.userId ? t.userId.charCodeAt(t.userId.length - 1) % cities.length : 0
      const city = cities[idx]
      if (!counts[city]) counts[city] = { total: 0, fraud: 0, blocked: 0 }
      counts[city].total++
      if (t.isFraud) counts[city].fraud++
      if (t.decision === 'BLOCK') counts[city].blocked++
    })
    const totalTxns = transactions.length
    return Object.entries(counts)
      .map(([city, data]) => ({
        city, ...data,
        volumeShare: totalTxns > 0 ? (data.total / totalTxns) * 100 : 0,
        fraudRate: data.total > 0 ? (data.fraud / data.total) * 100 : 0,
      }))
      .sort((a, b) => b.total - a.total)
  }, [transactions])

  /* ── Device type distribution ─────────────────────── */
  const deviceData = useMemo(() => {
    const counts = {}
    transactions.forEach(t => {
      counts[t.deviceType] = (counts[t.deviceType] || 0) + 1
    })
    const colors = ['#00d4ff', '#a855f7', '#10b981', '#f59e0b', '#ef4444']
    return Object.entries(counts).map(([name, value], i) => ({
      name, value, color: colors[i % colors.length],
    }))
  }, [transactions])
  
  /* ── Sector distribution (Domestic vs Foreign) ─────── */
  const sectorData = useMemo(() => {
    let domestic = 0, foreign = 0
    let domesticFraud = 0, foreignFraud = 0
    
    transactions.forEach(t => {
      if (t.countryMismatch) {
        foreign++
        if (t.isFraud) foreignFraud++
      } else {
        domestic++
        if (t.isFraud) domesticFraud++
      }
    })
    
    const totalTxns = transactions.length
    return [
      { 
        name: 'Domestic', 
        value: domestic, 
        volumeShare: totalTxns > 0 ? (domestic / totalTxns) * 100 : 0,
        fraudRate: domestic > 0 ? (domesticFraud / domestic) * 100 : 0, 
        color: '#10b981' 
      },
      { 
        name: 'Foreign', 
        value: foreign, 
        volumeShare: totalTxns > 0 ? (foreign / totalTxns) * 100 : 0,
        fraudRate: foreign > 0 ? (foreignFraud / foreign) * 100 : 0, 
        color: '#ef4444' 
      }
    ]
  }, [transactions])

  /* ── Risk scatter data (amountVsAvgRatio vs risk score) ─────── */
  const scatterData = useMemo(() =>
    transactions.slice(0, 100).map(t => ({
      ratio: parseFloat((t.amountVsAvgRatio || 1).toFixed(2)),
      riskScore: t.ensembleScore * 100,
      status: t.decision,
      amount: t.amount, // kept for tooltip only
    }))
    , [transactions])

  /* ── Behavioral risk radar — aligned to behavioural.py ── */
  const behaviorRadar = useMemo(() => {
    if (transactions.length === 0) return []
    const recent = transactions.slice(0, 50)

    // Rule 1: drain_to_unknown (35%) — senderFullyDrained AND isNewRecipient
    const drainRate = recent.filter(
      t => t.senderFullyDrained && t.isNewRecipient
    ).length / recent.length

    // Rule 2: high_amount_deviation (25%) — (amountVsAvgRatio - 1.5) / 3.5, clipped to [0,1]
    const avgRatio = recent.reduce((s, t) => s + (t.amountVsAvgRatio || 1), 0) / recent.length
    const amountDeviation = Math.min(100, Math.max(0, ((avgRatio - 1.5) / 3.5) * 100))

    // Rule 3a: risky_context — ip_risk_score * 0.5 component
    const avgIpRisk = recent.reduce((s, t) => s + (t.ipRiskScore || 0), 0) / recent.length

    // Rule 3b: risky_context — country_mismatch_suspicious * 0.3 component
    const geoMismatchRate = recent.filter(t => t.countryMismatch).length / recent.length

    // Rule 3c: risky_context — is_new_device * 0.2 component
    const newDeviceRate = recent.filter(t => t.isNewDevice).length / recent.length

    // Rule 4: rapid_session (20%) — tx_count_24h > 5 OR session_duration_seconds < 60
    const rapidSessionRate = recent.filter(
      t => (t.txCount24h > 5) || (t.sessionDurationSeconds < 60)
    ).length / recent.length

    return [
      { metric: 'Account Drain', value: drainRate * 100 },       // Rule 1 — 35%
      { metric: 'Amt Deviation', value: amountDeviation },        // Rule 2 — 25%
      { metric: 'IP / Proxy Risk', value: avgIpRisk * 100 },       // Rule 3a — risky_context
      { metric: 'Session Velocity', value: rapidSessionRate * 100 }, // Rule 4 — 20%
      { metric: 'Geo Anomaly', value: geoMismatchRate * 100 },  // Rule 3b — risky_context
      { metric: 'Device Anomaly', value: newDeviceRate * 100 },    // Rule 3c — risky_context
    ]
  }, [transactions])

  /* ── Rule contribution bar (for legend) ────────────── */
  const ruleContributions = useMemo(() => {
    if (behaviorRadar.length === 0) return []
    const drainVal = behaviorRadar.find(r => r.metric === 'Account Drain')?.value || 0
    const amtVal = behaviorRadar.find(r => r.metric === 'Amt Deviation')?.value || 0
    const ipVal = behaviorRadar.find(r => r.metric === 'IP / Proxy Risk')?.value || 0
    const geoVal = behaviorRadar.find(r => r.metric === 'Geo Anomaly')?.value || 0
    const devVal = behaviorRadar.find(r => r.metric === 'Device Anomaly')?.value || 0
    const velVal = behaviorRadar.find(r => r.metric === 'Session Velocity')?.value || 0

    // Composite risky_context = ipVal*0.5 + geoVal*0.3 + devVal*0.2
    const contextComposite = Math.min(100, ipVal * 0.5 + geoVal * 0.3 + devVal * 0.2)

    return [
      { rule: 'Drain → Unknown', weight: 0.35, score: drainVal, color: '#ef4444' },
      { rule: 'Amt Deviation', weight: 0.25, score: amtVal, color: '#f59e0b' },
      { rule: 'Risky Context', weight: 0.20, score: contextComposite, color: '#a855f7' },
      { rule: 'Rapid Session', weight: 0.20, score: velVal, color: '#00d4ff' },
    ]
  }, [behaviorRadar])

  /* ── Risk indicators — each maps to a behavioural.py rule ── */
  const riskIndicators = useMemo(() => {
    const recent = transactions.slice(0, 100)

    // Rule 1: drain_to_unknown (35%)
    const drainCount = recent.filter(t => t.senderFullyDrained && t.isNewRecipient).length
    const drainRate = recent.length > 0 ? drainCount / recent.length : 0

    // Rule 2: high_amount_deviation (25%)
    const highDeviationCount = recent.filter(t => (t.amountVsAvgRatio || 1) > 3).length

    // Rule 3: risky_context — IP component
    const proxyCount = recent.filter(t => t.isProxyIp).length
    const avgIpRisk = recent.length > 0
      ? recent.reduce((s, t) => s + (t.ipRiskScore || 0), 0) / recent.length : 0

    // Rule 3: risky_context — Geo component
    const geoMismatchCount = recent.filter(t => t.countryMismatch).length

    // Rule 3: risky_context — Device component
    const newDeviceCount = recent.filter(t => t.isNewDevice).length

    // Rule 4: rapid_session (20%)
    const rapidCount = recent.filter(
      t => (t.txCount24h > 5) || (t.sessionDurationSeconds < 60)
    ).length

    return [
      {
        id: 'drain',
        label: 'Account Drain',
        description: 'drain_to_unknown rule (35% weight)',
        severity: drainRate > 0.2 ? 'CRITICAL' : drainRate > 0.1 ? 'HIGH' : drainRate > 0.05 ? 'MEDIUM' : 'LOW',
        value: `${drainCount} detected`,
        weight: '35%',
        icon: AlertTriangle,
      },
      {
        id: 'amount',
        label: 'Amt Deviation',
        description: 'high_amount_deviation rule (25% weight)',
        severity: highDeviationCount > 10 ? 'HIGH' : highDeviationCount > 5 ? 'MEDIUM' : 'LOW',
        value: `${highDeviationCount} high-ratio`,
        weight: '25%',
        icon: TrendingUp,
      },
      {
        id: 'ip',
        label: 'IP / Proxy Risk',
        description: 'risky_context ip_risk_score × 0.5',
        severity: avgIpRisk > 0.6 ? 'CRITICAL' : avgIpRisk > 0.3 ? 'HIGH' : proxyCount > 5 ? 'MEDIUM' : 'LOW',
        value: `${proxyCount} proxy · ${(avgIpRisk * 100).toFixed(0)}% avg`,
        weight: '20%',
        icon: WifiOff,
      },
      {
        id: 'velocity',
        label: 'Session Velocity',
        description: 'rapid_session rule (20% weight)',
        severity: rapidCount > 20 ? 'HIGH' : rapidCount > 10 ? 'MEDIUM' : 'LOW',
        value: `${rapidCount} rapid sessions`,
        weight: '20%',
        icon: Zap,
      },
      {
        id: 'geo',
        label: 'Geo Anomaly',
        description: 'risky_context country_mismatch × 0.3',
        severity: geoMismatchCount > 15 ? 'HIGH' : geoMismatchCount > 5 ? 'MEDIUM' : 'LOW',
        value: `${geoMismatchCount} mismatches`,
        weight: '20%',
        icon: Globe,
      },
      {
        id: 'device',
        label: 'New Device',
        description: 'risky_context is_new_device × 0.2',
        severity: newDeviceCount > 20 ? 'HIGH' : newDeviceCount > 10 ? 'MEDIUM' : 'LOW',
        value: `${newDeviceCount} new devices`,
        weight: '20%',
        icon: Fingerprint,
      },
    ]
  }, [transactions])

  /* ─── Render ──────────────────────────────────────── */
  return (
    <div className="space-y-4">

      {/* ── Risk Indicators Grid ─────────────────────── */}
      <div className="grid grid-cols-6 gap-3">
        {transactions.length === 0 ? (
          <div className="col-span-6 bg-bg-100/50 rounded-2xl border border-white/5 py-12 text-center">
            <p className="text-amber-400 text-xs font-mono uppercase tracking-wider">
              ⚠ No transactions yet
            </p>
            <p className="text-muted-foreground text-[11px] mt-1">
              Start the backend and simulator to populate risk radar
            </p>
          </div>
        ) : (
          riskIndicators.map(ind => {
            const Icon = ind.icon
            return (
              <div
                key={ind.id}
                onClick={() => setSelectedRisk(selectedRisk === ind.id ? null : ind.id)}
                className={clsx(
                  'rounded-2xl border p-3 cursor-pointer transition-all bg-bg-100',
                  severityBg[ind.severity],
                  selectedRisk === ind.id && 'ring-1 ring-cyan-400/30'
                )}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Icon className={clsx('w-4 h-4', severityText[ind.severity])} />
                    <span className={clsx(
                      'text-[9px] font-semibold px-1.5 py-0.5 rounded uppercase tracking-wider',
                      severityBadge[ind.severity]
                    )}>
                      {ind.severity}
                    </span>
                  </div>
                  {/* Rule weight pill */}
                  <span className="text-[8px] font-mono text-muted-foreground/60 bg-white/5 px-1 py-0.5 rounded">
                    w={ind.weight}
                  </span>
                </div>
                <p className="text-xs font-medium text-foreground">{ind.label}</p>
                <p className="text-[9px] font-mono text-muted-foreground/70 mt-0.5 leading-tight">{ind.description}</p>
                <p className="text-[10px] font-mono text-muted-foreground mt-1">{ind.value}</p>
              </div>
            )
          })
        )}
      </div>

      {/* ── Middle Row: Radar + Scatter ──────────────── */}
      <div className="grid grid-cols-12 gap-4">

        {/* Left col: Radar + Context Analysis */}
        <div className="col-span-5 space-y-4">

          {/* Behavioral Risk Radar */}
          <div className="bg-bg-100 rounded-2xl border border-white/5 overflow-hidden">
            <div className="pb-1 pt-3 px-4 flex items-center justify-between">
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <RadarIcon className="w-3.5 h-3.5 text-cyan-400" />
                Behavioral Risk Radar
              </h2>
              <span className="text-[9px] font-mono text-muted-foreground/50">aligned to behavioural.py</span>
            </div>

            <div className="px-2 pb-1">
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={behaviorRadar}>
                  <PolarGrid stroke="oklch(0.25 0.03 260)" />
                  <PolarAngleAxis
                    dataKey="metric"
                    tick={{ fontSize: 10, fill: 'oklch(0.60 0.02 260)' }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 100]}
                    tick={{ fontSize: 8, fill: 'oklch(0.40 0.02 260)' }}
                  />
                  <Radar
                    name="Risk"
                    dataKey="value"
                    stroke="#ef4444"
                    fill="#ef4444"
                    fillOpacity={0.15}
                    strokeWidth={2}
                  />
                  <Tooltip 
                    contentStyle={tooltipStyle} 
                    itemStyle={tooltipItemStyle}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Rule weight contributions */}
            <div className="px-4 pb-4 space-y-2 border-t border-white/5 pt-3">
              <p className="text-[9px] uppercase tracking-wider text-muted-foreground/50 mb-2">
                Ensemble Rule Weights
              </p>
              {ruleContributions.map(r => (
                <div key={r.rule} className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted-foreground w-28 shrink-0">{r.rule}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/5 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${r.score}%`,
                        backgroundColor: r.color,
                        opacity: 0.8,
                      }}
                    />
                  </div>
                  <span className="text-[9px] font-mono w-8 text-right" style={{ color: r.color }}>
                    {r.score.toFixed(0)}
                  </span>
                  <span className="text-[9px] font-mono text-muted-foreground/40 w-8">
                    ×{r.weight}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* IP & Context Risk Analysis (replaces VPN-only panel) */}
          <div className="bg-bg-100 rounded-2xl border border-white/5 overflow-hidden">
            <div className="pb-2 pt-3 px-4">
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                <Server className="w-3.5 h-3.5 text-purple-400" />
                IP & Context Risk
                <span className="text-[8px] font-mono text-muted-foreground/40 ml-1">risky_context rule</span>
              </h2>
            </div>
            <div className="px-4 pb-4 space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white/[0.02] rounded-lg p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Proxy Transactions</p>
                  <p className="text-xl font-mono font-bold text-red-400">{contextStats.proxyCount}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {transactions.length > 0
                      ? ((contextStats.proxyCount / transactions.length) * 100).toFixed(1)
                      : 0}% of total
                  </p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Proxy Fraud Rate</p>
                  <p className="text-xl font-mono font-bold text-amber-400">
                    {(contextStats.proxyFraudRate * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    vs {(contextStats.nonProxyFraudRate * 100).toFixed(1)}% non-proxy
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-white/[0.02] rounded-lg p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Geo Mismatches</p>
                  <p className="text-lg font-mono font-bold text-purple-400">{contextStats.geoMismatchCount}</p>
                  <p className="text-[10px] text-muted-foreground">country_mismatch flag</p>
                </div>
                <div className="bg-white/[0.02] rounded-lg p-3">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">Avg IP Risk Score</p>
                  <p className="text-lg font-mono font-bold text-cyan-400">
                    {(contextStats.avgIpRisk * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {contextStats.newDeviceCount} new devices
                  </p>
                </div>
              </div>

              {/* Context sub-rule breakdown */}
              <div className="bg-white/[0.02] rounded-lg p-3 space-y-2">
                <p className="text-[9px] uppercase tracking-wider text-muted-foreground/50">
                  Context Sub-weights
                </p>
                {[
                  { label: 'ip_risk_score', sub: '×0.5', val: contextStats.avgIpRisk * 100, color: '#00d4ff' },
                  { label: 'country_mismatch_suspicious', sub: '×0.3', val: transactions.length > 0 ? (contextStats.geoMismatchCount / transactions.length) * 100 : 0, color: '#a855f7' },
                  { label: 'is_new_device', sub: '×0.2', val: transactions.length > 0 ? (contextStats.newDeviceCount / transactions.length) * 100 : 0, color: '#f59e0b' },
                ].map(item => (
                  <div key={item.label} className="flex items-center gap-2">
                    <span className="text-[9px] font-mono text-muted-foreground w-36 shrink-0">{item.label}</span>
                    <span className="text-[9px] font-mono text-muted-foreground/40 w-6">{item.sub}</span>
                    <div className="flex-1 h-1 rounded-full bg-white/5 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${Math.min(item.val, 100)}%`, backgroundColor: item.color, opacity: 0.7 }}
                      />
                    </div>
                    <span className="text-[9px] font-mono w-8 text-right" style={{ color: item.color }}>
                      {item.val.toFixed(0)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Right col: Scatter + City + Device */}
        <div className="col-span-7 space-y-4">

          {/* Amount Ratio vs Risk Score scatter */}
          <div className="bg-bg-100 rounded-2xl border border-white/5 overflow-hidden">
            <div className="pb-1 pt-3 px-4 flex items-center justify-between">
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Amount Deviation Ratio vs Risk Score
              </h2>
              <span className="text-[9px] font-mono text-muted-foreground/50">high_amount_deviation rule · 25% weight</span>
            </div>

            {/* Threshold annotations */}
            <div className="px-4 pb-1 flex items-center gap-4">
              <span className="flex items-center gap-1.5 text-[9px] font-mono text-muted-foreground/60">
                <div className="w-3 h-px bg-amber-400/60" style={{ borderTop: '1px dashed' }} />
                ratio=1.5× — scoring starts
              </span>
              <span className="flex items-center gap-1.5 text-[9px] font-mono text-muted-foreground/60">
                <div className="w-3 h-px bg-red-400/60" style={{ borderTop: '1px dashed' }} />
                ratio=5.0× — max deviation score
              </span>
            </div>

            <div className="px-2 pb-3">
              <ResponsiveContainer width="100%" height={250}>
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.25 0.03 260)" />
                  {/* Reference line at ratio=1.5 (scoring threshold from behavioural.py) */}
                  <XAxis
                    dataKey="ratio"
                    name="Amount Ratio"
                    type="number"
                    domain={[0, 'auto']}
                    tick={{ fontSize: 10, fill: 'oklch(0.60 0.02 260)', fontFamily: 'JetBrains Mono' }}
                    label={{ value: 'amount_vs_avg_ratio (×)', position: 'insideBottom', offset: -10, fontSize: 10, fill: 'oklch(0.50 0.02 260)' }}
                  >
                  </XAxis>
                  <YAxis
                    dataKey="riskScore"
                    name="Risk Score"
                    domain={[0, 100]}
                    tick={{ fontSize: 10, fill: 'oklch(0.60 0.02 260)', fontFamily: 'JetBrains Mono' }}
                    label={{ value: 'Risk %', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'oklch(0.50 0.02 260)' }}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    itemStyle={tooltipItemStyle}
                    formatter={(value, name) => {
                      if (name === 'Amount Ratio') return [`${Number(value).toFixed(2)}×`, 'Deviation Ratio']
                      if (name === 'Risk Score') return [`${Number(value).toFixed(1)}%`, 'Risk Score']
                      return [value, name]
                    }}
                  />
                  <Scatter data={scatterData.filter(d => d.status === 'APPROVE')} fill="#10b981" fillOpacity={0.6} />
                  <Scatter data={scatterData.filter(d => d.status === 'FLAG')} fill="#f59e0b" fillOpacity={0.7} />
                  <Scatter data={scatterData.filter(d => d.status === 'BLOCK')} fill="#ef4444" fillOpacity={0.8} />
                </ScatterChart>
              </ResponsiveContainer>
              <div className="flex items-center justify-between mt-1 px-2">
                <div className="flex items-center gap-4">
                  {[['#10b981', 'Approved'], ['#f59e0b', 'Flagged'], ['#ef4444', 'Blocked']].map(([color, label]) => (
                    <span key={label} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                      {label}
                    </span>
                  ))}
                </div>
                <span className="text-[9px] font-mono text-muted-foreground/40">
                  formula: clip((ratio − 1.5) / 3.5, 0, 1)
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-4">
              {/* City Distribution */}
              <div className="bg-bg-100 rounded-2xl border border-white/5 overflow-hidden">
                <div className="pb-2 pt-3 px-4">
                  <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                    <MapPin className="w-3.5 h-3.5 text-cyan-400" />
                    By City
                  </h2>
                </div>
                <div className="px-4 pb-4">
                  <div className="space-y-2">
                    {cityData.slice(0, 6).map(c => (
                      <div key={c.city} className="flex items-center gap-2">
                        <span className="text-xs text-foreground w-28 shrink-0 truncate">{c.city}</span>
                        <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ 
                              width: `${(c.total / Math.max(...cityData.map(d => d.total))) * 100}%`,
                              backgroundColor: '#22d3ee',
                              opacity: 0.6
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{c.total}</span>
                        <span className="text-[10px] font-mono w-12 text-right text-cyan-400/80">
                          {c.volumeShare.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* By Sector Distribution */}
              <div className="bg-bg-100 rounded-2xl border border-white/5 overflow-hidden">
                <div className="pb-2 pt-3 px-4">
                  <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                    <Globe className="w-3.5 h-3.5 text-orange-400" />
                    By Sector
                  </h2>
                </div>
                <div className="px-4 pb-4">
                  <div className="space-y-3">
                    {sectorData.map(s => (
                      <div key={s.name} className="flex items-center gap-2">
                        <span className="text-xs text-foreground w-20 shrink-0">{s.name}</span>
                        <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ 
                              width: `${(s.value / Math.max(...sectorData.map(d => d.value))) * 100}%`,
                              backgroundColor: s.color,
                              opacity: 0.7
                            }}
                          />
                        </div>
                        <span className="text-[10px] font-mono text-muted-foreground w-8 text-right">{s.value}</span>
                        <span className="text-[10px] font-mono w-12 text-right opacity-80" style={{ color: s.color }}>
                          {s.volumeShare.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Device Distribution */}
            <div className="bg-bg-100 rounded-2xl border border-white/5 overflow-hidden">
              <div className="pb-2 pt-3 px-4">
                <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                  <Smartphone className="w-3.5 h-3.5 text-cyan-400" />
                  By Device
                </h2>
              </div>
              <div className="px-2 pb-3">
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie
                      data={deviceData}
                      cx="50%" cy="50%"
                      innerRadius={45} outerRadius={70}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {deviceData.map((entry, index) => (
                        <Cell key={index} fill={entry.color} fillOpacity={0.8} />
                      ))}
                    </Pie>
                    <Tooltip 
                contentStyle={tooltipStyle} 
                itemStyle={tooltipItemStyle}
              />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex flex-wrap justify-center gap-3 mt-1">
                  {deviceData.map(d => (
                    <span key={d.name} className="flex items-center gap-1 text-[10px] text-muted-foreground">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                      {d.name}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}