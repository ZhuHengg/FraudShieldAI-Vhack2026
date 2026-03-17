import React, { useState, useEffect } from 'react'
import Panel from './shared/Panel'
import { PieChart, Pie, Cell, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis } from 'recharts'
import { Lock } from 'lucide-react'
import clsx from 'clsx'

export default function TransactionInspector({ selectedTxn, engine }) {
  const [apiData, setApiData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!selectedTxn) {
      setApiData(null)
      setError(null)
      return
    }

    const fetchResult = async () => {
      setLoading(true)
      setError(null)
      try {
        // Build payload matching backend schemas.py TransactionRequest
        const payload = {
          transaction_id: selectedTxn.id,
          amount: selectedTxn.amount,
          sender_id: selectedTxn.userId,
          receiver_id: "UNKNOWN_REC", // Dummy for simulator
          transaction_type: selectedTxn.merchantCategory.toLowerCase().includes('cash') ? 'cash_out' : 'transfer',
          timestamp: selectedTxn.timestamp.toISOString(),
          // Add behavioral triggers if present
          is_new_device: selectedTxn.newDevice ? 1 : 0,
          is_proxy_ip: selectedTxn.vpnDetected ? 1 : 0,
          tx_count_24h: selectedTxn.velocityFlag ? 10 : 1
        }

        const res = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        })

        if (!res.ok) throw new Error('API requested failed')
        const data = await res.json()
        setApiData(data)
      } catch (err) {
        console.error("Inspector API Error:", err)
        setError("Failed to fetch risk analysis.")
      } finally {
        setLoading(false)
      }
    }

    fetchResult()
  }, [selectedTxn])

  if (!selectedTxn) {
    return (
      <div className="flex flex-col gap-3 w-full h-full shrink-0 pb-2">
        <Panel className="p-4 flex-1 flex items-center justify-center border-border/50">
          <span className="font-mono text-[11px] text-muted-foreground tracking-widest uppercase text-center">
            Select a transaction<br />to inspect
          </span>
        </Panel>
      </div>
    )
  }

  // Use API data if available, fallback to simulator approximations for instant feedback
  const score = apiData ? (apiData.risk_score / 100) : selectedTxn.ensembleScore
  const decision = apiData ? apiData.risk_level.toUpperCase() : selectedTxn.decision

  const lgbScore = apiData ? apiData.supervised_score : selectedTxn.lgbScore
  const isoScore = apiData ? apiData.unsupervised_score : selectedTxn.xgbScore // Approximated
  const behScore = apiData ? Math.max(0, Math.min(1, (score - (lgbScore * 0.55 + isoScore * 0.25)) / 0.20)) : 0.1

  const reasons = apiData?.reasons || selectedTxn.riskFactors
  const finalReasons = reasons.length ? reasons : ["Normal behavior pattern"]

  // Theme Helpers
  const getColor = (lvl) => lvl === 'APPROVE' || lvl === 'LOW' ? '#34d399' : lvl === 'FLAG' || lvl === 'MEDIUM' ? '#fbbf24' : '#f87171'
  const primaryColor = getColor(decision)

  // Semicircle Stroke Math
  const radius = 60
  const circumference = 2 * Math.PI * radius
  const semiCircumference = circumference / 2
  const strokeDashoffset = semiCircumference - (score * semiCircumference)

  // Decisions Donut Data
  const donutData = [
    { name: 'APPROVE', value: engine.approved || 1, color: '#34d399' },
    { name: 'FLAG', value: engine.flagged || 0, color: '#fbbf24' },
    { name: 'BLOCK', value: engine.blocked || 0, color: '#f87171' }
  ]

  // Risk By Hour Data (Derived from engine.allTransactions)
  const hourMap = new Array(24).fill(0)
  engine.allTransactions.forEach(t => {
    if (t.decision !== 'APPROVE') {
      const hr = new Date(t.timestamp).getHours()
      hourMap[hr]++
    }
  })
  const hourlyData = hourMap.map((count, hr) => ({
    hour: `${hr}h`,
    count: count
  }))

  return (
    <div className="flex flex-col gap-3 w-full h-full shrink-0 overflow-y-auto custom-scrollbar pb-2 animate-in fade-in slide-in-from-right-4 duration-300">

      {/* 1. Scoring Breakdown */}
      <Panel className="p-4 shrink-0 border-border/50 relative overflow-hidden">
        {loading && <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-[#22d3ee] animate-pulse" />}
        <span className="font-mono text-[9px] text-muted-foreground font-medium tracking-widest uppercase mb-4 block">Scoring Breakdown</span>

        <div className="flex flex-col items-center">
          <div className="relative w-[140px] h-[75px] overflow-hidden flex justify-center">
            {/* Background Arc */}
            <svg viewBox="0 0 140 140" className="absolute top-0 w-[140px] h-[140px] rotate-180">
              <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" strokeDasharray={`${semiCircumference} ${circumference}`} />
              {/* Value Arc */}
              <circle cx="70" cy="70" r={radius} fill="none" stroke={primaryColor} strokeWidth="12" strokeDasharray={`${semiCircumference} ${circumference}`} strokeDashoffset={strokeDashoffset} className="transition-all duration-700 ease-out" />
            </svg>
            <div className="absolute bottom-1 w-full text-center">
              <div className="font-mono text-3xl font-bold tracking-tight" style={{ color: primaryColor }}>
                {score.toFixed(3)}
              </div>
            </div>
          </div>
          <div className="mt-1 font-mono text-[10px] tracking-[0.2em] font-bold" style={{ color: primaryColor }}>
            {decision}
          </div>
        </div>
      </Panel>

      {/* 2. Ensemble Layers */}
      <Panel className="p-4 shrink-0 border-border/50">
        <span className="font-mono text-[9px] text-muted-foreground font-medium tracking-widest uppercase mb-3 block">Ensemble Layers</span>
        <div className="space-y-3">
          {[
            { label: 'LightGBM (55%)', val: lgbScore, color: '#22d3ee' },
            { label: 'IsoForest (25%)', val: isoScore, color: '#34d399' },
            { label: 'Behavioral (20%)', val: behScore, color: '#fbbf24' }
          ].map(layer => (
            <div key={layer.label}>
              <div className="flex justify-between mb-1.5 mix-blend-screen">
                <span className="font-mono text-[9px] text-muted-foreground tracking-wider">{layer.label}</span>
                <span className="font-mono text-[10px]" style={{ color: layer.color }}>{layer.val.toFixed(2)}</span>
              </div>
              <div className="h-1.5 w-full bg-white/5 rounded-full relative overflow-hidden">
                <div className="h-full rounded-full absolute left-0 transition-all duration-500" style={{ width: `${layer.val * 100}%`, backgroundColor: layer.color }} />
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* 3. Risk Drivers */}
      <Panel className="p-4 shrink-0 border-border/50">
        <span className="font-mono text-[9px] text-muted-foreground font-medium tracking-widest uppercase mb-3 block">Risk Drivers</span>
        <div className="space-y-2">
          {finalReasons.map((r, i) => {
            const isNormal = r.toLowerCase().includes('normal')
            return (
              <div key={i} className={clsx(
                "px-3 py-2 rounded-lg border-l-2 bg-white/[0.02] border-white/5",
                isNormal ? "border-l-muted-foreground" : "border-l-[#f87171]"
              )}>
                <span className="font-sans text-[11px] text-foreground/90">{r}</span>
              </div>
            )
          })}

          <div className="px-3 py-2 rounded-lg border-l-2 border-l-[#34d399] bg-[#34d399]/[0.05] border-[#34d399]/10 flex items-start gap-2 mt-3">
            <Lock size={12} className="text-[#fbbf24] shrink-0 mt-[2px]" />
            <span className="font-sans text-[10px] text-muted-foreground leading-relaxed">
              PII hashed with SHA-256 before inference — no raw identities touch the model.
            </span>
          </div>
        </div>
      </Panel>

      {/* 4. Mini Charts Row */}
      <div className="grid grid-cols-2 gap-3 shrink-0">
        <Panel className="p-3 border-border/50">
          <span className="font-mono text-[9px] text-muted-foreground font-medium tracking-widest uppercase mb-1 block">Decisions</span>
          <div className="h-[90px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={donutData} innerRadius={28} outerRadius={38} paddingAngle={2} dataKey="value" stroke="rgba(0,0,0,0)" startAngle={180} endAngle={0}>
                  {donutData.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.color} />)}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel className="p-3 border-border/50">
          <span className="font-mono text-[9px] text-muted-foreground font-medium tracking-widest uppercase mb-1 block">Risk by Hour</span>
          <div className="h-[90px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData} margin={{ top: 10, right: 0, left: 0, bottom: -5 }}>
                <XAxis dataKey="hour" tickLine={false} axisLine={false} tick={{ fontSize: 7, fill: '#64748b' }} angle={-45} textAnchor="end" height={20} interval={1} />
                <YAxis hide />
                <Bar dataKey="count" fill="#f87171" radius={[1, 1, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

    </div>
  )
}
