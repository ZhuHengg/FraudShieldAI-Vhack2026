import React from 'react'
import KpiCard from '../components/shared/KpiCard'
import Panel from '../components/shared/Panel'
import TransactionFeed from '../components/dashboard/TransactionFeed'
import TuningPanel from '../components/dashboard/TuningPanel'
import ConfusionMatrix from '../components/dashboard/ConfusionMatrix'
import { FraudRateSparkline, ScoreHistogram } from '../components/dashboard/Charts'
import ModelDisagreementTable from '../components/dashboard/ModelDisagreementTable'
import TransactionInspector from '../components/TransactionInspector'
import { AlertCircle, X } from 'lucide-react'
import { formatScore } from '../utils/formatters'

export default function Dashboard({ engine }) {
  const { 
    params, transactions, matrix, trends, 
    sparkline, histogram, total, blocked, flagged, approved,
    updateParam, resetParams, triggerAttackBurst,
    weights, thresholds,
    engineOnline, backendStats, config,
    lgbAvgScore, isoAnomalyRate, behHitRate, disagreementRate,
    modelDisagreements
  } = engine

  // Derived Trend Directions
  const blockedTrend = trends.blockedRate > trends.fraudRatePrev ? 'up' : 'down'
  const precisionScore = matrix.precision

  // Fix 5: Use backend stats when available, show '--' when offline
  const statusDot = engineOnline ? 'live' : 'offline'
  
  const displayTotal   = backendStats?.total_transactions ?? total
  const displayBlocked = backendStats?.blocked            ?? blocked
  const displayFlagged = backendStats?.flagged            ?? flagged
  const displayApproved = backendStats?.approved          ?? approved
  const fraudRate      = backendStats?.fraud_rate_estimate ?? (displayTotal > 0 ? ((displayBlocked + displayFlagged) / displayTotal * 100).toFixed(1) : '0.0')

  const blockedPct = displayTotal > 0 ? ((displayBlocked / displayTotal) * 100).toFixed(1) : '0.0'
  const flaggedPct = displayTotal > 0 ? ((displayFlagged / displayTotal) * 100).toFixed(1) : '0.0'
  const approvedPct = displayTotal > 0 ? ((displayApproved / displayTotal) * 100).toFixed(1) : '0.0'

  const displayLatency = backendStats?.avg_latency_ms != null
    ? `${backendStats.avg_latency_ms.toFixed(1)}ms`
    : '--'

  return (
    <div className="max-w-[1400px] mx-auto space-y-6">
      
      {/* Engine Status Banner */}
      {!engineOnline && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-2xl flex items-center gap-3 animate-pulse">
          <AlertCircle size={20} />
          <div className="text-sm font-medium">
            ⚠ Risk Engine not loaded — predictions unavailable
          </div>
        </div>
      )}

      {/* KPI Section — 3 Rows */}
      <div className="space-y-6">
        {/* ROW 1: Transaction Volume & Outcomes */}
        <div className="grid grid-cols-4 gap-6">
          <KpiCard 
            label="Total Volume" 
            value={displayTotal.toLocaleString()} 
            subText="Unit counts processed"
            statusDot={statusDot}
          />
          <KpiCard 
            label="Approved" 
            value={`${displayApproved.toLocaleString()}`} 
            subText={`${approvedPct}% distribution`}
            trend="neutral"
            trendGood={true}
            statusDot={statusDot}
          />
          <KpiCard 
            label="Flagged" 
            value={`${displayFlagged.toLocaleString()}`} 
            subText={`${flaggedPct}% share`}
            trend={trends.flaggedRate > 0.05 ? 'up' : 'down'}
            trendGood={false}
            statusDot={statusDot}
          />
          <KpiCard 
            label="Blocked" 
            value={`${displayBlocked.toLocaleString()}`} 
            subText={`${blockedPct}% share`}
            trend={blockedTrend}
            trendGood={false}
            statusDot={statusDot}
          />
        </div>

        {/* ROW 2: System Health & Score Analysis */}
        <div className="grid grid-cols-4 gap-6">
          <KpiCard 
            label="Fraud Rate" 
            value={`${fraudRate}%`} 
            subText="Backend estimate (F+B)"
            trend={fraudRate > 5 ? 'up' : 'down'}
            trendGood={false}
            statusDot={statusDot}
          />
          <KpiCard 
            label="Avg Latency" 
            value={displayLatency} 
            subText="E2E inference time"
            statusDot={statusDot}
          />
          <KpiCard 
            label="IsoForest Anomaly" 
            value={`${isoAnomalyRate.toFixed(1)}%`}
            subText="Unsupervised rate (>0.5)"
            statusDot={statusDot}
          />
          <KpiCard 
            label="Behavioral Rule Hits" 
            value={`${behHitRate.toFixed(1)}%`}
            subText="Non-normal signatures"
            statusDot={statusDot}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 h-[720px]">
        {/* LEFT COLUMN: Charts & Feed */}
        <div className="col-span-2 flex flex-col gap-6 h-full min-h-0">
          <div className="grid grid-cols-2 gap-6 shrink-0">
            <Panel title="Fraud rate — last 60s">
              <FraudRateSparkline data={sparkline} />
            </Panel>
            <Panel title="Risk score distribution">
              <ScoreHistogram data={histogram} threshold={thresholds?.approve ?? 0.35} bufferWidth={(thresholds?.block ?? 0.70) - (thresholds?.approve ?? 0.35)} />
            </Panel>
          </div>
          
          <TransactionFeed 
            transactions={transactions} 
            totalTotalBufferCount={total} 
            engineOnline={engineOnline} 
            engine={engine} 
            className="flex-1 min-h-0"
          />
        </div>

        {/* RIGHT COLUMN: Tuning & Matrix */}
        <div className="col-span-1 flex flex-col gap-6 h-full min-h-0">
          <TuningPanel 
            params={params} 
            updateParam={updateParam} 
            resetParams={resetParams}
            triggerAttackBurst={triggerAttackBurst}
            weights={weights} 
            thresholds={thresholds} 
            className="shrink-0"
          />
          
          <ConfusionMatrix matrix={matrix} className="flex-1 min-h-0" />
        </div>
      </div>

      {/* BOTTOM ROW: Model Disagreement Table */}
      <ModelDisagreementTable 
        engine={engine} 
        onSelect={(txn) => engine.setSelectedTxn(txn)} 
      />

      {/* Transaction Details Inspector Side-Panel Overlay */}
      {engine.selectedTxn && (
        <div 
          className="fixed inset-0 z-[100] flex justify-end bg-[rgba(10,14,23,0.7)] backdrop-blur-sm animate-in fade-in duration-300"
          onClick={(e) => e.target === e.currentTarget && engine.setSelectedTxn(null)}
        >
          <div className="h-full bg-bg-100 shadow-2xl border-l border-border pointer-events-auto">
            <TransactionInspector selectedTxn={engine.selectedTxn} engine={engine} />
          </div>
        </div>
      )}
    </div>
  )
}
