import React from 'react'
import Panel from '../components/shared/Panel'
import {
  Play, Pause, Flame, RefreshCw, Trash2, CheckCircle2, AlertTriangle, XCircle, Activity
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, Cell, ReferenceLine
} from 'recharts'
import clsx from 'clsx'
import { formatCurrency, formatTime } from '../utils/formatters'
import TransactionInspector from '../components/TransactionInspector'

export default function LiveSimulator({ engine }) {
  const {
    params, updateParam, transactions,
    isRunning, setIsRunning, triggerAttackBurst, resetParams, clearData,
    total, approved, flagged, blocked, matrix, avgLatency,
    sparkline, histogram
  } = engine

  const ActionBtn = ({ icon: Icon, label, color, onClick }) => (
    <button
      onClick={onClick}
      className={clsx(
        "flex items-center gap-2 px-3 py-1.5 rounded-lg font-mono text-[10px] font-bold tracking-widest uppercase transition-all border whitespace-nowrap",
        color === 'yellow' ? "bg-[rgba(245,158,11,0.1)] text-[#f59e0b] border-[rgba(245,158,11,0.3)] hover:bg-[rgba(245,158,11,0.15)]" :
          color === 'red' ? "bg-[rgba(239,68,68,0.1)] text-[#ef4444] border-[rgba(239,68,68,0.3)] hover:bg-[rgba(239,68,68,0.15)]" :
            "bg-transparent text-text-secondary border-border hover:bg-bg-200"
      )}
    >
      <Icon size={12} />
      {label}
    </button>
  )

  const StatBlock = ({ label, value, colorClass }) => (
    <div className="bg-bg-100 rounded-xl px-3 py-3 border border-border flex flex-col items-center justify-center gap-1">
      <span className="font-mono text-[9px] text-text-muted font-bold tracking-widest uppercase">{label}</span>
      <span className={clsx("font-mono text-[20px] font-bold leading-none", colorClass)}>{value}</span>
    </div>
  )

  const getStatusColor = (d) => d === 'APPROVE' ? 'text-[#10b981]' : d === 'FLAG' ? 'text-[#f59e0b]' : 'text-[#ef4444]'
  const getStatusIcon = (d) => d === 'APPROVE'
    ? <CheckCircle2 size={13} className="text-[#10b981] shrink-0" />
    : d === 'FLAG'
      ? <AlertTriangle size={13} className="text-[#f59e0b] shrink-0" />
      : <XCircle size={13} className="text-[#ef4444] shrink-0" />

  // Ensure sparkline has at least some non-zero variance so the chart line is visible
  const chartData = sparkline.map((d, i) => ({ ...d, rate: d.rate || 0 }))
  const hasData = chartData.some(d => d.rate > 0)

  return (
    <div className="flex flex-col gap-3 h-[calc(100vh-112px)] overflow-hidden">

      {/* ── ACTION BAR ── */}
      <div className="flex items-center justify-between shrink-0">
        <div className="flex gap-2">
          <ActionBtn icon={isRunning ? Pause : Play} label={isRunning ? "PAUSE SIMULATION" : "RESUME"} color={isRunning ? "yellow" : "green"} onClick={() => setIsRunning(!isRunning)} />
          <ActionBtn icon={Flame} label="ATTACK BURST (10 TXN)" color="red" onClick={triggerAttackBurst} />
          <ActionBtn icon={RefreshCw} label="RESET PARAMS" onClick={resetParams} />
          <ActionBtn icon={Trash2} label="CLEAR" onClick={clearData} />
        </div>
        <div className="flex items-center gap-5">
          <div className="flex items-center gap-2">
            <span className="font-mono text-[9px] text-muted-foreground font-bold tracking-widest uppercase">SPEED:</span>
            <input type="range" min="0.5" max="5" step="0.5" value={params.simulationSpeed}
              onChange={e => updateParam('simulationSpeed', parseFloat(e.target.value))} className="w-20" />
            <span className="font-mono text-[11px] text-[#22d3ee] font-bold w-14">{params.simulationSpeed} txn/s</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={clsx("w-2 h-2 rounded-full", isRunning ? "bg-[#34d399] animate-pulse" : "bg-muted-foreground")} />
            <span className="font-mono text-[10px] text-muted-foreground font-bold tracking-widest">{isRunning ? 'LIVE' : 'PAUSED'}</span>
          </div>
        </div>
      </div>

      {/* ── 8 STAT BLOCKS ── */}
      <div className="grid grid-cols-6 gap-2 shrink-0">
        <StatBlock label="TOTAL" value={total.toLocaleString()} colorClass="text-[#22d3ee]" />
        <StatBlock label="APPROVED" value={approved.toLocaleString()} colorClass="text-[#34d399]" />
        <StatBlock label="FLAGGED" value={flagged.toLocaleString()} colorClass="text-[#fbbf24]" />
        <StatBlock label="BLOCKED" value={blocked.toLocaleString()} colorClass="text-[#f87171]" />
        <StatBlock label="LATENCY" value={`${avgLatency}ms`} colorClass="text-[#c084fc]" />
        <StatBlock label="ACCURACY" value={`${(matrix.accuracy * 100).toFixed(1)}%`} colorClass="text-foreground" />
      </div>

      {/* ── MAIN CONTENT (flex row, fills remaining height) ── */}
      <div className="flex gap-3 flex-1 min-h-0">

        {/* LEFT: Trend + Feed */}
        <div className="flex flex-col gap-3 flex-[2] min-w-0">

          {/* Fraud Rate Trend */}
          <Panel className="shrink-0 p-4" style={{ height: '160px' }}>
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-3 h-3 text-[#f87171]" />
              <span className="font-mono text-[9px] text-muted-foreground font-bold tracking-widest uppercase">FRAUD RATE TREND (60S)</span>
              {!hasData && <span className="font-mono text-[9px] text-muted-foreground ml-2">— waiting for fraud events</span>}
            </div>
            <div style={{ height: '100px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorFraud" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f87171" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Tooltip
                    contentStyle={{ backgroundColor: 'var(--bg-100)', border: '1px solid var(--border-md)', borderRadius: '8px' }}
                    itemStyle={{ color: '#f87171', fontFamily: 'JetBrains Mono', fontSize: '11px' }}
                    labelStyle={{ color: '#cbd5e1', fontSize: '10px' }}
                  />
                  <XAxis dataKey="time" hide />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 9, fill: '#cbd5e1' }}
                    axisLine={false} tickLine={false} width={40}
                    tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                  />
                  {/* Ghost baseline so the chart doesn't look dead */}
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
                  <Area type="monotone" dataKey="rate" stroke="#f87171" strokeWidth={1.5} fill="url(#colorFraud)" dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          {/* Live Transaction Feed */}
          <div className="bg-bg-100 rounded-2xl relative overflow-hidden flex flex-col flex-1 min-h-0 transition-shadow duration-300">
            <div className="px-4 py-2.5 flex justify-between items-center border-b border-white/5 shrink-0">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-[#34d399] animate-pulse" />
                <span className="font-mono text-[9px] text-muted-foreground font-medium tracking-widest uppercase">LIVE TRANSACTION FEED</span>
              </div>
              <span className="font-mono text-[9px] text-muted-foreground">{total} in buffer</span>
            </div>

            {/* Column Headers + List Container */}
            <div className="flex-1 flex flex-col min-h-0">
              <div className="flex items-center gap-3 px-4 py-1.5 border-b border-white/5 shrink-0">
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-14 shrink-0">TIME</span>
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-4 shrink-0"></span>
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-24 shrink-0">TX ID</span>
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-24 shrink-0">USER</span>
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-20 shrink-0 text-right">AMOUNT</span>
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-28 shrink-0 pl-3">MERCHANT</span>
                <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-10 shrink-0">CTY</span>
                <div className="flex-1 flex justify-end gap-3 pr-2">
                  <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-14 text-right">RISK</span>
                  <span className="font-mono text-[8px] text-muted-foreground tracking-widest uppercase w-14 text-right">STATUS</span>
                  <span className="w-14 shrink-0"></span>{/* Space for Inspect btn */}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto custom-scrollbar">
                {transactions.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-muted-foreground font-mono text-[11px]">Waiting for stream…</div>
                ) : (
                  <div className="flex flex-col">
                    {transactions.map((t, i) => (
                      <div key={t.id}
                        onClick={() => engine.setSelectedTxn(t)}
                        className={clsx(
                          "flex items-center gap-3 px-4 py-1.5 transition-all text-[11px] font-mono animate-pop-out cursor-pointer",
                          t.decision === 'BLOCK' ? 'bg-[#f87171]/[0.03]' :
                            t.decision === 'FLAG' ? 'bg-[#fbbf24]/[0.02]' : 'hover:bg-bg-50/40',
                          engine.selectedTxn?.id === t.id && "ring-1 ring-inset ring-white/10 bg-white/5"
                        )}>
                        <span className="text-muted-foreground w-14 shrink-0">{formatTime(t.timestamp)}</span>
                        {getStatusIcon(t.decision)}
                        <span className={clsx("w-24 shrink-0", t.decision === 'APPROVE' ? 'text-[#22d3ee]/60' : getStatusColor(t.decision))}>{t.id.slice(4, 14).toUpperCase()}</span>
                        <span className="text-foreground/80 w-24 shrink-0 truncate">{t.userId.replace('user_', '')}</span>
                        <span className="text-foreground w-20 shrink-0 text-right">{formatCurrency(t.amount).replace('MYR ', 'RM ')}</span>
                        <span className="text-muted-foreground w-28 shrink-0 truncate pl-3">{t.merchantCategory}</span>
                        <span className="text-muted-foreground w-10 shrink-0">{t.country}{t.vpnDetected && <span className="ml-1 text-[#f87171] bg-[#f87171]/10 px-1 rounded text-[8px]">VPN</span>}</span>
                        <div className="flex-1 flex justify-end items-center gap-3">
                          <div className="w-12 h-1 bg-white/5 rounded-full overflow-hidden">
                            <div className={clsx("h-full rounded-full", t.decision === 'BLOCK' ? 'bg-[#f87171]' : t.decision === 'FLAG' ? 'bg-[#fbbf24]' : 'bg-[#34d399]')}
                              style={{ width: `${t.ensembleScore * 100}%` }} />
                          </div>
                          <span className={clsx("w-8 text-right", t.decision === 'BLOCK' ? 'text-[#f87171]' : t.decision === 'FLAG' ? 'text-[#fbbf24]' : 'text-[#34d399]')}>{(t.ensembleScore * 100).toFixed(0)}%</span>
                          <span className={clsx("w-14 text-right", t.decision === 'APPROVE' ? 'text-[#34d399]' : getStatusColor(t.decision))}>{t.decision}</span>
                          <button 
                            className="ml-2 px-2 py-1 rounded bg-white/5 hover:bg-white/10 text-[9px] font-mono text-text-muted transition-colors border border-white/10 shrink-0"
                            onClick={(e) => {
                              // Stop propagation so we don't double fire if the row is also clickable
                              e.stopPropagation();
                              engine.setSelectedTxn(t);
                            }}
                          >
                            INSPECT
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: Transaction Detail Inspector (handles its own width + animation) */}
        <TransactionInspector selectedTxn={engine.selectedTxn} engine={engine} />
      </div>
    </div>
  )
}
