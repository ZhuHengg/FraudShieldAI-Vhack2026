import React, { useState } from 'react'
import Panel from '../shared/Panel'
import Badge from '../shared/Badge'
import { formatCurrency, formatTime } from '../../utils/formatters'
import clsx from 'clsx'
import DrillDownPanel from './DrillDownPanel'

export default function TransactionFeed({ transactions, totalTotalBufferCount = 500, engineOnline = true }) {
  const [selectedTxn, setSelectedTxn] = useState(null)

  // ─── Fix 4: Offline banner ──────────────────────────────────────────────
  if (!engineOnline) {
    return (
      <Panel className="col-span-1 h-[500px] border border-border">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 rounded-full bg-[#ef4444]" />
          <h3 className="section-label">LIVE TRANSACTION STREAM</h3>
        </div>
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-center max-w-md">
            <div className="text-[32px] mb-3">⚠</div>
            <h4 className="font-mono text-[14px] font-bold text-[#f59e0b] mb-3 tracking-wider">RISK ENGINE OFFLINE</h4>
            <p className="font-mono text-[11px] text-text-muted leading-relaxed mb-4">
              Transactions cannot be scored without the backend.<br />
              Start the backend server at <span className="text-cyan-400">http://localhost:8000</span> to begin processing.
            </p>
            <div className="bg-bg-200 rounded-lg p-3 mb-4 text-left border border-border">
              <code className="font-mono text-[11px] text-text-secondary">
                $ cd backend && uvicorn api.main:app<br />
                &nbsp;&nbsp;--reload --port 8000
              </code>
            </div>
            <div className="flex items-center justify-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-[#f59e0b] animate-pulse" />
              <span className="font-mono text-[10px] text-text-muted">Retrying connection...</span>
            </div>
          </div>
        </div>
      </Panel>
    )
  }

  return (
    <>
      <Panel className="col-span-1 h-[500px] border border-border"
        action={<span className="font-mono text-[10px] text-text-muted">{totalTotalBufferCount} in buffer</span>}
      >
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 rounded-full bg-[#10b981] animate-pulse-slow shadow-glow-approve" />
          <h3 className="section-label">LIVE TRANSACTION STREAM</h3>
        </div>

        {/* Table Header */}
        <div className="grid grid-cols-12 gap-2 pb-2 mb-2 border-b border-border text-[10px] font-mono font-bold uppercase tracking-[0.15em] text-text-muted select-none">
          <div className="col-span-2">Time</div>
          <div className="col-span-2">ID</div>
          <div className="col-span-3">User</div>
          <div className="col-span-3 text-right">Amount</div>
          <div className="col-span-2 text-right">Decision</div>
        </div>

        {/* Scrollable Body */}
        <div className="overflow-y-auto h-[380px] -mx-2 px-2 custom-scrollbar">
          {transactions.map((txn, idx) => (
            <div
              key={`${txn.id}-${new Date(txn.timestamp).getTime()}`}
              onClick={() => setSelectedTxn(txn)}
              className={clsx(
                "grid grid-cols-12 gap-2 items-center py-2.5 px-2 rounded-xl text-[12px] font-mono hover:bg-bg-200 cursor-pointer transition-colors duration-150 group",
                idx === 0 && "animate-slide-in"
              )}
            >
              <div className="col-span-2 text-text-secondary">{formatTime(txn.timestamp)}</div>
              <div className="col-span-2 text-text-primary font-bold">{txn.id.slice(4, 10)}</div>
              <div className="col-span-3 text-text-secondary truncate pr-2">{txn.userId}</div>
              <div className="col-span-3 text-right text-text-primary font-bold pl-2">
                {formatCurrency(txn.amount)}
              </div>
              <div className="col-span-2 text-right">
                <Badge decision={txn.decision} />
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {selectedTxn && (
        <DrillDownPanel txn={selectedTxn} onClose={() => setSelectedTxn(null)} />
      )}
    </>
  )
}
