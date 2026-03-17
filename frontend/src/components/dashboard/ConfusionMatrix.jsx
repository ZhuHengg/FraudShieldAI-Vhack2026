import React from 'react'
import Panel from '../shared/Panel'
import { formatScore } from '../../utils/formatters'

export default function ConfusionMatrix({ matrix }) {
  const { tp, fp, fn, tn, precision, recall, f1 } = matrix

  return (
    <Panel className="border border-border">
      <div className="flex justify-between items-center mb-6">
        <h3 className="section-label">Confusion Matrix</h3>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-6 font-mono text-[13px] text-center font-bold">
        {/* TP (Green) */}
        <div className="bg-[rgba(16,185,129,0.1)] text-[#10b981] p-4 rounded-xl flex flex-col justify-center border border-[rgba(16,185,129,0.3)] shadow-sm">
          <div className="text-[20px] mb-1">{tp}</div>
          <div className="text-[10px] opacity-80">TRUE POSITIVE</div>
        </div>
        {/* FP (Amber) */}
        <div className="bg-[rgba(245,158,11,0.1)] text-[#f59e0b] p-4 rounded-xl flex flex-col justify-center border border-[rgba(245,158,11,0.3)] shadow-sm">
          <div className="text-[20px] mb-1">{fp}</div>
          <div className="text-[10px] opacity-80">FALSE POSITIVE</div>
        </div>
        {/* FN (Red) */}
        <div className="bg-[rgba(239,68,68,0.1)] text-[#ef4444] p-4 rounded-xl flex flex-col justify-center border border-[rgba(239,68,68,0.3)] shadow-sm">
          <div className="text-[20px] mb-1">{fn}</div>
          <div className="text-[10px] opacity-80">FALSE NEGATIVE</div>
        </div>
        {/* TN (Neutral) */}
        <div className="bg-bg-200 text-text-secondary p-4 rounded-xl flex flex-col justify-center border border-border shadow-sm">
          <div className="text-[20px] mb-1">{tn}</div>
          <div className="text-[10px] opacity-80">TRUE NEGATIVE</div>
        </div>
      </div>

      {/* Stats row below matrix */}
      <div className="grid grid-cols-3 gap-2 text-center border-t border-border pt-4">
        <div>
          <div className="section-label mb-1">Precision</div>
          <div className="font-mono font-bold text-text-primary text-[14px]">{formatScore(precision)}</div>
        </div>
        <div className="border-l border-border pl-2">
          <div className="section-label mb-1">Recall</div>
          <div className="font-mono font-bold text-text-primary text-[14px]">{formatScore(recall)}</div>
        </div>
        <div className="border-l border-border pl-2">
          <div className="section-label mb-1">F1 Score</div>
          <div className="font-mono font-bold text-text-primary text-[14px]">{formatScore(f1)}</div>
        </div>
      </div>

      {/* Ground truth label */}
      <div className="mt-4 pt-3 border-t border-border">
        <p className="font-mono text-[10px] text-text-muted text-center leading-relaxed">
          Ground truth: <span className="text-[#ef4444]">Attack Burst = FRAUD</span> · <span className="text-[#10b981]">Normal simulation = LEGIT</span>
          <br />All transactions scored by real backend model
        </p>
      </div>
    </Panel>
  )
}
