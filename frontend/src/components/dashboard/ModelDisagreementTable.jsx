import React from 'react'
import Panel from '../shared/Panel'
import { AlertCircle, ChevronRight } from 'lucide-react'
import clsx from 'clsx'
import { formatCurrency } from '../../utils/formatters'

export default function ModelDisagreementTable({ engine, onSelect }) {
  const { modelDisagreements = [] } = engine

  const ScoreCell = ({ value, colorClass, isPercentage = true }) => (
    <td className={clsx("py-3 px-4 text-center font-mono font-bold", colorClass)}>
      {(value * 100).toFixed(1)}{isPercentage ? '%' : ''}
    </td>
  )

  return (
    <Panel 
      title="Model Disagreement Cases — Where LightGBM and IsoForest Conflict Most" 
      icon={<AlertCircle className="w-4 h-4 text-amber-500" />}
      className="col-span-full"
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left font-mono text-[10px]">
          <thead>
            <tr className="border-b border-border text-text-muted uppercase tracking-widest font-bold">
              <th className="py-3 px-4">TX ID</th>
              <th className="py-3 px-4 text-right">Amount</th>
              <th className="py-3 px-4 text-center text-cyan-400">LGB</th>
              <th className="py-3 px-4 text-center text-purple-400">ISO</th>
              <th className="py-3 px-4 text-center text-amber-400">BEH</th>
              <th className="py-3 px-4 text-center">Final</th>
              <th className="py-3 px-4 text-center text-amber-500">Delta</th>
              <th className="py-3 px-4">Reason</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/30">
            {modelDisagreements.length === 0 ? (
              <tr>
                <td colSpan="8" className="py-12 text-center text-text-muted italic text-[12px]">
                  No significant model disagreements detected in the current stream.
                </td>
              </tr>
            ) : (
              modelDisagreements.map((item) => (
                <tr 
                  key={item.id}
                  onClick={() => onSelect?.(item._raw)}
                  className="hover:bg-bg-200/50 cursor-pointer transition-colors group border-l-2 border-l-transparent hover:border-l-amber-500"
                >
                  <td className="py-3 px-4 text-text-primary">
                    <span className="bg-bg-300 px-2 py-1 rounded text-[10px] text-text-secondary border border-border">
                      {item.id.slice(0, 11)}...
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right text-text-primary font-bold">
                    {formatCurrency(item.amount).replace('MYR', 'RM')}
                  </td>
                  <ScoreCell value={item.lgbScore} colorClass="text-cyan-400" />
                  <ScoreCell value={item.isoScore} colorClass="text-purple-400" />
                  <ScoreCell value={item.behScore} colorClass="text-amber-400" />
                  <td className="py-3 px-4 text-center">
                    <span className={clsx(
                      "px-3 py-1 rounded-full text-[9px] font-bold tracking-widest border",
                      item.ensembleScore < 0.35 ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" :
                      item.ensembleScore < 0.70 ? "bg-amber-500/10 text-amber-400 border-amber-500/20" :
                      "bg-red-500/10 text-red-400 border-red-500/20"
                    )}>
                      {item.ensembleScore < 0.35 ? 'APPROVE' : item.ensembleScore < 0.70 ? 'FLAG' : 'BLOCK'}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center font-bold text-amber-500">
                    ±{(item.delta * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 text-text-muted max-w-[200px] truncate">
                    {item.reason}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
