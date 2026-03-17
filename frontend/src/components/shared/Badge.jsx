import React from 'react'
import clsx from 'clsx'

export default function Badge({ decision }) {
  const isApprove = decision === 'APPROVE' || decision === 'LOW'
  const isFlag = decision === 'FLAG' || decision === 'MEDIUM'
  const isBlock = decision === 'BLOCK' || decision === 'HIGH' || decision === 'CRITICAL'

  return (
    <span className={clsx(
      "px-2 py-[2px] rounded-pill font-mono text-[9px] whitespace-nowrap border font-bold tracking-widest uppercase",
      {
        'bg-[rgba(0,255,136,0.1)] text-[#00ff88] border-[#00ff88]/30': isApprove,
        'bg-[rgba(255,170,0,0.1)] text-[#ffaa00] border-[#ffaa00]/30': isFlag,
        'bg-[rgba(255,68,68,0.1)] text-[#ff4444] border-[#ff4444]/30': isBlock,
        'bg-bg-200 text-text-secondary border-border': !isApprove && !isFlag && !isBlock
      }
    )}>
      {decision}
    </span>
  )
}
