import React from 'react'
import clsx from 'clsx'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export default function KpiCard({
  value,
  label,
  subText,
  trend,         // 'up', 'down', 'neutral'
  trendGood,     // true if up is good, false if up is bad
  statusDot,     // 'live', 'offline', or null
}) {
  const isUp = trend === 'up'
  const isDown = trend === 'down'
  
  const TrendIcon = isUp ? TrendingUp : isDown ? TrendingDown : Minus
  let trendColor = 'text-text-muted'
  if (isUp) trendColor = trendGood ? 'text-approve' : 'text-block'
  if (isDown) trendColor = trendGood ? 'text-block' : 'text-approve'

  return (
    <div className="bg-bg-100 rounded-2xl relative overflow-hidden transition-all duration-300 p-5 pl-6 flex flex-col justify-between border border-border">
      
      <div className="relative z-10 flex flex-col h-full justify-between">
        <div className="flex justify-between items-start mb-2">
            <h3 className="section-label">{label}</h3>
            {statusDot && (
              <span className={clsx(
                "flex items-center gap-1 font-mono text-[9px] font-bold uppercase tracking-wider",
                statusDot === 'live' ? 'text-[#10b981]' : 'text-[#ef4444]'
              )}>
                <span className={clsx(
                  "w-1.5 h-1.5 rounded-full inline-block",
                  statusDot === 'live' ? 'bg-[#10b981] animate-pulse-slow' : 'bg-[#ef4444]'
                )} />
                {statusDot === 'live' ? 'LIVE' : 'OFFLINE'}
              </span>
            )}
        </div>
        <div className="flex items-end gap-3 select-none">
          <div className="font-mono font-bold text-[24px] leading-none text-text-primary pb-1">
            {value}
          </div>
          {trend && (
            <div className={clsx("flex items-center pb-[2px]", trendColor)}>
              <TrendIcon size={18} strokeWidth={2.5} />
            </div>
          )}
        </div>
        {subText && (
          <p className="font-mono text-[11px] text-text-muted mt-2 opacity-80">
            {subText}
          </p>
        )}
      </div>
    </div>
  )
}
