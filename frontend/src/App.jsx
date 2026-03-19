import React, { useState } from 'react'
import Header from './components/layout/Header'
import Sidebar from './components/layout/Sidebar'
import Dashboard from './pages/Dashboard'
import TransactionInvestigation from './pages/TransactionInvestigation'
import RiskRadar from './pages/RiskRadar'

import FraudAnalysis from './pages/FraudAnalysis'
import FraudSimulator from './pages/FraudSimulator'
import TransactionLab from './pages/TransactionLab'
import { useTransactionEngine } from './hooks/useTransactionEngine'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')
  const engine = useTransactionEngine()

  return (
    <div className="flex h-screen bg-bg-base overflow-hidden">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <div className="flex-1 flex flex-col h-full bg-grid-pattern bg-grid">
        <Header engine={engine} activeTab={activeTab} />
        <main className="flex-1 overflow-y-auto overflow-x-hidden p-6 relative">
          {activeTab === 'dashboard' && <Dashboard engine={engine} />}
          {activeTab === 'search' && <TransactionInvestigation engine={engine} />}
          {activeTab === 'radar' && <RiskRadar engine={engine} />}
          {activeTab === 'analysis' && <FraudAnalysis engine={engine} />}
          {activeTab === 'simulator' && <FraudSimulator engine={engine} />}
          {activeTab === 'lab' && <TransactionLab />}
        </main>
      </div>
    </div>
  )
}

export default App
