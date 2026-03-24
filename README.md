<div align="center">

<br>

<pre align="center">
███████╗██████╗  █████╗ ██╗   ██╗██████╗ ███████╗██╗  ██╗██╗███████╗██╗     ██████╗      █████╗ ██╗
██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗    ██╔══██╗██║
█████╗  ██████╔╝███████║██║   ██║██║  ██║███████╗███████║██║█████╗  ██║     ██║  ██║    ███████║██║
██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║    ██╔══██║██║
██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝███████║██║  ██║██║███████╗███████╗██████╔╝    ██║  ██║██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝     ╚═╝  ╚═╝╚═╝
</pre>

**Real-Time Fraud Detection for the Unbanked**

<br>

### *"Three layers of intelligence. One score you can trust. Every transaction protected."*

<br>

**A production-ready Stacking Ensemble Machine Learning Architecture for real-time fraud detection**

*Built for V Hack 2026 · Case Study 2: Safeguarding the Unbanked*

*Powered by LightGBM · Isolation Forest · SHAP Explainability · FastAPI · React*

<br>

[![React](https://img.shields.io/badge/React-18.x-61DAFB?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-00C0FF?logo=lightgbm)](#)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-FF6F00)](#)

<br>

</div>

---

## 📑 Table of Contents

- [Team Members](#-team-members)
- [Chosen Case Study](#-chosen-case-study)
- [Proposed Solution](#-proposed-solution)
- [Key Features](#-key-features)
- [Technologies Used](#-technologies-used)
- [System Architecture](#-system-architecture)
- [Project Structure & Roles](#-project-structure--roles)
- [Business Model](#-business-model)
- [Market Segment](#-market-segment)
- [Competitor Analysis](#-competitor-analysis)
- [Future Improvements & Expansion](#-future-improvements--expansion)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)

---

## 👥 Team Members

| Name | Role | Key Contributions |
|:---|:---|:---|
| **Chua Zhu Heng** | Full-Stack Developer | API development (FastAPI), Ensemble scoring pipeline, Isolation Forest model, system integration |
| **Daniel Koh** | Machine Learning Engineer | LightGBM supervised model, Ensemble weight calibration, Empirical threshold tuning |
| **Yeethong** | Frontend Developer | React dashboard UI, data visualization, user experience design |

---

## 📋 Chosen Case Study

**Case Study 2 — Digital Trust: Real-Time Fraud Shield for the Unbanked**

We chose this case study for four reasons:

- ASEAN has 290M+ unbanked adults entering the digital economy through e-wallets as their only financial tool — they are the most vulnerable and the least protected
- Malaysia lost RM1.6B to scams in 2024 alone, with only ~2% of victims recovering their funds
- Our team's expertise in ML directly maps to fraud detection and anomaly scoring — we could build something real, not theoretical
- It addresses SDG 8.10 directly — financial inclusion through trusted digital payments

---

## 💡 Proposed Solution

FraudShield AI is powered by the **TriShield Ensemble Engine**, a 3-layer ML fusion model that scores every transaction in real-time.

### The 3 Layers

**Layer 1 — LightGBM (Supervised)**
Detects known fraud patterns learned from historical labeled transaction data. Chosen over XGBoost for faster inference, better handling of imbalanced data, and native support for categorical features.

**Layer 2 — Isolation Forest (Unsupervised)**
Learns what normal transactions look like and flags anything that deviates — catches fraud patterns that have never appeared before, without needing any labeled examples.

**Layer 3 — BehavioralProfiler (Rule-based)**
Scores contextual red flags based on human-defined rules including account fully drained to a new recipient, amount far exceeding user average, suspicious IP or new device, and rapid session velocity. Every decision has a human-readable reason — no black box.

### Why Ensemble

No single model catches everything. LightGBM misses novel fraud patterns. Isolation Forest misses contextual red flags. BehavioralProfiler has no memory of historical data. Together, three complementary weaknesses become one strong system.

### Supporting Pillars

**Privacy-First Architecture**
Sender and receiver IDs are irreversibly hashed with SHA-256 before any model sees the data. Laplace differential privacy noise is applied on aggregate exported scores. PDPA (Malaysia) and GDPR compliant by design, not as an afterthought.

**Empirical Tuning**
Ensemble weights and dual thresholds (approve / flag / block) are calibrated on validation data — not guessed at 0.5. Separate thresholds create a review zone for borderline cases instead of auto-blocking legitimate users.

**Closed-Loop Retraining**
Admin decisions on flagged transactions are stored as new labeled data and fed back into the retraining pipeline. The model continuously adapts to new fraud patterns without manual intervention from data scientists.

> *"Fraud evolves. So do we."*

---

## ✨ Key Features

### 🧮 Machine Learning & Risk Scoring
| Feature | Description |
|:---|:---|
| **Stacking Ensemble** | Combines L1 Supervised, Unsupervised & Behavioural predictions via weighted fusion |
| **LightGBM Integration** | Blazingly fast gradient boosting for known fraud vectors |
| **PyOD Isolation Forest** | Behavioural anomaly detection without historical labels |
| **Behavioural Rule Engine** | Domain-expert rules: drain detection, amount deviation, risky context, rapid sessions |
| **Empirical Threshold Tuning** | Optimal approve/flag/block thresholds derived from precision-recall convergence analysis |
| **Privacy Masking** | SHA-256 PII hashing before model inference — privacy by design |

### 🕵️ Investigation & Explainability
| Feature | Description |
|:---|:---|
| **SHAP Explainability** | Real-time waterfall charts showing weighted feature importance per transaction |
| **Case Management** | Admin dashboard for investigators to review and override FLAG cases |
| **Live Simulator** | Real-time transaction stream with engine KPIs, fraud rates, and live scoring |
| **Transaction Lab** | Interactive model tuning lab with threshold sensitivity analysis |

### 📊 Dashboards & Visualisation
| Feature | Description |
|:---|:---|
| **Risk Radar** | Live risk heatmap and distribution analysis across all transactions |
| **Model Insights** | 3-layer ensemble agreement gaps, weight analysis, and confidence breakdown |
| **Fraud Analysis** | Detailed pattern analysis with model disagreement detection |

---

## 🛠️ Technologies Used

### Machine Learning
| Component | Technology | Reason |
|:---|:---|:---|
| Supervised Model | LightGBM | Fast inference, handles imbalance, tree-based |
| Unsupervised Model | Isolation Forest (scikit-learn) | Label-free anomaly detection |
| Behavioral Engine | Custom BehavioralProfiler | Interpretable, works day one |
| Class Imbalance | SMOTE | Prevents model ignoring rare fraud cases |
| Model Serialization | Joblib | Fast .pkl artifact loading |

### Backend
| Component | Technology |
|:---|:---|
| API Framework | FastAPI (Python async) |
| Schema Validation | Pydantic |
| Privacy Layer | SHA-256 hashing + Laplace DP |

### Frontend
| Component | Technology |
|:---|:---|
| UI Framework | React + Vite |
| Styling | Tailwind CSS |
| Hosting | Vercel |

### Data & Storage
| Component | Technology |
|:---|:---|
| Database | PostgreSQL via Supabase |
| Auth & Storage | Supabase |
| Model Files | .pkl artifacts |

### Datasets
- PaySim synthetic financial transaction dataset
- IEEE-CIS Fraud Detection dataset

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                     FRONTEND (React + Vite)                  │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────────────┐  │
│  │ Live Simulator │ │  Risk Radar   │ │  Case Management  │  │
│  ├──────────────┤ ├───────────────┤ ├─────────────────────┤  │
│  │  Dashboard   │ │ Model Insights│ │  Transaction Lab   │  │
│  └──────┬───────┘ └───────┬───────┘ └──────────┬──────────┘  │
│         └─────────────────┼────────────────────┘             │
└───────────────────────────┼──────────────────────────────────┘
                            │ REST API (JSON)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Uvicorn)                │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  API Endpoints                                       │    │
│  │  POST /predict · POST /explain · GET /stats · GET /config │
│  └─────────────────────────┬────────────────────────────┘    │
│                            │                                  │
│  ┌─────────────────────────▼────────────────────────────┐    │
│  │              ENSEMBLE INFERENCE ENGINE                │    │
│  │                                                       │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │    │
│  │  │   LightGBM   │ │  Isolation   │ │ Behavioural  │  │    │
│  │  │ (Supervised)  │ │   Forest     │ │ Rule Engine  │  │    │
│  │  │   Layer 1-A  │ │  Layer 1-B   │ │  Layer 1-C   │  │    │
│  │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘  │    │
│  │         └────────────────┼────────────────┘           │    │
│  │                          ▼                            │    │
│  │           ┌─────────────────────────┐                 │    │
│  │           │ Weighted Ensemble Fusion │                 │    │
│  │           │  + SHAP Explainability   │                 │    │
│  │           └────────────┬────────────┘                 │    │
│  │                        ▼                              │    │
│  │              APPROVE / FLAG / BLOCK                    │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Privacy Layer │  │ Preprocessing│  │ .pkl Models  │       │
│  │ (SHA-256 PII) │  │ & Features   │  │ (Serialised) │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure & Roles

Designed for **zero merge conflicts** — each ML engineer works in their own dedicated directory.

```text
Vhack/
├── backend/  # 👤 Zhu Heng/Daniel — backend
│   ├── api/ 
│   │   ├── main.py              # FastAPI app + route handlers
│   │   ├── inference.py         # 🧠 EnsembleEngine (core ML pipeline)
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── behavioural.py       # Behavioural Rule Engine (L1-C)
│   │   └── privacy.py           # PII masking (SHA-256)
│   ├── models/
│   │   ├── supervised/          # 👤 Daniel — LightGBM artifacts
│   │   ├── unsupervised/        # 👤 Zhu Heng — Isolation Forest artifacts
│   │   └── ensemble/            # 🔗 Ensemble config & weights
│   ├── data/                    # Training datasets
│   └── requirements.txt         # Python dependencies
├── frontend/          # 👤 Yeethong — frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx            
│   │   │   ├── LiveSimulator.jsx        # Real-time transaction stream
│   │   │   ├── TransactionInvestigation.jsx  # Case management + SHAP
│   │   │   ├── RiskRadar.jsx            # Risk distribution heatmap
│   │   │   ├── FraudAnalysis.jsx        # Model insights & analysis
│   │   │   ├── FraudSimulator.jsx       # Fraud scenario testing
│   │   │   └── TransactionLab.jsx       # Threshold tuning lab
│   │   ├── components/          # Reusable UI components
│   │   ├── hooks/               # Custom React hooks (useTransactionEngine)
│   │   └── App.jsx              # App entry point & routing
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 💼 Business Model

FraudShield AI operates as a **B2B SaaS API** — Super Apps, neo-banks, and e-wallet operators license FraudShield as a managed fraud detection service.

### Three Revenue Streams

- **Real-Time Fraud Scoring API** — scores every transaction in ~23ms, returns Approve / Flag / Block with human-readable reasons
- **Dashboard & Analytics** — live fraud heatmaps, anomaly trends, false positive tracking for compliance teams
- **Model Retraining & Updates** — automatic retraining on new fraud signals keeps detection ahead of evolving tactics

### Pricing Tiers

| Tier | Target | Price |
|:---|:---|:---|
| Starter | Regional fintechs, under 1M transactions/month | RM10,000/month |
| Growth | Mid-size e-wallets, 1–10M transactions/month | RM30,000/month |
| Enterprise | GCash, Maybank, GrabPay, 10M+ transactions/month | Custom |

### Primary Customers
- Malaysia: Touch 'n Go, Boost
- Philippines: GCash, Maya, GoPay, TrueMoney
- Regional: Grab, TikTok Shop Pay

### Secondary Customers
- Malaysia & SEA banks: Maybank, CIMB, RHB
- Philippines commercial banks: BDO, BPI, UnionBank
- Neo-banks: TONIK, GXS Bank, SeaBank
- Regulators: BSP / BNM mandated compliance push

---

## 📊 Market Segment

| | Market | Size |
|:---|:---|:---|
| TAM | Global AI fraud detection market | RM382B by 2033 (CAGR 22.6%) |
| SAM | Asia Pacific fraud detection | RM11.4B (CAGR 26.8% — fastest globally) |
| SOM | ASEAN target: PH, MY, TH, ID | RM708M (entry via PH & MY) |

**Why ASEAN is the right market:**
- Identity fraud surged 121% in ASEAN in 2024
- 20–30 e-wallet platforms and 50+ banks across target countries
- Capturing just 2% of the APAC market at Growth tier average equals RM228M ARR
- BSP and BNM are mandating stronger fraud controls — regulatory tailwind, not headwind

---

## ⚔️ Competitor Analysis

| Feature | FraudShield AI | Stripe Radar | SEON | Feedzai | FICO Falcon |
|:---|:---|:---|:---|:---|:---|
| Approach | 3-layer ensemble | ML (black box) | Rule-based + digital footprint | ML scoring engine | Neural Network |
| Privacy-First | ✅ | ❌ | ❌ | ❌ | Partial |
| Explainability | ✅ | ❌ | ✅ | Partial | ❌ |
| Closed-Loop Retraining | ✅ | ❌ | Partial | ✅ | Partial |
| Latency | ~23ms ✅ | ~50ms | ~100ms | ~80ms | ~100ms |
| PDPA Compliant | ✅ | ❌ | ❌ | ❌ | ❌ |
| ASEAN-Tuned | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pricing | Freemium (RM) | USD Enterprise | USD Subscription | USD Enterprise | USD Enterprise |

Every competitor solves part of the problem. None combines ASEAN localisation, pre-inference privacy, ensemble modelling, and accessible pricing simultaneously.

---

## 🚀 Future Improvements & Expansion

**1. Graph Neural Network (GNN) for Fraud Ring Detection**
Current models score transactions individually. GNN maps relationships between accounts to detect coordinated fraud rings and money mule networks — catching fraud that looks normal in isolation but suspicious as a network pattern.

**2. Federated Learning — Cross-Bank Collaboration**
Each bank trains the model locally and only shares model weight updates, never raw transaction data. This enables stronger collective fraud detection across institutions while maintaining full data privacy and regulatory compliance.

**3. Mobile SDK for Super App Integration**
A lightweight SDK that Super Apps like GrabPay, TNG, and Maya can embed directly. No external API calls needed — fraud scored natively inside the app. Faster, offline-capable, and removes the biggest barrier to adoption: integration complexity.

**Longer-term vision:** Become ASEAN's trusted, open fraud intelligence layer — accessible to every operator, not just those with enterprise budgets.

---

## ⚙️ Installation & Setup

### Prerequisites

| Requirement | Version | Purpose |
|:---|:---:|:---|
| Node.js | 18.x+ | Frontend runtime |
| Python | 3.10+ | Backend runtime |
| Git | Latest | Version control |

### 1. Clone & Setup Environment

```bash
git clone <your-repository-url>
cd Vhack
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

---

## 🖥️ Running the Application

### Start Backend API
From the `backend/` directory:
```bash
uvicorn api.main:app --reload
```
View the interactive API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start Frontend Dashboard
From the `frontend/` directory:
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🔍 Case Management Dashboard

This project includes a fully functional React frontend dashboard designed for specialised investigators.

| Page | Description |
|:---|:---|
| **Dashboard** | Real-time KPIs — total transactions, approval/flag/block rates, average latency, fraud rate estimate |
| **Live Simulator** | Streams real-time transactions hitting the engine with live 3-layer ensemble scoring |
| **Transaction Investigation** | Case management with real-time `/predict` and `/explain` API integration + SHAP waterfall charts |
| **Risk Radar** | Risk distribution analysis and heatmap visualisation across all processed transactions |
| **Fraud Analysis (Model Insights)** | 3-layer ensemble confidence breakdown, model agreement gaps, and weight analysis |
| **Transaction Lab** | Interactive threshold tuning with precision-recall convergence analysis |

---

<div align="center">

**© 2026 · Built for V Hack 2026 · Case Study 2: Safeguarding the Unbanked**

</div>