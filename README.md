# 🛡️ Vhack - Real-Time Fraud Shield for the Unbanked

<div align="center">
  
  **A production-ready Stacking Ensemble Machine Learning Architecture for real-time fraud detection**
  
  [![React](https://img.shields.io/badge/React-18.x-61DAFB?logo=react)](https://reactjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org/)
  [![LightGBM](https://img.shields.io/badge/LightGBM-Model-00C0FF?logo=lightgbm)](#)
</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution](#-solution)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Project Structure & Roles](#-project-structure--roles)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Case Management Dashboard](#-case-management-dashboard)

---

## 🚀 Overview

**Vhack** is a comprehensive, production-ready system that provides a robust backend and frontend for real-time fraud detection. At its core, it powers a **Stacking Ensemble Machine Learning Architecture**, analyzing transactions instantly, explaining precisely why a transaction is flagged using SHAP values, and allowing human investigators to make final decisions seamlessly.

### 💡 Value Proposition

> "Safeguard the unbanked with real-time, explainable, and scalable layer-driven AI fraud detection."

---

## 🚨 Problem Statement

As financial inclusion efforts expand to the unbanked, new risks emerge:
- **Evolving Fraud Tactics:** Scammers adapt quickly to static rules.
- **Hidden Behavioral Anomalies:** Not all fraud fits a historical template; some appear as subtle behavioral shifts.
- **Black-Box Suspensions:** Legitimate users get randomly blocked without clear explanation.
- **Development Bottlenecks:** ML Teams building fraud layers often block each other with merge conflicts and integration issues.

---

## 🛡️ Solution

Vhack brings a structured, multi-layered approach:

1. **Layer 1 - Supervised Model (LightGBM):** Targets *known* fraud patterns from labeled data.
2. **Layer 1 - Unsupervised Model (Isolation Forest):** Targets *behavioral anomalies* without labels, capturing zero-day threats.
3. **Layer 2 - Meta-Learner (Logistic Regression):** Intelligently combines L1 predictions into a final 0-100 risk score based on confidence.
4. **Zero-Conflict Engineering:** The workspace is built specifically so independent engineers can develop their respective models completely parallel to one another.

---

## ✨ Key Features

### 🧮 Machine Learning & Risk Scoring
| Feature | Description |
|---------|-------------|
| **Stacking Architecture** | Combines L1 Supervised & Unsupervised predictions using L2 Meta-Learning |
| **LightGBM Integration** | Blazingly fast gradient boosting for known fraud vectors |
| **PyOD Isolation Forest** | Behavioral anomaly detection without historical labels |
| **Modular Contracts** | Guaranteed model compatibility using strict Python Interfaces |

### 🕵️ Investigation & Explainability
| Feature | Description |
|---------|-------------|
| **SHAP Explainability** | Real-time Waterfall charts showing feature importance per transaction |
| **Case Management** | Admin dashboard for investigators to override or review 'FLAG' cases |
| **Live Simulator** | Real-time stream of incoming transactions vs engine KPIs |

### 🏗️ Engineering & DevOps
| Feature | Description |
|---------|-------------|
| **Zero Merge Conflict Design** | Dedicated sandboxes for Member A and Member B |
| **FastAPI Backend** | High-concurrency async endpoints for model inference |

---

## 🛠️ Tech Stack

### Frontend UI & Investigation
```text
• React 18          # UI Framework
• Vite              # Build Tool & Dev Server
• TailwindCSS       # Utility-first CSS
• Recharts          # Live Dashboard Analytics
```

### Backend API & ML Engine
```text
• FastAPI           # High-performance async Python framework
• Uvicorn           # ASGI Web Server
• LightGBM          # Supervised Layer 1 Model
• PyOD              # Unsupervised Layer 1 Model
• Scikit-Learn      # Meta-Learner & Evaluations
• SHAP              # Explainable AI
```

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                     FRONTEND (React)                         │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────────────┐  │
│  │ Live Simulator │ │ Insights Hub  │ │  Case Management  │  │
│  └──────┬───────┘ └───────┬───────┘ └──────────┬──────────┘  │
│         └─────────────────┼────────────────────┘             │
└───────────────────────────┼──────────────────────────────────┘
                            │ REST API 
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                         │
│                                                              │
│              ┌───────────────────────────┐                   │
│              │        /predict           │                   │
│              └────────────┬──────────────┘                   │
│                           │                                  │
│ ┌─────────────────────────▼────────────────────────────────┐ │
│ │                  ML INFERENCE ENGINE                     │ │
│ │                                                          │ │
│ │   ┌─────────────────┐        ┌─────────────────┐         │ │
│ │   │ Layer 1 (Member A)│        │ Layer 1 (Member B)│         │ │
│ │   │    LightGBM     │        │ Isolation Forest│         │ │
│ │   └────────┬────────┘        └────────┬────────┘         │ │
│ │            │                          │                  │ │
│ │            └────────────┬─────────────┘                  │ │
│ │                         ▼                                │ │
│ │              ┌─────────────────────┐                     │ │
│ │              │ Layer 2 Meta-Learner│                     │ │
│ │              │ Logistic Regression │                     │ │
│ │              └─────────────────────┘                     │ │
│ └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure & Roles

We have designed the repository for **zero merge conflicts**. Each ML engineer works in their own dedicated directory maintaining the exact `interface.py` specifications.

```text
Vhack/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI Routes (/predict, /explain)
│   │   ├── ml/           # 🧠 ML Pipeline (Core)
│   │   │   ├── interface.py       # Contract both L1 models MUST follow
│   │   │   ├── supervised/        # 👤 Member A works here (LGBM)
│   │   │   ├── unsupervised/      # 👤 Member B works here (PyOD)
│   │   │   └── ensemble/          # 🔗 Combines L1 models (Meta)
│   │   └── services/     # Business logic & risk classification
│   ├── models/           # Generated .pkl models
│   ├── data/             # Training datasets
│   └── requirements.txt  # Python Dependencies
├── frontend/
│   ├── src/
│   │   ├── components/   # React Reusable UI
│   │   │   ├── dashboard/
│   │   │   ├── investigation/
│   │   │   └── insights/
│   │   ├── services/     # API Integration
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
└── README.md             # This file
```

---

## ⚙️ Installation & Setup

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
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
uvicorn app.api.main:app --reload
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

This project includes a fully functional React frontend dashboard designed for specialized investigators.

1. **Live Simulator & Dashboard**: Streams real-time transactions hitting the engine, showing active KPIs, fraud rates, and live 3-layer ensemble scoring.
2. **Transaction Investigation (Case Management)**: Investigates transactions with real-time API integrations (`/predict` and `/explain`).
3. **SHAP Explainability Waterfall**: Visualizes feature importance and contribution to the final ensemble score.
4. **Model Insights**: In-depth visualization of the 3-layer ensemble model's confidence, agreement gaps, and threshold tuning.