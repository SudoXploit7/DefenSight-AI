# DefenSight AI - Autonomous Network Defense Copilot

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RAG](https://img.shields.io/badge/RAG-Powered-purple.svg)]()

> **AI-powered security log analysis platform that transforms raw firewall, IDS, and network logs into actionable intelligence using Retrieval-Augmented Generation (RAG) and Large Language Models.**

---

## Screenshots

### User Registration (New User)
![User Registration](project_screenshots/register.png)  
*A secure onboarding flow where new users create an account. Passwords are hashed using bcrypt and never stored in plaintext.*

---

### Login & Authentication
![Login Screen](project_screenshots/login.png)  
*Multi-user authentication system with session-based login protection and bcrypt hashing.*

---

### Upload & Normalization Trigger
![Upload Interface](project_screenshots/upload.png)  
*Upload raw security logs (CSV, JSON, XML, LOG, TXT). Files are normalized into a unified schema and indexed automatically.*

---

### Normalization Summary View
![Normalization Summary](project_screenshots/normalization.png)  
*Displays total parsed records, detected log types, and key metadata extracted from uploaded files.*

---

### Detailed Normalized Data View
![Normalized Files Detail](project_screenshots/normalized_detail.png)  
*Tabular structured JSON visualization with preserved fields such as timestamps, severity, source/destination IPs, protocol, and attack category.*

---

### Technical Analysis Summary (AI-Generated)
![Technical Summary](project_screenshots/tech_summary.png)  
*Deep-dive technical summary generated using Retrieval-Augmented Generation over normalized log data.*

---

### Executive-Level Summary
![Executive Summary](project_screenshots/exec_summary.png)  
*High-level business-focused security insights written for leadership and non-technical stakeholders.*

---

### Interactive SOC Assistant (Chat Interface)
![Chat Interface](project_screenshots/chat.png)  
*Ask security-focused questions in natural language and receive context-aware answers grounded in your logs.*

---

### Email Report Delivery
![Email Report](project_screenshots/email.png)  
*Generate reports and email them securely with one click via integrated SMTP support.*

---

### Downloadable PDF Report
![PDF Report](project_screenshots/pdf_report.png)  
*Export fully formatted security assessment reports (technical + executive) for audit and documentation.*

---

## Key Features

### **Security Analysis**
- **Multi-format log ingestion**: CSV, XML, JSON, LOG, TXT
- **Intelligent normalization**: Automatic type detection and field extraction
- **Semantic search**: 768-dimensional vector embeddings for contextual retrieval
- **RAG-powered insights**: Grounded AI responses using actual log evidence

### **AI Capabilities**
- **Groq LLM integration**: Llama 3.3 70B with 128k context window
- **Conversational interface**: Ask questions in natural language
- **Automated reporting**: Technical and executive summaries
- **Threat correlation**: Identify attack patterns across disparate logs

### **Data Processing**
- **Vector database**: ChromaDB with HNSW indexing for fast similarity search
- **Batch processing**: Efficient embedding generation (64 docs/batch)
- **Session management**: Append or start fresh analysis workflows
- **Real-time ingestion**: Watch folder for automatic log processing

### **Authentication & Security**
- **Multi-user system**: Flask-Login with secure session management
- **Password hashing**: Bcrypt with salt for credential storage
- **SQLite database**: Lightweight user management
- **Protected routes**: Login required for all analysis features

---

## Architecture

```
                         ┌──────────────────────────────────────┐
                         │            User Interface            │
                         │   Flask Web App + HTML Templates     │
                         └───────────────┬──────────────────────┘
                                         │
                                         │  User uploads logs /
                                         │  asks questions
                                         ▼
                    ┌──────────────────────────────┐
                    │     Normalization Engine     │
                    │        (format_con.py)       │
                    └───────────────┬──────────────┘
                                    │
                                    │  Converts logs into
                                    │  structured JSON
                                    ▼
                     ┌────────────────────────────┐
                     │  Vectorization + Indexing  │
                     │   (live_ingest.py RAG DB)  │
                     └───────────────┬────────────┘
                                     │
                                     │  Create embeddings
                                     ▼
         ┌────────────────────────────────────────────────────────────┐
         │                      Vector Database                       │
         │                          ChromaDB                          │
         │                                                            │
         │  • Stores 768-dim sentence embeddings                      │
         │  • Supports semantic similarity search                     │
         │  • Uses HNSW indexing for fast recall                      │
         └───────────────┬────────────────────────────────────────────┘
                         │
                         │ Retrieve Top-K Relevant Chunks
                         ▼
                 ┌────────────────────────────┐
                 │       RAG Engine           │
                 │      (rag_engine.py)       │
                 └───────────────┬────────────┘
                                 │
                                 │ Build prompt with retrieved context
                                 ▼
                     ┌───────────────────────────┐
                     │         Groq API          │
                     │     (LLaMA 3.3 model)     │
                     └───────────────┬───────────┘
                                     │
                                     │ AI response /
                                     │ report generation
                                     ▼
                         ┌───────────────────────────────┐
                         │       Final Output Layer      │
                         │   • Technical Summary         │
                         │   • Executive Summary         │
                         │   • PDF Export                │
                         │   • Chat Assistant            │
                         └───────────────────────────────┘

```

**Tech Stack:**
- **Backend**: Python 3.8+, Flask 3.0
- **Vector DB**: ChromaDB 0.4.22
- **Embeddings**: SentenceTransformers (multi-qa-mpnet-base-dot-v1)
- **LLM**: Groq API (Llama 3.3 70B Versatile)
- **Auth**: Flask-Login, bcrypt
- **PDF**: ReportLab
- **Frontend**: Bootstrap 5, Vanilla JS

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Groq API key
- 4GB+ RAM (8GB recommended)
- 10GB+ free disk space

### Installation

```bash
git clone https://github.com/SudoXploit7/DefenSight-AI.git
cd DefenSight-AI
```

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate   # macOS/Linux
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

```bash
python init_db.py init
python init_db.py create-admin
```

```bash
python gui_app.py
```

Visit:
```
http://localhost:5000
```

**Default credentials:**
- Username: `admin`
- Password: `admin123`

> Change these immediately in production.

---

## Project Structure
```
DefenSight AI/
│
├── DefenSight AI_db/          # ChromaDB vector database (auto-generated)
│
├── incoming_logs/             # (Optional) Live-ingest watch folder
│
├── instance/                  # Flask instance folder
│
├── normalized/                # Normalized JSON output files
│
├── project_screenshots/       # Screenshots used in README
│
├── raw_data/                  # Uploaded raw logs (CSV/XML/JSON/LOG/TXT)
│
├── static/                    # CSS & JavaScript assets
│   ├── styles.css
│   └── assistant.js
│
├── templates/                 # HTML UI Pages
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── upload.html
│   ├── normalized_list.html
│   ├── normalize.html
│   └── index.html
│
├── test_data/                 # Sample logs for demo/testing
│
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore rules
│
├── auth.py                    # Authentication logic (Flask-Login + bcrypt)
├── chat.py                    # CLI chat utility (optional)
├── format_con.py              # Log normalization engine
├── gui_app.py                 # Main Flask Web Application
├── LICENSE                    # MIT License
├── live_ingest.py             # Real-time log ingestion pipeline
├── rag_engine.py              # RAG pipeline + Groq API integration
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SudoXploit7/DefenSight-AI&type=Date)](https://star-history.com/#SudoXploit7/DefenSight-AI&Date)

---

## Author

**Soumyadipta Birabar**

- GitHub: [@SudoXploit7](https://github.com/SudoXploit7)
- LinkedIn: [Soumyadipta Birabar](https://linkedin.com/in/soumyadb)

---

<p align="center">
  <strong>Built with ❤️ for the cybersecurity community</strong>
</p>

<p align="center">
  <sub>DefenSight AI • Transforming Security Data into Actionable Intelligence</sub>
</p>
