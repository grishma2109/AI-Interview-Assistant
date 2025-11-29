# AI Interview Agent

## Overview
The AI Interview Agent is a Streamlit-based application designed to conduct automated candidate interviews, analyze responses, and generate a full report including scores, feedback, and resume insights.

---

## Features
- Candidate information collection (name, email, phone, role)
- Resume upload & parsing (PDF)
- AI-generated interview questions (technical & HR)
- Candidate answer evaluation with feedback
- Voice question support (optional)
- Session management for multi-step interviews
- PDF report generation with overall score and eligibility
- Easy-to-use web interface

---

## Limitations
- Requires Gemini API key for AI scoring and summarization
- Only supports text-based PDFs for resume parsing
- AI scoring may not fully replicate human judgment
- Voice transcription may require additional API setup (OpenAI Whisper)

---

## Tech Stack & APIs
- **Frontend:** Streamlit
- **PDF Handling:** PyPDF, FPDF
- **Data Handling:** pandas
- **LLM:** Google Gemini 1.5
- **Optional Voice Transcription:** OpenAI Whisper
- **Python Utilities:** re, json, pathlib, datetime

---

## Setup & Run Instructions

1. Clone the repository:
```bash
git clone <YOUR_REPO_URL>
cd AI-Interview-Agent
