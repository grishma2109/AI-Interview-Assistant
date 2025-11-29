# app.py — AI Interview Agent (Option C: Full Applicant Profile + Eligibility)
# Usage:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Recommended requirements.txt entries:
# streamlit
# google-generativeai
# pypdf
# fpdf
# openai (optional for Whisper)
# pandas
# python-dotenv (optional)

import os
import re
import json
from pathlib import Path
from datetime import datetime
import streamlit as st

# -------------------------
# Optional third-party imports (handled gracefully)
# -------------------------
try:
    import pypdf
except Exception:
    pypdf = None

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

try:
    import pandas as pd
except Exception:
    pd = None

# Gemini client (try official package; if not present we set genai to None)
def import_genai_safe():
    try:
        import google.generativeai as genai  # preferred package name
        return genai
    except Exception:
        try:
            from google import genai as genai2  # some installs expose this
            return genai2
        except Exception:
            return None

genai = import_genai_safe()

# Helper to call st.rerun compatibly
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            # nothing we can do — just continue
            pass

# -------------------------
# App config & directories
# -------------------------
st.set_page_config(page_title="Interview Agent — Full Profile", layout="wide")
BASE = Path.cwd()
DATA_DIR = BASE / "data_interview"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR = BASE / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Session state defaults
# -------------------------
if "stage" not in st.session_state:
    st.session_state.stage = "start"
if "candidate" not in st.session_state:
    st.session_state.candidate = {}
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "resume_summary" not in st.session_state:
    st.session_state.resume_summary = ""
if "skills" not in st.session_state:
    st.session_state.skills = []
if "resume_years" not in st.session_state:
    st.session_state.resume_years = 0.0
if "tech_questions" not in st.session_state:
    st.session_state.tech_questions = []
if "hr_questions" not in st.session_state:
    st.session_state.hr_questions = []
if "voice_questions" not in st.session_state:
    st.session_state.voice_questions = [
        "Please briefly introduce yourself and highlight one project you're proud of.",
        "Explain a technical concept from your resume in simple terms.",
        "Why do you want this role and how would you contribute in the first 3 months?"
    ]
if "voice_answers" not in st.session_state:
    st.session_state.voice_answers = [None] * len(st.session_state.voice_questions)
if "voice_transcripts" not in st.session_state:
    st.session_state.voice_transcripts = [None] * len(st.session_state.voice_questions)
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "report_summary" not in st.session_state:
    st.session_state.report_summary = {}

# -------------------------
# LLM wrappers (Gemini)
# -------------------------
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") or (os.getenv("GENAI_API_KEY") or None)
# If using a .env file locally, user can call load_dotenv() before running

def configure_genai_if_available():
    if genai is None:
        return False
    try:
        # different SDK versions offer different config APIs
        if hasattr(genai, "configure"):
            genai.configure(api_key=GOOGLE_KEY)
        elif hasattr(genai, "Client"):
            # nothing to configure globally
            pass
        return True
    except Exception:
        return False

GENAI_AVAILABLE = False
if GOOGLE_KEY and genai is not None:
    GENAI_AVAILABLE = configure_genai_if_available()

def gemini_generate_text(prompt: str, model: str = "gemini-1.5") -> str:
    """Try multiple call patterns; return text or raise."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini not configured or api key missing.")
    # try common patterns
    try:
        # Preferred: google.generativeai model.generate_text style (varies by version)
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=GOOGLE_KEY)
            if hasattr(client, "generate_text"):
                resp = client.generate_text(model=model, prompt=prompt)
                # resp may be object or dict-like
                text = getattr(resp, "text", None) or resp.get("output", resp.get("text")) if isinstance(resp, dict) else None
                if text:
                    return text
        # older/newer patterns
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=model, prompt=prompt)
            text = getattr(resp, "text", None) or (resp.get("output_text") if isinstance(resp, dict) else None)
            if text:
                return text
        # fallback: some SDKs use models.generate_content
        if hasattr(genai, "models") and hasattr(genai.models, "generate_content"):
            resp = genai.models.generate_content(model=model, contents=prompt)
            text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
            if text:
                return text
    except Exception as e:
        raise RuntimeError(f"Gemini call failed: {e}") from e
    raise RuntimeError("No supported generate API found in genai package.")

def gemini_transcribe(audio_path: str, model: str = "gemini-1.5") -> str:
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini not configured or api key missing.")
    try:
        # Try common SDK patterns
        if hasattr(genai, "audio") and hasattr(genai.audio, "transcribe"):
            with open(audio_path, "rb") as af:
                resp = genai.audio.transcribe(model=model, file=af)
            text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            if text:
                return text
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=GOOGLE_KEY)
            if hasattr(client, "audio") and hasattr(client.audio, "transcriptions"):
                with open(audio_path, "rb") as af:
                    resp = client.audio.transcriptions.create(file=af, model=model)
                text = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
                if text:
                    return text
    except Exception as e:
        raise RuntimeError(f"Gemini transcription failed: {e}") from e
    raise RuntimeError("Gemini transcription not supported in installed SDK.")

# -------------------------
# Resume helpers
# -------------------------
def extract_text_from_pdf(path: Path) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf not installed. Add to requirements.")
    reader = pypdf.PdfReader(str(path))
    text = ""
    for page in reader.pages:
        try:
            text += (page.extract_text() or "") + "\n"
        except Exception:
            continue
    return text.strip()

def resume_skill_years_from_text(text: str):
    # simple heuristics
    skills = []
    common_skills = ["python","java","sql","c++","c#","javascript","react","node","tensorflow","pytorch","pandas","numpy","ml","dl","aws","docker","kubernetes","git"]
    low = text.lower()
    for s in common_skills:
        if s in low:
            skills.append(s)
    # years of experience
    m = re.search(r"(\d{1,2})\s+(?:years|yrs|year)\b", low)
    yrs = float(m.group(1)) if m else 0.0
    return skills, yrs

def gemini_resume_summary(text: str) -> dict:
    """Use Gemini to produce resume summary (skills, summary, experience)."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini not available.")
    prompt = (
        "You are a resume analyzer. Given the resume plain text below, "
        "produce JSON with keys: skills (list of short skill strings), "
        "summary (2-3 short sentences), experience_years (number), "
        "education (short string). Respond ONLY with JSON.\n\n"
        f"Resume Text:\n{text[:4000]}\n\n"
    )
    out = gemini_generate_text(prompt, model="gemini-1.5")
    # try parse
    try:
        parsed = json.loads(out)
        return parsed
    except Exception:
        # fallback heuristics
        skills, yrs = resume_skill_years_from_text(text)
        return {"skills": skills, "summary": (text[:400] + "...") if text else "", "experience_years": yrs, "education": ""}

# -------------------------
# Question generation & evaluation
# -------------------------
DEFAULT_TECH_QS = [
    "Explain a project where you applied your core technical skill end-to-end.",
    "Describe the architecture of a system you built and the trade-offs you made.",
    "How do you approach debugging a production incident?",
    "Explain synchronous vs asynchronous programming and when to use each.",
    "What is a bottleneck you found in a past project and how you improved it?",
    "Describe how you design RESTful APIs and handle versioning.",
    "Explain a machine-learning model you've trained and how you validated it.",
    "How do you ensure code quality and maintainability in a team?"
]
DEFAULT_HR_QS = [
    "Tell me about a time you faced a conflict at work and how you resolved it.",
    "What are your career goals for the next 2 years?",
    "How do you handle constructive feedback?",
    "Why do you want to join this company / role?"
]

def generate_technical_questions(resume_text: str, role: str, count: int = 8):
    if GENAI_AVAILABLE:
        prompt = f"Generate {count} concise technical interview questions (one per line) for a candidate applying for role '{role}'. Use resume context: {resume_text[:2000]}"
        try:
            out = gemini_generate_text(prompt, model="gemini-1.5")
            qs = [l.strip() for l in out.splitlines() if l.strip()]
            if len(qs) >= count:
                return qs[:count]
        except Exception:
            pass
    return DEFAULT_TECH_QS[:count]

def generate_hr_questions(count: int = 4):
    if GENAI_AVAILABLE:
        prompt = f"Generate {count} concise HR interview questions (one per line)."
        try:
            out = gemini_generate_text(prompt, model="gemini-1.5")
            qs = [l.strip() for l in out.splitlines() if l.strip()]
            if len(qs) >= count:
                return qs[:count]
        except Exception:
            pass
    return DEFAULT_HR_QS[:count]

def evaluate_answer_with_llm(q, ans, qtype, role):
    # try Gemini to return JSON {"score": <0-5>, "remarks": "..."}
    if GENAI_AVAILABLE:
        prompt = (
            "You are an interviewer assistant. Given the question, answer and role, "
            "return ONLY a JSON object: {\"score\": <0-5>, \"remarks\": \"short remarks\"}.\n\n"
            f"Role: {role}\nQuestion: {q}\nType: {qtype}\nCandidate Answer: {ans[:2000]}\n"
        )
        try:
            out = gemini_generate_text(prompt, model="gemini-1.5")
            # extract JSON blob
            try:
                parsed = json.loads(out)
            except Exception:
                import re
                m = re.search(r"\{.*\}", out, re.DOTALL)
                parsed = json.loads(m.group(0)) if m else None
            if parsed and "score" in parsed:
                sc = float(parsed.get("score", 3))
                sc = max(0.0, min(5.0, sc))
                return sc, str(parsed.get("remarks", "")).strip()
        except Exception:
            pass
    # fallback heuristic
    length_score = min(5, max(0, len(ans.split()) // 30))
    skill_hits = sum(1 for s in resume_skill_years_from_text(ans)[0] if s.lower() in ans.lower())
    score = min(5, length_score + skill_hits)
    return float(score or 3), "Heuristic fallback scoring."

# -------------------------
# PDF report generator
# -------------------------
def generate_report_pdf(candidate, skills, qa_history, role, resume_summary_text, category_avgs, overall_score, decision):
    if FPDF is None:
        raise RuntimeError("fpdf not installed.")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Interview Report - {candidate.get('name','')}", ln=True)
    pdf.cell(0, 8, f"Role: {role}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.utcnow().isoformat()}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Resume Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, resume_summary_text or "")
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Scores", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 7, f"Technical avg: {category_avgs.get('technical',0):.2f}", ln=True)
    pdf.cell(0, 7, f"HR avg: {category_avgs.get('hr',0):.2f}", ln=True)
    pdf.cell(0, 7, f"Voice avg: {category_avgs.get('voice',0):.2f}", ln=True)
    pdf.cell(0, 7, f"Overall avg: {overall_score:.2f}", ln=True)
    pdf.cell(0, 7, f"Decision: {decision}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Question Details", ln=True)
    pdf.set_font("Arial", size=10)
    for i, qa in enumerate(qa_history, 1):
        pdf.multi_cell(0, 6, f"{i}. [{qa['type'].upper()}] {qa['question']}")
        pdf.multi_cell(0, 6, f"Answer: {qa['answer']}")
        pdf.cell(0, 6, f"Score: {qa['score']:.2f}  Remarks: {qa['remarks']}", ln=True)
        pdf.ln(2)
    return pdf.output(dest="S").encode("latin-1")

# -------------------------
# START UI
# -------------------------
st.title("🤖 Interview Agent — Full Candidate Profile")

if st.session_state.stage == "start":
    st.markdown(
        """
        <div style='padding:20px;border-radius:8px;background:linear-gradient(180deg,black,#f3f6ff)'>
        <h2>Welcome to the AI Interview Agent</h2>
        <p>This workflow collects candidate details, parses the resume (smart summary), generates questions, collects answers (text + voice), and produces a final candidate report with eligibility decision.</p>
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("🚀 Start Interview"):
        st.session_state.stage = "collect_info"
        safe_rerun()
    st.stop()

# -------------------------
# Candidate info & resume upload
# -------------------------
if st.session_state.stage == "collect_info":
    st.header("Step 1 — Candidate Info & Resume")
    with st.form("candidate_form"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        role = st.text_input("Role applying for")
        uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
        submitted = st.form_submit_button("Save & Continue")
    if submitted:
        if not name or not email or not role:
            st.warning("Please fill name, email and role.")
        else:
            st.session_state.candidate = {"name": name, "email": email, "phone": phone}
            st.session_state.role = role
            # handle resume
            if uploaded:
                dest = DATA_DIR / f"resume_{name.replace(' ','_')}.pdf"
                with open(dest, "wb") as f:
                    f.write(uploaded.getbuffer())
                try:
                    text = extract_text_from_pdf(dest)
                except Exception as e:
                    st.warning(f"Resume parsing failed: {e}")
                    text = ""
                st.session_state.resume_text = text
                # extract skills, years heuristically first
                skills, yrs = resume_skill_years_from_text(text)
                st.session_state.skills = skills
                st.session_state.resume_years = yrs
                # try Gemini summary
                try:
                    if GENAI_AVAILABLE and text.strip():
                        parsed = gemini_resume_summary(text)
                        st.session_state.resume_summary = parsed.get("summary", "") or st.session_state.resume_summary
                        st.session_state.skills = parsed.get("skills", st.session_state.skills)
                        st.session_state.resume_years = parsed.get("experience_years", st.session_state.resume_years)
                    else:
                        # fallback summary
                        st.session_state.resume_summary = (text[:500] + "...") if text else ""
                except Exception as e:
                    st.warning(f"Resume summarization failed: {e}")
                    st.session_state.resume_summary = (text[:500] + "...") if text else ""
            else:
                st.session_state.resume_text = ""
                st.session_state.resume_summary = ""
                st.session_state.skills = []
                st.session_state.resume_years = 0.0

            st.session_state.stage = "generate_questions"
            safe_rerun()
    st.stop()

# -------------------------
# Generate questions
# -------------------------
if st.session_state.stage == "generate_questions":
    st.header("Step 2 — Generating Questions")
    st.info("Generating 8 technical and 4 HR questions (Gemini if available, else defaults).")
    try:
        tech_qs = generate_technical_questions(st.session_state.resume_text, st.session_state.role, count=8)
        hr_qs = generate_hr_questions(count=4)
    except Exception as e:
        st.warning(f"Question generation failed: {e}. Using defaults.")
        tech_qs = DEFAULT_TECH_QS[:8]
        hr_qs = DEFAULT_HR_QS[:4]

    # ensure sizes
    if len(tech_qs) < 8:
        tech_qs = (tech_qs + DEFAULT_TECH_QS)[:8]
    if len(hr_qs) < 4:
        hr_qs = (hr_qs + DEFAULT_HR_QS)[:4]

    st.session_state.tech_questions = tech_qs
    st.session_state.hr_questions = hr_qs
    st.session_state.current_q_index = 0
    st.session_state.stage = "qna"
    safe_rerun()

# -------------------------
# Q&A (text + voice questions)
# -------------------------
elif st.session_state.stage == "qna":
    st.header("Step 3 — Interview (text & voice answers)")
    # resume quick score (0-10)
    skill_score = min(6, len(st.session_state.skills))
    yrs = float(st.session_state.get("resume_years", 0) or 0)
    exp_score = min(4, int(min(4, yrs)))
    resume_score = round((skill_score + exp_score), 1)
    st.metric("Resume quick score (0-10 approx)", resume_score)

    all_qs = [{"q": q, "type": "technical"} for q in st.session_state.tech_questions] + [{"q": q, "type": "hr"} for q in st.session_state.hr_questions]
    idx = st.session_state.current_q_index
    total = len(all_qs) + len(st.session_state.voice_questions)
    st.write(f"Question {idx+1} / {total}")

    # text questions
    if idx < len(all_qs):
        current = all_qs[idx]
        st.subheader(f"Q (text) — {current['type'].capitalize()}")
        st.write(current["q"])
        with st.form(key=f"text_answer_form_{idx}"):
            text_ans = st.text_area("Type your answer here", height=200)
            submit_ans = st.form_submit_button("Submit Answer")
        if submit_ans:
            if not text_ans.strip():
                st.warning("Please enter an answer.")
            else:
                score, remarks = evaluate_answer_with_llm(current["q"], text_ans, current["type"], st.session_state.role)
                st.session_state.qa_history.append({
                    "question": current["q"],
                    "type": current["type"],
                    "answer": text_ans,
                    "score": float(score),
                    "remarks": remarks
                })
                st.session_state.current_q_index += 1
                safe_rerun()
        st.stop()

    # voice questions
    else:
        v_idx = idx - len(all_qs)
        if v_idx < len(st.session_state.voice_questions):
            st.subheader(f"Q (voice) — {v_idx+1} of {len(st.session_state.voice_questions)}")
            st.write(st.session_state.voice_questions[v_idx])
            st.markdown("Record locally and upload audio (wav/m4a/mp3). We'll attempt Gemini transcription first, Whisper next (if configured), else paste transcript manually.")
            uploaded_audio = st.file_uploader(f"Upload audio answer for voice Q{v_idx+1}", type=["wav","m4a","mp3","ogg"], key=f"audio_{v_idx}")
            if uploaded_audio:
                tmp_path = DATA_DIR / f"voice_q{v_idx+1}_{st.session_state.candidate.get('name','candidate')}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                st.success("Audio saved.")
                transcript = None
                # attempt Gemini transcription
                if GENAI_AVAILABLE:
                    try:
                        transcript = gemini_transcribe(str(tmp_path))
                        st.success("Gemini transcription succeeded.")
                    except Exception as e:
                        st.warning(f"Gemini transcription failed: {e}")
                # try OpenAI Whisper as fallback
                if transcript is None and os.getenv("OPENAI_API_KEY"):
                    try:
                        import openai
                        openai.api_key = os.getenv("OPENAI_API_KEY")
                        with open(tmp_path, "rb") as af:
                            resp = openai.Audio.transcribe("whisper-1", af)
                        transcript = resp.get("text", "").strip()
                        st.success("Whisper transcription succeeded.")
                    except Exception as e:
                        st.warning(f"Whisper transcription failed: {e}")
                if transcript:
                    st.text_area("Transcript (edit if needed)", value=transcript, key=f"trans_{v_idx}", height=150)
                    if st.button("Submit voice answer (use transcript)", key=f"submit_trans_{v_idx}"):
                        score, remarks = evaluate_answer_with_llm(st.session_state.voice_questions[v_idx], transcript, "voice", st.session_state.role)
                        st.session_state.voice_answers[v_idx] = str(tmp_path)
                        st.session_state.voice_transcripts[v_idx] = transcript
                        st.session_state.qa_history.append({
                            "question": st.session_state.voice_questions[v_idx],
                            "type": "voice",
                            "answer": transcript,
                            "score": float(score),
                            "remarks": remarks
                        })
                        st.session_state.current_q_index += 1
                        safe_rerun()
                # manual transcript fallback
                manual = st.text_area("Or paste transcript manually (fallback)", key=f"manual_trans_{v_idx}", height=150)
                if st.button("Submit voice answer (manual transcript)", key=f"submit_manual_{v_idx}"):
                    if not manual.strip():
                        st.warning("Please paste or type transcript.")
                    else:
                        score, remarks = evaluate_answer_with_llm(st.session_state.voice_questions[v_idx], manual, "voice", st.session_state.role)
                        st.session_state.voice_answers[v_idx] = str(tmp_path)
                        st.session_state.voice_transcripts[v_idx] = manual
                        st.session_state.qa_history.append({
                            "question": st.session_state.voice_questions[v_idx],
                            "type": "voice",
                            "answer": manual,
                            "score": float(score),
                            "remarks": remarks
                        })
                        st.session_state.current_q_index += 1
                        safe_rerun()
            st.stop()
        else:
            # finished
            st.success("All questions answered. Computing report & eligibility...")
            tech_scores = [q["score"] for q in st.session_state.qa_history if q["type"] == "technical"]
            hr_scores = [q["score"] for q in st.session_state.qa_history if q["type"] == "hr"]
            voice_scores = [q["score"] for q in st.session_state.qa_history if q["type"] == "voice"]
            tech_avg = (sum(tech_scores) / len(tech_scores)) if tech_scores else 0.0
            hr_avg = (sum(hr_scores) / len(hr_scores)) if hr_scores else 0.0
            voice_avg = (sum(voice_scores) / len(voice_scores)) if voice_scores else 0.0
            overall_avg = (tech_avg + hr_avg + voice_avg) / 3.0
            # Rule-based decision
            if tech_avg >= 3.0 and hr_avg >= 2.5 and voice_avg >= 2.5 and overall_avg >= 3.2:
                decision = "Eligible"
            elif (tech_avg >= 2.8 and hr_avg >= 2.3 and voice_avg >= 2.3 and overall_avg >= 3.0):
                decision = "Maybe (Further interview recommended)"
            else:
                decision = "Not eligible"
            st.session_state.report_summary = {
                "tech_avg": tech_avg, "hr_avg": hr_avg, "voice_avg": voice_avg, "overall_avg": overall_avg, "decision": decision
            }
            st.session_state.stage = "done"
            safe_rerun()

# -------------------------
# DONE: show report, download PDF
# -------------------------
elif st.session_state.stage == "done":
    st.header("Interview Completed — Report & Eligibility")
    summary = st.session_state.get("report_summary", {})
    tech_avg = summary.get("tech_avg", 0.0)
    hr_avg = summary.get("hr_avg", 0.0)
    voice_avg = summary.get("voice_avg", 0.0)
    overall_avg = summary.get("overall_avg", 0.0)
    decision = summary.get("decision", "Not available")

    st.metric("Technical average (0-5)", round(tech_avg, 2))
    st.metric("HR average (0-5)", round(hr_avg, 2))
    st.metric("Voice average (0-5)", round(voice_avg, 2))
    st.metric("Overall average (0-5)", round(overall_avg, 2))
    st.markdown(f"### Eligibility decision: **{decision}**")

    st.markdown("#### Candidate details")
    cand = st.session_state.get("candidate", {})
    st.write(cand)
    st.markdown("#### Resume summary")
    st.write(st.session_state.get("resume_summary", ""))

    st.markdown("#### Detailed question-by-question results")
    for i, qa in enumerate(st.session_state.qa_history, 1):
        st.markdown(f"**{i}. ({qa['type']}) {qa['question']}**")
        st.write(f"Answer: {qa['answer']}")
        st.write(f"Score: {qa['score']:.2f}")
        st.write(f"Remarks: {qa['remarks']}")
        st.markdown("---")

    # Generate PDF bytes
    try:
        pdf_bytes = generate_report_pdf(
            st.session_state.candidate,
            st.session_state.skills,
            st.session_state.qa_history,
            st.session_state.role,
            st.session_state.resume_summary,
            {"technical": tech_avg, "hr": hr_avg, "voice": voice_avg},
            overall_avg,
            decision
        )
    except Exception as e:
        st.warning(f"Could not generate PDF: {e}")
        pdf_bytes = None

    if pdf_bytes:
        st.download_button("Download Candidate Report (PDF)", data=pdf_bytes, file_name=f"report_{cand.get('name','candidate')}.pdf", mime="application/pdf")
    else:
        st.info("PDF generation not available (fpdf missing or error).")

    if st.button("Restart Interview"):
        keys = ["stage","candidate","resume_text","resume_summary","skills","resume_years","tech_questions","hr_questions","qa_history","current_q_index","voice_transcripts","voice_answers","report_summary","role"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.stage = "start"
        safe_rerun()
