# app.py - Interview Agent with Rule-based Eligibility (Option C)
# Put this file in your project root and run with: streamlit run app.py

import os
from pathlib import Path
from datetime import datetime
import streamlit as st
import json

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Interview Agent", layout="wide")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data_interview"
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR = "vectorstore"
MOTION_FLAG = BASE / "motion.flag"

# -------------------- TITLE --------------------
st.title("🤖 Interview Agent — voice Q&A, resume rating")

# -------------------- SESSION STATE --------------------
if "stage" not in st.session_state:
    st.session_state.stage = "start"
if "candidate" not in st.session_state:
    st.session_state.candidate = {}
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "skills" not in st.session_state:
    st.session_state.skills = []
if "tech_questions" not in st.session_state:
    st.session_state.tech_questions = []
if "hr_questions" not in st.session_state:
    st.session_state.hr_questions = []
if "voice_questions" not in st.session_state:
    st.session_state.voice_questions = [
        "Please briefly introduce yourself and highlight one project you're proud of.",
        "Explain a technical concept from your resume in simple terms.",
        "Why do you want this role and how will you contribute in the first 3 months?"
    ]
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # list of dicts {question,type,answer,score,remarks,category}
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "voice_answers" not in st.session_state:
    st.session_state.voice_answers = [None] * len(st.session_state.voice_questions)
if "voice_transcripts" not in st.session_state:
    st.session_state.voice_transcripts = [None] * len(st.session_state.voice_questions)
if "resume_years" not in st.session_state:
    st.session_state.resume_years = 0.0

# -------------------- Gemini helper (robust) --------------------
def _import_genai():
    try:
        from google import genai  # type: ignore
        return genai
    except Exception:
        try:
            import google.generativeai as genai  # type: ignore
            return genai
        except Exception as e:
            raise ImportError("Could not import Gemini (genai) client. Install `google-generativeai` or correct package.") from e

def gemini_generate(prompt: str, model: str = "gemini-2.0-flash") -> str:
    genai = _import_genai()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    # Try common patterns
    try:
        # pattern: google.generativeai usage
        if hasattr(genai, "configure"):
            try:
                genai.configure(api_key=key)
            except Exception:
                pass
            if hasattr(genai, "models") and hasattr(genai.models, "generate_content"):
                resp = genai.models.generate_content(model=model, contents=prompt)
                txt = getattr(resp, "text", None) or getattr(resp, "output_text", None)
                if txt:
                    return txt
        # pattern: client object
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=key)
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                resp = client.models.generate_content(model=model, contents=prompt)
                txt = getattr(resp, "text", None) or getattr(resp, "output_text", None)
                if txt:
                    return txt
            if hasattr(client, "generate_text"):
                resp = client.generate_text(model=model, prompt=prompt)
                txt = getattr(resp, "text", None) or str(resp)
                if txt:
                    return txt
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}") from e
    raise RuntimeError("Gemini client exists but no supported generate method found. Update gemini_generate() to match your SDK.")

def transcribe_with_gemini(audio_path: str, model: str = "gemini-2.0-flash") -> str:
    genai = _import_genai()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment.")
    # Try several plausible SDK patterns for audio transcription
    try:
        if hasattr(genai, "audio") and hasattr(genai.audio, "transcribe"):
            with open(audio_path, "rb") as af:
                resp = genai.audio.transcribe(model=model, file=af)
            txt = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
            if txt:
                return txt
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=key)
            if hasattr(client, "audio") and hasattr(client.audio, "transcriptions"):
                with open(audio_path, "rb") as af:
                    resp = client.audio.transcriptions.create(file=af, model=model)
                txt = getattr(resp, "text", None) or (resp.get("text") if isinstance(resp, dict) else None)
                if txt:
                    return txt
    except Exception as e:
        raise RuntimeError(f"Gemini transcription failed: {e}") from e
    raise RuntimeError("No supported Gemini transcription method found in installed SDK.")

# -------------------- PDF parsing & basic resume helpers --------------------
def extract_text_from_pdf(fpath: Path) -> str:
    try:
        import pypdf
    except Exception:
        raise RuntimeError("Missing pypdf. Add to requirements.")
    reader = pypdf.PdfReader(str(fpath))
    text = ""
    for page in reader.pages:
        try:
            text += (page.extract_text() or "") + "\n"
        except Exception:
            continue
    return text

def extract_skills_and_summary(text: str):
    skills = []
    common = ["python","java","sql","c++","c#","javascript","react","node","tensorflow","pytorch","pandas","numpy","ml","dl","aws","docker"]
    for s in common:
        if s in text.lower():
            skills.append(s)
    summary = (text[:500] + "...") if text else ""
    return skills, summary

def extract_experience_years(text: str) -> float:
    import re
    m = re.search(r"(\d+)\s+years?", text.lower())
    if m:
        try:
            return float(m.group(1))
        except:
            return 0.0
    return 0.0

# -------------------- Question generation (force 8 tech + 4 hr) --------------------
DEFAULT_TECH_QS = [
    "Explain a project where you applied your core technical skill end-to-end.",
    "Describe the architecture of a system you built and the trade-offs you made.",
    "How do you approach debugging a production incident?",
    "Explain the difference between synchronous and asynchronous programming and when to use each.",
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

def generate_technical_questions(resume_text, role, count=8):
    # Attempt to use Gemini to produce count questions; otherwise fallback to DEFAULT_TECH_QS
    prompt = f"Generate {count} concise technical interview questions (one per line) for a candidate applying for '{role}'. Resume: {resume_text}"
    try:
        out = gemini_generate(prompt)
        qs = [l.strip() for l in out.splitlines() if l.strip()]
        if len(qs) >= count:
            return qs[:count]
    except Exception:
        pass
    return DEFAULT_TECH_QS[:count]

def generate_hr_questions(count=4):
    prompt = f"Generate {count} concise HR interview questions (one per line)."
    try:
        out = gemini_generate(prompt)
        qs = [l.strip() for l in out.splitlines() if l.strip()]
        if len(qs) >= count:
            return qs[:count]
    except Exception:
        pass
    return DEFAULT_HR_QS[:count]

# -------------------- Answer evaluation --------------------
def evaluate_answer_with_llm(q, ans, qtype, role):
    # Ask Gemini to produce a JSON: {"score": <0-5>, "remarks": "..."}
    prompt = (
        "You are an interviewer assistant. Given the question, candidate answer, role and question type, "
        "return ONLY a JSON object like {\"score\": <number 0-5>, \"remarks\": \"short remarks\"}.\n\n"
        f"Question: {q}\nRole: {role}\nType: {qtype}\nCandidate Answer: {ans}\n\n"
        "Score fairly and concisely."
    )
    try:
        txt = gemini_generate(prompt)
        # extract JSON
        try:
            parsed = json.loads(txt)
        except Exception:
            import re
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else None
        if parsed and isinstance(parsed, dict) and "score" in parsed:
            score = float(parsed.get("score", 3))
            remarks = str(parsed.get("remarks", "")).strip()
            score = max(0.0, min(5.0, score))
            return score, remarks
    except Exception:
        pass
    # Heuristic fallback
    length_score = min(5, max(0, len(ans.split()) // 30))
    skill_hits = sum(1 for s in extract_skills_and_summary(ans)[0] if s.lower() in ans.lower())
    score = min(5, length_score + skill_hits)
    return float(score or 3), "Fallback heuristic scoring."

# -------------------- Report PDF generator --------------------
def generate_report_pdf(candidate, skills, qa_history, role, eligibility_decision, category_avgs, overall_score):
    try:
        from fpdf import FPDF
    except Exception:
        raise RuntimeError("Missing fpdf. Add to requirements.")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Interview Report - {candidate.get('name','')}", ln=True)
    pdf.cell(0, 8, f"Role: {role}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.utcnow().isoformat()}", ln=True)
    pdf.cell(0, 8, "", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Summary Scores", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Technical average: {category_avgs.get('technical',0):.2f}", ln=True)
    pdf.cell(0, 8, f"HR average: {category_avgs.get('hr',0):.2f}", ln=True)
    pdf.cell(0, 8, f"Voice average: {category_avgs.get('voice',0):.2f}", ln=True)
    pdf.cell(0, 8, f"Overall average: {overall_score:.2f}", ln=True)
    pdf.cell(0, 8, f"Eligibility: {eligibility_decision}", ln=True)
    pdf.cell(0, 8, "", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Skills", ln=True)
    pdf.set_font("Arial", size=11)
    for s in skills:
        pdf.cell(0, 8, f"- {s}", ln=True)
    pdf.cell(0, 8, "", ln=True)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Question Details", ln=True)
    pdf.set_font("Arial", size=10)
    for i, qa in enumerate(qa_history, 1):
        pdf.multi_cell(0, 7, f"{i}. [{qa['type'].upper()}] {qa['question']}")
        pdf.multi_cell(0, 7, f"Answer: {qa['answer']}")
        pdf.cell(0, 6, f"Score: {qa['score']:.2f}  Remarks: {qa['remarks']}", ln=True)
        pdf.cell(0, 4, "", ln=True)
    return pdf.output(dest="S").encode("latin-1")

# -------------------- START SCREEN --------------------
if st.session_state.stage == "start":
    st.markdown(
        """
        <div style='text-align:center; padding:40px; border-radius:12px; background:linear-gradient(180deg,#f8fafc,#eef2ff)'>
            <h1 style='font-size:40px;'>🤖 AI Interview Agent</h1>
            <p>Upload resume → Answer 12 questions (8 tech + 4 HR) → Provide 3 voice responses → Get a final report and eligibility decision.</p>
            <button onclick="document.getElementById('start_btn').click()" style="background:#5b6cff;color:white;padding:12px 24px;border-radius:8px;border:none;font-weight:700;cursor:pointer;">🚀 Start Interview</button>
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("Start Interview", key="start_btn"):
        st.session_state.stage = "collect_info"
        st.experimental_rerun()
    st.stop()

# -------------------- SIDEBAR KEYS --------------------
st.sidebar.header("API Keys & Settings")
gk = st.sidebar.text_input("Google (Gemini) API key", type="password")
if gk:
    os.environ["GOOGLE_API_KEY"] = gk
    st.sidebar.success("Google key set for session.")
ok = st.sidebar.text_input("OpenAI API key (optional - Whisper)", type="password")
if ok:
    os.environ["OPENAI_API_KEY"] = ok
    st.sidebar.success("OpenAI key set for session.")

# -------------------- STAGE 1: Candidate Info --------------------
if st.session_state.stage == "collect_info":
    st.header("Step 1 — Candidate Info")
    with st.form("info"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        role = st.text_input("Role applying for")
        submit = st.form_submit_button("Next: Upload Resume")
    if submit:
        if not name or not email or not role:
            st.warning("Please fill name, email and role.")
        else:
            st.session_state.candidate = {"name": name, "email": email, "phone": phone, "applied_role": role, "timestamp": datetime.utcnow().isoformat()}
            st.session_state.role = role
            st.session_state.stage = "upload_resume"
            st.experimental_rerun()

# -------------------- STAGE 2: Upload Resume --------------------
elif st.session_state.stage == "upload_resume":
    st.header("Step 2 — Upload Resume (PDF)")
    uploaded = st.file_uploader("Upload PDF resume", type=["pdf"])
    if uploaded:
        dest = DATA_DIR / f"resume_{st.session_state.candidate.get('name','candidate').replace(' ','_')}.pdf"
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        try:
            text = extract_text_from_pdf(dest)
        except Exception as e:
            st.warning(f"Resume parsing issue: {e}")
            text = ""
        st.session_state.resume_text = text
        skills, summary = extract_skills_and_summary(text)
        st.session_state.skills = skills
        st.session_state.resume_years = extract_experience_years(text)
        st.success("Resume parsed.")
        st.session_state.stage = "generate_questions"
        st.experimental_rerun()
    st.write("Tip: paste resume text in the box below if upload not working.")
    txt = st.text_area("Paste resume text (optional)", height=200)
    if txt and st.button("Use pasted text"):
        st.session_state.resume_text = txt
        skills, summary = extract_skills_and_summary(txt)
        st.session_state.skills = skills
        st.session_state.resume_years = extract_experience_years(txt)
        st.session_state.stage = "generate_questions"
        st.experimental_rerun()
    st.stop()

# -------------------- STAGE 3: Generate Questions --------------------
elif st.session_state.stage == "generate_questions":
    st.header("Step 3 — Generating Questions")
    st.write("Generating questions (Gemini)...")
    try:
        st.session_state.tech_questions = generate_technical_questions(st.session_state.resume_text, st.session_state.role, count=8)
        st.session_state.hr_questions = generate_hr_questions(count=4)
    except Exception as e:
        st.warning(f"Generation failed: {e}. Using defaults.")
        st.session_state.tech_questions = DEFAULT_TECH_QS[:8]
        st.session_state.hr_questions = DEFAULT_HR_QS[:4]
    # ensure lengths
    if len(st.session_state.tech_questions) < 8:
        st.session_state.tech_questions = (st.session_state.tech_questions + DEFAULT_TECH_QS)[:8]
    if len(st.session_state.hr_questions) < 4:
        st.session_state.hr_questions = (st.session_state.hr_questions + DEFAULT_HR_QS)[:4]
    st.session_state.current_q_index = 0
    st.session_state.stage = "qna"
    st.experimental_rerun()

# -------------------- STAGE 4: Q&A --------------------
elif st.session_state.stage == "qna":
    st.header("Step 4 — Interview (text & voice answers)")
    skill_score = min(6, len(st.session_state.skills))
    yrs = float(st.session_state.get("resume_years", 0) or 0)
    exp_score = min(4, int(min(4, yrs)))
    resume_score = round((skill_score + exp_score), 1)
    st.metric("Resume quick score (0-10 approx)", resume_score)

    all_qs = [{"q": q, "type": "technical"} for q in st.session_state.tech_questions] + [{"q": q, "type": "hr"} for q in st.session_state.hr_questions]
    idx = st.session_state.current_q_index
    total = len(all_qs) + len(st.session_state.voice_questions)
    st.write(f"Question {idx+1} / {total}")

    # TEXT QUESTIONS
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
                st.experimental_rerun()
    else:
        # VOICE QUESTIONS
        v_idx = idx - len(all_qs)
        if v_idx < len(st.session_state.voice_questions):
            st.subheader(f"Q (voice) — {v_idx+1} of {len(st.session_state.voice_questions)}")
            st.write(st.session_state.voice_questions[v_idx])
            st.markdown("Record locally and upload audio file (wav/m4a/mp3). Gemini transcription attempted; else OpenAI Whisper fallback if configured; else manual.")
            uploaded_audio = st.file_uploader(f"Upload audio answer for voice Q{v_idx+1}", type=["wav","m4a","mp3","ogg"], key=f"audio_{v_idx}")
            if uploaded_audio:
                tmp_path = DATA_DIR / f"voice_q{v_idx+1}_{st.session_state.candidate.get('name','candidate')}.wav"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())
                st.success("Audio saved. Attempting transcription...")
                transcript = None
                try:
                    transcript = transcribe_with_gemini(str(tmp_path))
                    st.success("Gemini transcription succeeded.")
                except Exception as ge:
                    st.warning(f"Gemini transcription failed: {ge}")
                    if os.getenv("OPENAI_API_KEY"):
                        try:
                            import openai
                            openai.api_key = os.getenv("OPENAI_API_KEY")
                            with open(tmp_path, "rb") as af:
                                resp = openai.Audio.transcribe("whisper-1", af)
                            transcript = resp.get("text", "").strip()
                            st.success("Whisper transcription succeeded.")
                        except Exception as oe:
                            st.warning(f"Whisper transcription failed: {oe}")
                    else:
                        st.info("No OpenAI key — please paste transcript manually if needed.")
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
                        st.experimental_rerun()
                # Manual fallback entry always available
                manual = st.text_area("Paste transcript of your audio here (fallback)", key=f"manual_trans_{v_idx}", height=150)
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
                        st.experimental_rerun()
        else:
            # all answered -> compute report & eligibility
            st.success("All questions answered. Computing report & eligibility...")
            # compute averages by category
            tech_scores = [q["score"] for q in st.session_state.qa_history if q["type"] == "technical"]
            hr_scores = [q["score"] for q in st.session_state.qa_history if q["type"] == "hr"]
            voice_scores = [q["score"] for q in st.session_state.qa_history if q["type"] == "voice"]
            tech_avg = (sum(tech_scores) / len(tech_scores)) if tech_scores else 0.0
            hr_avg = (sum(hr_scores) / len(hr_scores)) if hr_scores else 0.0
            voice_avg = (sum(voice_scores) / len(voice_scores)) if voice_scores else 0.0
            # overall average of category averages
            overall_avg = (tech_avg + hr_avg + voice_avg) / 3.0
            # Apply Option C rules:
            eligible = False
            if tech_avg >= 3.0 and hr_avg >= 2.5 and voice_avg >= 2.5 and overall_avg >= 3.2:
                eligible = True
                decision = "Eligible"
            else:
                # borderline check
                if (tech_avg >= 2.8 and hr_avg >= 2.3 and voice_avg >= 2.3 and overall_avg >= 3.0):
                    decision = "Maybe (Further interview recommended)"
                else:
                    decision = "Not eligible"
            # store summary in session_state for the done page
            st.session_state.report_summary = {
                "tech_avg": tech_avg,
                "hr_avg": hr_avg,
                "voice_avg": voice_avg,
                "overall_avg": overall_avg,
                "decision": decision,
                "eligible": eligible
            }
            st.session_state.stage = "done"
            st.experimental_rerun()

    # Sidebar progress
    st.sidebar.header("Progress")
    st.sidebar.write(f"Answered: {len(st.session_state.qa_history)} / {total}")
    st.sidebar.write(f"Remaining (approx): {total - len(st.session_state.qa_history)}")

# -------------------- STAGE 5: DONE & REPORT --------------------
elif st.session_state.stage == "done":
    st.header("Interview Completed — Report & Eligibility")

    summary = st.session_state.get("report_summary", {})
    tech_avg = summary.get("tech_avg", 0.0)
    hr_avg = summary.get("hr_avg", 0.0)
    voice_avg = summary.get("voice_avg", 0.0)
    overall_avg = summary.get("overall_avg", 0.0)
    decision = summary.get("decision", "Not available")

    st.metric("Technical average (0-5)", round(tech_avg,2))
    st.metric("HR average (0-5)", round(hr_avg,2))
    st.metric("Voice average (0-5)", round(voice_avg,2))
    st.metric("Overall average (0-5)", round(overall_avg,2))
    st.markdown(f"### Eligibility decision: **{decision}**")

    st.markdown("#### Detailed question-by-question results")
    for i, qa in enumerate(st.session_state.qa_history, 1):
        st.markdown(f"**{i}. ({qa['type']}) {qa['question']}**")
        st.write(f"Answer: {qa['answer']}")
        st.write(f"Score: {qa['score']:.2f}")
        st.write(f"Remarks: {qa['remarks']}")
        st.markdown("---")

    # Downloadable PDF with full results
    try:
        pdf_bytes = generate_report_pdf(
            st.session_state.candidate,
            st.session_state.skills,
            st.session_state.qa_history,
            st.session_state.role,
            decision,
            {"technical": tech_avg, "hr": hr_avg, "voice": voice_avg},
            overall_avg
        )
    except Exception as e:
        st.warning(f"Could not generate PDF: {e}")
        pdf_bytes = b"%PDF-1.4\n%placeholder\n"

    st.download_button("Download Candidate Report (PDF)", data=pdf_bytes, file_name=f"report_{st.session_state.candidate.get('name','candidate')}.pdf", mime="application/pdf")

    if st.button("Restart Interview"):
        # clear tracked keys
        keys = ["stage","candidate","resume_text","skills","tech_questions","hr_questions","qa_history","current_q_index","role","voice_questions","voice_answers","voice_transcripts","resume_years","report_summary"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state.stage = "start"
        st.experimental_rerun()
