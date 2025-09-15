# main.py
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from retrieval import search
from generator import build_generation_prompt, run_ollama_mistral, FALLBACK_TEXT
from verifier import verify_answer
from ingest import ingest_file  # optional re-ingest if you wire an upload endpoint

APP = FastAPI()
ROOT = Path(__file__).parent

# Create support folders
SUPPORT_DIR = ROOT / "support"
FORMS_DIR = SUPPORT_DIR / "forms"
FEEDBACK_DIR = SUPPORT_DIR / "feedback"
FORMS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

# Modern Chat UI (HTML/CSS/JS)
CHAT_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Amazon Platform Chatbot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#f4f6f8;
      --card:#ffffff;
      --primary:#0b78e3;
      --muted:#6b7280;
      --glass: rgba(255,255,255,0.6);
    }
    html,body{
      height:100%;
      margin:0;
      font-family:Inter,Segoe UI,Roboto,Arial,sans-serif;
      background:var(--bg);
      color:#0f1724;
    }

    /* Header */
    header{
      position:fixed;
      top:0;left:0;right:0;
      background:linear-gradient(90deg,#052c5b 0%, #083f7a 100%);
      color:white;
      padding:16px 24px;
      box-shadow:0 6px 18px rgba(3,7,18,0.08);
      z-index:60;
      display:flex;
      align-items:center;
      justify-content:space-between;
    }
    header h1{font-size:18px;margin:0;font-weight:700}
    header .sub{font-size:13px;opacity:.9}

    /* Page layout: slim left sidebar + center chat */
    .page{
      display:grid;
      grid-template-columns:220px 1fr;
      gap:20px;
      padding:92px 20px 20px; /* account for fixed header */
      max-width:1200px;
      margin:0 auto;
      height:calc(100vh - 92px);
      box-sizing:border-box;
    }

    /* Sidebar (left) */
    .sidebar{
      background:var(--card);
      border-radius:12px;
      box-shadow:0 8px 30px rgba(10,15,25,0.06);
      padding:18px;
      display:flex;
      flex-direction:column;
      gap:14px;
      position:sticky;
      top:92px;
      align-self:start;
    }
    .sidebar h2{font-size:16px;margin:0;color:#083f7a;font-weight:700}
    .sidebar p{margin:0;color:var(--muted);font-size:13px}
    .support-btn{
      margin-top:6px;
      width:100%;
      padding:10px 12px;
      border-radius:10px;
      border:none;
      background:linear-gradient(90deg,#0b78e3,#0266c8);
      color:white;
      cursor:pointer;
      font-weight:600;
      box-shadow:0 8px 20px rgba(11,120,227,0.14);
    }
    .support-btn:hover{transform:translateY(-2px)}

    /* Chat panel */
    .panel{
      background:var(--card);
      border-radius:12px;
      box-shadow:0 8px 30px rgba(10,15,25,0.06);
      overflow:hidden;
      display:flex;
      flex-direction:column;
      min-height:0;
    }
    .chat-area{display:flex;flex-direction:column;min-height:0}

    /* Only messages area scrolls */
    .messages{
      padding:18px;
      flex:1;
      overflow-y:auto;
      display:flex;
      flex-direction:column;
      gap:12px;
      background:linear-gradient(180deg, rgba(255,255,255,0.6), rgba(245,247,250,0.6));
      min-height:0;
    }
    .msg{
      max-width:78%;
      padding:12px 14px;
      border-radius:12px;
      box-shadow:0 2px 8px rgba(10,15,25,0.04);
      line-height:1.4;
      white-space:pre-wrap;
      word-wrap:break-word;
    }
    .user{align-self:flex-end;background:linear-gradient(180deg,var(--primary),#0266c8);color:white;}
    .assistant{align-self:flex-start;background:linear-gradient(180deg,#fff,#fbfdff);border:1px solid #eef3fb;color:#08263a}
    .sources{font-size:12px;color:var(--muted);margin-top:6px}

    /* Composer */
    .composer{
      display:flex;gap:10px;padding:12px;border-top:1px solid #eef3f6;align-items:center;
      background:linear-gradient(180deg, rgba(255,255,255,0.6), rgba(250,251,253,0.6));
    }
    .input{flex:1;display:flex;gap:10px;align-items:center}
    .input input{
      flex:1;padding:12px 14px;border-radius:10px;border:1px solid #e6eef8;background:transparent;outline:none;
      box-shadow:none;font-size:14px;
    }
    .btn{background:var(--primary);color:white;padding:10px 14px;border-radius:10px;border:none;cursor:pointer;transition:transform .12s ease,box-shadow .12s}
    .btn:hover{transform:translateY(-2px);box-shadow:0 8px 20px rgba(11,120,227,0.16)}

    /* Feedback bar */
    .feedback-bar{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;border:1px solid #e6eef8;background:#fff;box-shadow:0 2px 8px rgba(4,10,22,0.03);margin:12px}
    .stars{display:flex;gap:6px}
    .star{width:28px;height:28px;border-radius:6px;background:#f1f5f9;display:inline-flex;align-items:center;justify-content:center;cursor:pointer;transition:transform .12s}
    .star:hover{transform:translateY(-4px)}
    .star.active{background:linear-gradient(90deg,#ffd166,#ff9f1c);color:#07203b;font-weight:700}

    /* Modal overlay */
    .overlay{position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(2,6,23,0.45);display:none;align-items:center;justify-content:center;z-index:80}
    .modal{width:560px;background:var(--card);border-radius:12px;padding:18px;box-shadow:0 24px 80px rgba(3,7,18,0.45)}
    textarea{width:100%;height:120px;padding:10px;border-radius:8px;border:1px solid #e6eef8}
    .small{font-size:13px;color:var(--muted)}

    /* small responsive */
    @media (max-width:980px){
      .page{grid-template-columns:1fr; padding:120px 12px; height:calc(100vh - 120px)}
      .sidebar{display:none}
    }
  </style>
</head>
<body>
  <header>
    <div style="display:flex;flex-direction:column">
      <h1>Amazon Platform Chatbot</h1>
      <div class="sub">Local RAG • Mistral (local) — Instructions start from the Amazon homepage</div>
    </div>
    <div style="display:flex;gap:12px;align-items:center">
      <div style="font-size:13px;color:rgba(255,255,255,0.95)">v1 — Local Demo</div>
    </div>
  </header>

  <div class="page">
    <!-- LEFT SIDEBAR -->
    <div class="sidebar">
      <h2>Assistant</h2>
      <p class="small">Your AI guide for buyer & seller flows on the Amazon homepage. Ask practical step-by-step questions.</p>

      <button class="support-btn" id="supportBtn">Contact Support</button>

      <div style="margin-top:6px" class="card">
        <div style="font-weight:600;margin-bottom:6px">Quick tips</div>
        <div class="small muted">Try: "Where to find Crocs?", "How to track orders?", "How to list a product?"</div>
      </div>
    </div>

    <!-- CENTER CHAT AREA -->
    <div class="panel chat-area">
      <div class="meta" style="padding:14px 18px;border-bottom:1px solid #f0f3f6;font-weight:600">Ask questions starting from the Amazon homepage</div>

      <!-- messages (only this area scrolls) -->
      <div id="messages" class="messages" aria-live="polite"></div>

      <!-- composer -->
      <div class="composer">
        <div class="input">
          <input id="query" type="text" placeholder="Ask about Amazon (buyers & sellers)..." autocomplete="off"/>
        </div>
        <button id="send" class="btn">Send</button>
      </div>

      <!-- feedback bar (hidden until first non-fallback answer) -->
      <div id="feedbackBar" style="display:none;">
        <div class="feedback-bar" style="margin:12px">
          <div class="muted">Was this answer helpful?</div>
          <div class="stars" id="stars">
            <div class="star" data-value="1">★</div>
            <div class="star" data-value="2">★</div>
            <div class="star" data-value="3">★</div>
            <div class="star" data-value="4">★</div>
            <div class="star" data-value="5">★</div>
          </div>
          <button id="giveFeedbackBtn" class="btn" style="padding:8px 10px">Give feedback</button>
        </div>
      </div>
    </div>
  </div>

  <!-- overlay + modal (used for both feedback & support) -->
  <div id="overlay" class="overlay" role="dialog" aria-modal="true">
    <div class="modal" id="modalContent"></div>
  </div>

<script>
/* Elements */
const messagesEl = document.getElementById("messages");
const queryEl = document.getElementById("query");
const sendBtn = document.getElementById("send");
const feedbackBar = document.getElementById("feedbackBar");
const starsEl = document.getElementById("stars");
const giveFeedbackBtn = document.getElementById("giveFeedbackBtn");
const overlay = document.getElementById("overlay");
const modalContent = document.getElementById("modalContent");
const supportBtn = document.getElementById("supportBtn");

let lastResponseWasFallback = false;
let lastRetrieval = null;
let lastAnswerText = "";
let selectedRating = 0;

/* Helpers */
function appendMessage(role, text, small=false){
  const d = document.createElement("div");
  d.className = "msg " + (role==="user" ? "user":"assistant");
  d.textContent = text;
  messagesEl.appendChild(d);
  if(small){
    d.style.opacity = 0.85;
    d.style.fontSize = "13px";
    d.style.maxWidth = "90%";
  }
  // smooth scroll
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* Stars interaction */
starsEl.addEventListener("click", (e) => {
  const s = e.target.closest(".star");
  if(!s) return;
  selectedRating = Number(s.dataset.value);
  starsEl.querySelectorAll(".star").forEach(st => {
    st.classList.toggle("active", Number(st.dataset.value) <= selectedRating);
  });
});

/* Enter key to send */
queryEl.addEventListener("keydown", (e) => {
  if(e.key === "Enter"){
    e.preventDefault();
    sendBtn.click();
  }
});

/* Send message */
sendBtn.addEventListener("click", async () => {
  const q = queryEl.value.trim();
  if(!q) return;
  appendMessage("user", q);
  queryEl.value = "";
  appendMessage("assistant", "…thinking…");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify({ query: q })
    });
    const j = await res.json();

    // remove thinking placeholder if present
    if(messagesEl.lastChild && messagesEl.lastChild.textContent === "…thinking…"){
      messagesEl.removeChild(messagesEl.lastChild);
    }

    // append answer
    const answerText = j.answer || "No answer returned.";
    appendMessage("assistant", answerText);
    lastAnswerText = answerText;
    lastRetrieval = j.retrieved || [];
    lastResponseWasFallback = !!j.is_ood || (j.answer && j.answer.includes("Sorry, I cannot answer that"));

    // show sources if present
    if(lastRetrieval && lastRetrieval.length){
      const src = lastRetrieval.map(x => `[${x.line_no}]`).join(", ");
      appendMessage("assistant", "Sources: " + src, true);
    }

    // Show feedback bar only if answer is not fallback
    if(!lastResponseWasFallback){
      feedbackBar.style.display = "block";
    } else {
      feedbackBar.style.display = "none";
      // auto open support modal on fallback
      openSupportModal(q);
    }

    console.log("DEBUG /chat:", {is_ood: j.is_ood, verified: j.verified, max_score: j.max_score});
  } catch(err) {
    // remove thinking if present
    if(messagesEl.lastChild && messagesEl.lastChild.textContent === "…thinking…"){
      messagesEl.removeChild(messagesEl.lastChild);
    }
    appendMessage("assistant", "Error: " + String(err));
  }
});

/* Feedback modal flow */
giveFeedbackBtn.addEventListener("click", () => {
  if(selectedRating <= 0){
    alert("Please select a rating (1-5 stars) first.");
    return;
  }
  openFeedbackModal();
});

function openFeedbackModal(){
  overlay.style.display = "flex";
  modalContent.innerHTML = `
    <h3>Feedback on response</h3>
    <div class="small muted" style="margin-bottom:8px">Rating: ${selectedRating} / 5</div>
    <textarea id="fbComments" placeholder="Tell us what went well or what could be improved..."></textarea>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px">
      <button id="fbCancel" class="btn" style="background:#e6eef8;color:#08344a">Cancel</button>
      <button id="fbSubmit" class="btn">Submit feedback</button>
    </div>
  `;
  document.getElementById("fbCancel").onclick = closeModal;
  document.getElementById("fbSubmit").onclick = async () => {
    const comments = document.getElementById("fbComments").value || "";
    const form = new FormData();
    form.append("user", "anonymous");
    form.append("rating", String(selectedRating));
    form.append("comments", comments);
    form.append("answer", lastAnswerText);
    const res = await fetch("/feedback", { method:"POST", body: form });
    const j = await res.json();
    alert("Thanks — feedback saved (ref: " + (j.path || "saved") + ").");
    closeModal();
    feedbackBar.style.display = "none";
    selectedRating = 0;
    starsEl.querySelectorAll(".star").forEach(st => st.classList.remove("active"));
  };
}

/* Support modal: open from left button or on fallback */
supportBtn.addEventListener("click", () => openSupportModal());

function openSupportModal(userQuery = ""){
  overlay.style.display = "flex";
  modalContent.innerHTML = `
    <h3>Contact Support</h3>
    <div class="small muted" style="margin-bottom:8px">We couldn't find an answer in the document. Please describe your issue and we'll save it for the support team.</div>
    <input id="sname" placeholder="Your name (optional)" style="width:100%;padding:10px;border-radius:8px;border:1px solid #e6eef8;margin-bottom:8px"/>
    <input id="semail" placeholder="Email (optional)" style="width:100%;padding:10px;border-radius:8px;border:1px solid #e6eef8;margin-bottom:8px"/>
    <textarea id="smsg" placeholder="Describe your question for support...">${userQuery ? "User query: " + userQuery : ""}</textarea>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:10px">
      <button id="sCancel" class="btn" style="background:#e6eef8;color:#08344a">Cancel</button>
      <button id="sSubmit" class="btn">Send to support</button>
    </div>
  `;
  document.getElementById("sCancel").onclick = closeModal;
  document.getElementById("sSubmit").onclick = async () => {
    const name = document.getElementById("sname").value || "anonymous";
    const email = document.getElementById("semail").value || "";
    const message = document.getElementById("smsg").value || ("User query: " + (userQuery || ""));
    const form = new FormData();
    form.append("name", name);
    form.append("email", email);
    form.append("message", message);
    const res = await fetch("/support", { method:"POST", body: form });
    const j = await res.json();
    alert("Support request saved. Reference: " + (j.path || "saved"));
    closeModal();
  };
}

/* Modal helpers */
function closeModal(){
  overlay.style.display = "none";
  modalContent.innerHTML = "";
}
overlay.addEventListener("click", (e) => { if(e.target === overlay) closeModal(); });

/* Focus helper */
window.addEventListener('load', ()=> setTimeout(()=> { try { queryEl.focus(); } catch(e){} }, 150) );
</script>
</body>
</html>

"""


@APP.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(CHAT_HTML)

@APP.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error":"empty query"}, status_code=400)

    # Retrieval
    retrieved, is_ood, max_score = search(query)
    print(f"[main] /chat retrieved={len(retrieved)} is_ood={is_ood} max_score={max_score:.4f}")

    # If OOD return fallback and let UI open support modal
    if is_ood:
        return {"answer": FALLBACK_TEXT, "is_ood": True, "retrieved": retrieved, "max_score": max_score, "verified": False}

    # Build prompt and generate
    prompt = build_generation_prompt(query, retrieved)
    print(f"[main] Prompt length: {len(prompt)}")
    try:
        gen = run_ollama_mistral(prompt)
    except Exception as e:
        print(f"[main] generator error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    # Verify (strict)
    verified, final = verify_answer(query, retrieved, gen)
    print(f"[main] verification: verified={verified}")
    return {"answer": final, "is_ood": False, "retrieved": retrieved, "verified": verified, "max_score": max_score}

# Feedback endpoint: saves a txt file with rating and comments
@APP.post("/feedback")
async def feedback(user: str = Form(...), rating: int = Form(...), comments: str = Form(None), answer: str = Form(None)):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = FEEDBACK_DIR / f"feedback_{ts}_{uuid.uuid4().hex}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(f"user: {user}\nrating: {rating}\ncomments:\n{comments}\n\nanswer:\n{answer}\n")
    print(f"[main] saved feedback -> {fname}")
    return {"status":"saved", "path": str(fname)}

# Support endpoint: saves a JSON file with user support request
@APP.post("/support")
async def support(name: str = Form(...), email: str = Form(None), message: str = Form(...)):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = FORMS_DIR / f"support_{ts}_{uuid.uuid4().hex}.json"
    data = {"name": name, "email": email, "message": message, "ts": ts}
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[main] saved support -> {fname}")
    return {"status":"saved", "path": str(fname)}

# Optional: upload new context and re-ingest
@APP.post("/upload_context")
async def upload_context(file: UploadFile = File(...)):
    text = (await file.read()).decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    ingest_file(lines)
    return {"status":"ingested", "lines": len(lines)}
