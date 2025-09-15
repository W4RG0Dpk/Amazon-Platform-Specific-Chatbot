# generator.py
import subprocess
from typing import List, Dict

# Ollama command (local)
OLLAMA_CMD = ["ollama", "run", "mistral"]

# Exact fallback text used across the app
FALLBACK_TEXT = "Sorry, I cannot answer that from the provided document. Would you like to contact support?"

def run_ollama_mistral(prompt: str, timeout: int = 60) -> str:
    """
    Run local Ollama Mistral and return stdout text.
    Raises RuntimeError on non-zero exit or timeout.
    """
    try:
        res = subprocess.run(OLLAMA_CMD, input=prompt, text=True, capture_output=True, encoding='utf-8', timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ollama timed out: {e}")
    if res.returncode != 0:
        raise RuntimeError(f"ollama error (code {res.returncode}):\n{res.stderr}")
    return res.stdout.strip()

def detect_greeting(text: str) -> bool:
    greetings=["hello", "hi", "greetings", "good morning", 
               "good afternoon", "good evening", "hey",
               "how are you doing","yo","are you fine", "howdy","what's up"]
    return any(greet in text.lower() for greet in greetings)
def build_generation_prompt(query: str, retrieved: List[Dict]) -> str:
    ctx = "\n".join([f"[{r['line_no']}] {r['text']}" for r in retrieved]) if retrieved else ""

    greeting_note = ""
    if detect_greeting(query):
        greeting_note = "The user greeted you. First reply with a warm and short greeting (like 'Hi there!'), " \
                        "then continue answering the question using the context below."

    if not retrieved:
        return (
            f"User question: {query}\n\n"
            "This question cannot be answered from the provided knowledge base. "
            f"Reply exactly: \"{FALLBACK_TEXT}\""
        )

    prompt = f"""
You are a professional Amazon Help Assistant. Follow these rules strictly:
{greeting_note}
1) Use ONLY the information in the CONTEXT block below. Do NOT invent facts.
2) If the CONTEXT does NOT contain the answer, respond exactly:
   "{FALLBACK_TEXT}"
3) Provide step-by-step actionable directions that start from the Amazon homepage.
4) When referencing context, cite the line numbers in square brackets (e.g., [23]).
5) Be concise, professional, and polite.


CONTEXT:
{ctx}

User Question:
{query}

Answer:
"""
    return prompt.strip()


def build_verification_prompt(query: str, retrieved: List[Dict], answer: str) -> str:
    """
    Build a verification prompt that asks the model to answer YES or NO if the answer
    strictly relies on the provided context.
    """
    ctx = "\n".join([f"[{r['line_no']}] {r['text']}" for r in retrieved])
    verification = f"""
CONTEXT:
{ctx}

User Question:
{query}

Proposed Answer:
{answer}

Task:
Based only on the CONTEXT above, does the Proposed Answer rely ONLY on the provided CONTEXT (no external facts or assumptions)?
Answer with a single word: YES or NO.
"""
    return verification.strip()
