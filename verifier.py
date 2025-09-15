# verifier.py
from typing import List, Dict, Tuple
from generator import run_ollama_mistral, build_verification_prompt, FALLBACK_TEXT

def verify_answer(query: str, retrieved: List[Dict], answer: str) -> Tuple[bool, str]:
    """
    Ask the model whether its proposed answer relies only on the retrieved context.
    Returns (verified_boolean, final_answer). If verification fails or returns NO, final_answer is FALLBACK_TEXT.
    """
    if not retrieved:
        return False, FALLBACK_TEXT
    ver_prompt = build_verification_prompt(query, retrieved, answer)
    try:
        out = run_ollama_mistral(ver_prompt)
    except Exception as e:
        print(f"[verifier] verification failed: {e}")
        return False, FALLBACK_TEXT
    text = out.strip().upper()
    print(f"[verifier] verification output: {text[:200]!r}")
    if "YES" in text:
        return True, answer
    else:
        return False, FALLBACK_TEXT
