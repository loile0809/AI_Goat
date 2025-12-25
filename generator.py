import re
import torch

BAD_PATTERNS = [
    r"\bprint\s*\(",
    r"\binput\s*\(",
    r"\bopen\s*\(",
    r"^\s*import\s+",
    r"^\s*from\s+\w+\s+import\s+",
    r"\b__import__\b",
    r"\bos\.system\b",
    r"\bsubprocess\b",
    r"\byield\b",           # no generator
    r"\bsum\s*\(\s*\[",     # sum([ ... ]) -> waste memory, prefer sum(...)
]



def _strip_fences(s: str) -> str:
    return (s or "").replace("```python", "").replace("```", "")

def _extract_body(text: str) -> str:
    text = _strip_fences(text)

    if "if __name__" in text:
        text = text.split("if __name__")[0]

    # cut after first def header if present
    m = re.search(r"def\s+\w+\s*\(.*?\)\s*:\s*\n", text)
    if m:
        text = text[m.end():]

    # stop before next top-level def/class
    text = re.split(r"\n(?=(def|class)\s+\w+)", text)[0]

    # drop leading docstring echoes
    text = re.sub(r'^\s*""".*?"""\s*', "", text, flags=re.DOTALL)

    # trim blank lines
    lines = text.splitlines()
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()

    return "\n".join(lines) if lines else "pass"

def _is_bad(body: str) -> bool:
    for p in BAD_PATTERNS:
        if re.search(p, body, flags=re.MULTILINE):
            return True
    return False

@torch.inference_mode()
def generate_function_bodies(
    tokenizer,
    model,
    device,
    code_prefix: str,
    n: int = 16,
    max_new_tokens: int = 160,
    temperature: float = 0.9,
    top_p: float = 0.95,
    tries: int = 3,
):
    if not code_prefix.endswith("\n"):
        code_prefix += "\n"

    candidates = []
    for _ in range(tries):
        enc = tokenizer(code_prefix, return_tensors="pt", truncation=True, max_length=256).to(device)
        outs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            repetition_penalty=1.1,
        )
        for o in outs:
            txt = tokenizer.decode(o, skip_special_tokens=True)
            body = _extract_body(txt)
            if not _is_bad(body):
                candidates.append(body)

    uniq, seen = [], set()
    for c in candidates:
        k = c.strip()
        if k and k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq
