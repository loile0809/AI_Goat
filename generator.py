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
    r"\byield\b",           # no generator
    r"^\s*//",              # C-style comments
    r"@interface",          # Obj-C
    r"#import\s+<",         # C/Obj-C include
    r"#include\s+<",        # C include
]



def _strip_fences(s: str) -> str:
    return (s or "").replace("```python", "").replace("```", "")

def _extract_body(text: str) -> str:
    text = _strip_fences(text)

    # If the model repeats the signature (rare in Seq2Seq but possible), cut it
    m = re.search(r"def\s+\w+\s*\(.*?\)\s*:\s*\n", text)
    if m:
        text = text[m.end():]

    # Cut at common stop boundaries
    # Stop at next function/class definition
    text = re.split(r"\n\s*(def|class|@)\s+\w+", text)[0]
    
    # Stop at if __name__ == "__main__":
    text = text.split('if __name__')[0]

    # Stop at common garbage lines seen in CodeT5 benchmarks
    garbage_markers = [
        "\n# pylint:", "\n# pragma:", "\n# NOQA", 
        "\ndef test_", "\nclass Test", 
        "\nprint(", "\nassert ",
    ]
    for marker in garbage_markers:
        if marker in text:
            text = text.split(marker)[0]

    # drop leading docstring echoes
    text = re.sub(r'^\s*""".*?"""\s*', "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*'''.*?'''\s*", "", text, flags=re.DOTALL)

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
    max_new_tokens: int = 200,
    temperature: float = 0.65, # Lower temp for stability
    top_p: float = 0.95,
    tries: int = 3,
):
    if not code_prefix.endswith("\n"):
        code_prefix += "\n"

    # CodeT5 is a Seq2Seq model. It generates the target sequence (body) directly.
    # However, sometimes it generates garbage if not constrained.
    
    candidates = []
    for _ in range(tries):
        enc = tokenizer(code_prefix, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        # 770M model fits in memory, no batching needed
        outs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        for o in outs:
            txt = tokenizer.decode(o, skip_special_tokens=True)
            body = _extract_body(txt)
            if not _is_bad(body) and body.strip():
                candidates.append(body)

    uniq, seen = [], set()
    for c in candidates:
        k = c.strip()
        if k and k not in seen:
            uniq.append(c)
            seen.add(k)
    return uniq
