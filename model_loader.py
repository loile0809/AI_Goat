import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_codet5(checkpoint: str = "Salesforce/codet5p-770m-py"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    print(f"DEBUG: torch version={torch.__version__} cuda_available={torch.cuda.is_available()}")
    import sys
    print(f"DEBUG: Python Executable: {sys.executable}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG: Selected device={device}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16 if device == "cuda" else None,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    return tokenizer, model, device
