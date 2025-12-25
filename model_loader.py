import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_codet5(checkpoint: str = "Salesforce/codet5p-770m-py"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    dtype=torch.float16 if device == "cuda" else None,
    ).to(device)

    model.eval()
    return tokenizer, model, device
