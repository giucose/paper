import json
import os
from os import path
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import csv

from profiles.ai_usage_profile.whodunit.compute_whodunit_metrics import compute_diff_and_store
from profiles.ai_usage_profile_eop.whodunit.compute_whodunit_metrics import get_relevant_extensions

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        qkv = self.Wqkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        scores = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
        
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - mask) * -1e9
            
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.Wo(out)

class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size=2624):
        super().__init__()
        self.Wi = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        a, b = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.silu(a) * b)

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, first_layer=False):
        super().__init__()
        if not first_layer:
            self.attn_norm = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size)
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp     = SwiGLUMLP(hidden_size, intermediate_size=2624)
        self.first_layer = first_layer

    def forward(self, x, attention_mask=None):
        normed = x if self.first_layer else self.attn_norm(x)
        x = x + self.attn(normed, attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class EncoderOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state
        
class TextEncoder(nn.Module): # :white_check_mark: class name restored
    def __init__(self, vocab_size=50368, hidden_size=1024, num_layers=28):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            "tok_embeddings": nn.Embedding(vocab_size, hidden_size),
            "norm": RMSNorm(hidden_size),
        })
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, first_layer=(i == 0))
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings["tok_embeddings"](input_ids)
        x = self.embeddings["norm"](x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        # :white_check_mark: return object with .last_hidden_state to match TLModel's original usage
       
        return EncoderOutput(x)

# ── Top-level model ──────────────────────────────────────────────────────────

class TLModel(nn.Module):
    def __init__(self, hidden_size=1024, projection_dim=256, num_classes=4): # :white_check_mark: 256 not 128
        super().__init__() # :white_check_mark: was missing
        self.text_encoder = TextEncoder(hidden_size=hidden_size)
        self.text_projection = nn.Linear(hidden_size, projection_dim) # :white_check_mark: 1024 not 768
        self.classifier = nn.Linear(projection_dim, num_classes)

    def forward(self, input_ids=None, attention_mask=None):
        encoder_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = encoder_out.last_hidden_state.mean(dim=1) # :white_check_mark: mean pooling (not CLS)
        x = F.relu(self.text_projection(x))  # :white_check_mark: relu matches original
        logits = self.classifier(x)
        return logits


def load_model():
    # 1. Load tokenizer directly - no transformers needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_dir, "droiddetect_tokenizer", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 2. Load model
    model = TLModel()
    weights_path = os.path.join(current_dir, "droiddetect_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    
    return tokenizer, model

def classify_visual(text, tokenizer, model):
    
    id2label = {
        0: "HUMAN_GENERATED",
        1: "MACHINE_GENERATED",
        2: "MACHINE_REFINED",
        3: "MACHINE_GENERATED_ADVERSARIAL",
    }
    
    # Tokenize directly with tokenizers library, no transformers
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids])
    attention_mask = torch.tensor([encoding.attention_mask])

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = F.softmax(logits, dim=-1)[0]
    predicted = probs.argmax().item()
    print(f"Prediction : {id2label[predicted]}")
    print(f"Confidence : {probs[predicted]:.1%}")
    print()
    for i, p in enumerate(probs):
        bar = "█" * int(p * 40)
        print(f"  {id2label[i]:<32} {p:.4f}  {bar}")
      
def classify_numerical(text, tokenizer, model):
    
    # Tokenize directly with tokenizers library, no transformers
    encoding = tokenizer.encode(text)
    input_ids = torch.tensor([encoding.ids])
    attention_mask = torch.tensor([encoding.attention_mask])

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = F.softmax(logits, dim=-1)[0]
    
    return probs

def compute_droiddetect_metrics(
    repo_path,
    repository_id,
    droiddetect_tokenizer,
    droiddetect_model,
):
    path = Path.cwd() / "system_files" / "droiddetect_metrics_eop" / f"{repository_id}_droiddetect_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    headers_droiddetect = [
        "filepath",
        "prob_human_droiddetect", "prob_machine_droiddetect",
        "prob_machine_refined_droiddetect",
        "prob_machine_generated_adversarial_droiddetect",
    ]

    # Load already-processed filepaths if CSV exists
    already_processed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_processed.add(row["filepath"])
        print(f"[INFO] Resuming for {repository_id}: {len(already_processed)} files already processed")

    relevant_extensions = get_relevant_extensions()
    files = [
        f for f in repo_path.rglob("*")
        if f.is_file() and "." + f.name.split(".")[-1].lower() in relevant_extensions
    ]
    num_files = len(files)
    print(f"[INFO] Processing {num_files} files in repo {repository_id} with droiddetect...")

    # Open in append mode; write header only if file is new
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers_droiddetect)

        for i, file in enumerate(files):
            filepath_str = str(file)
            if filepath_str in already_processed:
                print(f"[INFO] Skipping already-processed file: {i+1}/{num_files} — {file.name}")
                continue

            print(f"[INFO] ({repository_id}) Processing file: {i+1}/{num_files} — {file.name}")
            try:
                with open(file, "r", encoding="utf-8") as fh:
                    code = fh.read()
            except Exception as e:
                print(f"[WARNING] Skipping {file}: {e}")
                continue

            probs = classify_numerical(code, droiddetect_tokenizer, droiddetect_model)
            probs_list = probs.detach().cpu().tolist()

            writer.writerow([filepath_str] + probs_list)
            f.flush()  # Ensure row is written to disk immediately

# tokenizer, model = load_model()

# ai_generated_python_code = """
# def compute_ai_usage_profile(repo_id, repo_path, commits):
#     def compute_ai_usage_profile(repo_id, repo_path, commits):
#     try
#         link = f"{repo_id} ({len(commits)} commits)"
#         print(f"[INFO] Processing {link}...")
# """

# classify_visual(ai_generated_python_code, tokenizer, model)
# probs, predicted, confidence = classify_numerical(ai_generated_python_code, tokenizer, model)
# print(f"Predictions: {probs}, Predicted class: {predicted}, Confidence: {confidence:.4f}")
