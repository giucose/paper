import json
import os
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

def detect_from_directory(repo_path, tokenizer, model, commit, repository_id):
    import os
    import json
    import torch
    from pathlib import Path
    from collections import defaultdict

    storage_path = Path().parent.parent.parent.parent / "system_files" / "droiddetect_temp_storage" / "individual_files" / f"{repository_id}" / f"{commit}" / "droiddetect_file_results.json"

    dir_results = defaultdict(list)
    processed_files = set()
    file_results = []

    num_files = sum(len(files) for _, _, files in os.walk(repo_path))
    
    # Arbitrary threshold to avoid processing commits with too many file diffs, which could be very time-consuming for DroidDetect
    # This is a safeguard and can be adjusted based on performance observations
    # Observedd that some commits including pushing changes to many files (e.g. 1000+) can cause DroidDetect to run for hours, so we set a threshold to skip such commits
    if(num_files > 200): 
        print(f"[WARNING] Too many files ({num_files}) in commit {commit} at {repo_path}, skipping DroidDetect for this commit.")
        return None
    
    # ---- LOAD EXISTING STATE ----
    if storage_path.exists():
        try:
            with open(storage_path, "r", encoding="utf-8") as f:
                file_results = json.load(f)
                processed_files = {entry["file_path"] for entry in file_results}

            # rebuild directory tensors
            for entry in file_results:
                tensor = torch.tensor(entry["probabilities"])
                dir_results[entry["directory"]].append(tensor)

            print(f"[INFO] Resuming from {len(processed_files)} files")
        except Exception:
            print("[WARNING] Corrupt checkpoint, restarting")
            file_results = []
            processed_files = set()
    else:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
    # ---- PROCESS FILES ----
    for root, _, files in os.walk(repo_path):
        total = len(files)
        print(f"[INFO] Processing {total} files...")

        for i, file in enumerate(files):
            file_path = os.path.join(root, file)
            
            if file_path in processed_files:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as rf:
                    code = rf.read()
            except Exception as e:
                print(f"[WARNING] Skipping {file_path}: {e}")
                continue

            probs = classify_numerical(code, tokenizer, model)

            dir_results[root].append(probs)

            record = {
                "file_path": file_path,
                "directory": root,
                "probabilities": probs.detach().cpu().tolist()
            }

            file_results.append(record)
            processed_files.add(file_path)

            # ---- ATOMIC SAVE ----
            tmp_path = str(storage_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as wf:
                json.dump(file_results, wf)
            os.replace(tmp_path, storage_path)

            print(f"[DEBUG] {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    # ---- FINAL AGGREGATION ----
    average_results = {
        directory: torch.stack(probs_list).mean(dim=0)
        for directory, probs_list in dir_results.items()
    }

    return average_results

def compute_droiddetect_metrics(
    commits, 
    repo_path,
    repository_id,
    year,
    droiddetect_tokenizer, 
    droiddetect_model,
    storage_dir 
):
    path = storage_dir / "droiddetect_metrics.csv"
    if path.exists():
        print(f"[INFO] DROIDDETECT metrics already computed for {repository_id}, loading from {path}")
        # Load and return existing data
        rows = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                commit = row["commit"]
                rows[commit] = row
            headers_droiddetect = reader.fieldnames
        return rows, headers_droiddetect

    rows = {}
    base_dir = compute_diff_and_store(
        commits,
        repo_path,
        repository_id,
        year,
    )
  
    temp_storage_dir = Path().parent.parent.parent.parent / "system_files" / "droiddetect_temp_storage"
    temp_storage_dir.mkdir(parents=True, exist_ok=True)  
    temp_storage_path = temp_storage_dir / f"{repository_id}_droiddetect_temp.csv"
    headers_droiddetect = [
        "commit",
        "prob_human_droiddetect", "prob_machine_droiddetect",
        "prob_machine_refined_droiddetect",
        "prob_machine_generated_adversarial_droiddetect",
    ]
    
    already_processed_commits = set()
    if temp_storage_path.exists():
        with open(temp_storage_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_processed_commits.add(row["commit"])
    else:
        with open(temp_storage_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers_droiddetect)
    
    print("[INFO] Running DroidDetect on commit diffs...")
    with open(temp_storage_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for i, commit in enumerate(commits):
            if commit in already_processed_commits:
                print(f"[INFO] Commit {commit} already processed for DroidDetect, skipping... ({ i + 1 }/{ len(commits) })")
                continue

            flat_dir = base_dir / commit / "flat"
            print(f"[INFO] Processing commit {commit} for DroidDetect... ({ i + 1 }/{ len(commits) })")
            average_results = detect_from_directory(flat_dir, droiddetect_tokenizer, droiddetect_model, commit, repository_id)
            if(average_results is None):
                print(f"[WARNING] No results for commit {commit} (too many file diffs), skipping...")
                continue
            tensors = list(average_results.values())
            if tensors:
                probs = torch.stack(tensors).mean(dim=0).tolist()
                writer.writerow([commit] + probs)
            else:
                print(f"[WARNING] No DroidDetect results for commit {commit}, skipping.")
            
    path.parent.mkdir(parents=True, exist_ok=True)
    
    droiddetect_probs: Dict[str, Dict[Any, Any]] = {}
    
    with open(temp_storage_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            commit = row["commit"]
            droiddetect_probs[commit] = {
                "prob_human_droiddetect": float(row["prob_human_droiddetect"]),
                "prob_machine_droiddetect": float(row["prob_machine_droiddetect"]),
                "prob_machine_refined_droiddetect": float(row["prob_machine_refined_droiddetect"]),
                "prob_machine_generated_adversarial_droiddetect": float(row["prob_machine_generated_adversarial_droiddetect"]),
            }
            
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers_droiddetect)

        for commit in commits:
            if commit not in droiddetect_probs:
                continue

            probs_dict = droiddetect_probs[commit]
            writer.writerow([
                commit,
                probs_dict["prob_human_droiddetect"],
                probs_dict["prob_machine_droiddetect"],
                probs_dict["prob_machine_refined_droiddetect"],
                probs_dict["prob_machine_generated_adversarial_droiddetect"],
            ])
    
    return droiddetect_probs, headers_droiddetect

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
