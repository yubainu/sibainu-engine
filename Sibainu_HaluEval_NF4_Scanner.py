import torch, requests, numpy as np, pandas as pd, sys, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

# --- Configuration ---
API_URL = "https://yubainu-sibainu-engine.hf.space/analyze_raw" 
HF_TOKEN = ""
MODEL_ID = "google/gemma-2b" # Using base model for pure state extraction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Initializing Sibainu Scanner on {DEVICE}...")

# 4-bit Quantization (NF4) for RTX 3050 (4GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config,
    device_map="auto", 
    token=HF_TOKEN,
    attn_implementation="eager"
)

def scan_sample(question, answer, is_first=True):
    prompt = f"Question: {question}\nAnswer: "
    ins = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    p_len = ins["input_ids"].shape[1]
    t_tokens = tokenizer(answer, return_tensors="pt").input_ids.to(DEVICE)
    
    if t_tokens[0, 0] == tokenizer.bos_token_id: 
        t_tokens = t_tokens[:, 1:]

    with torch.no_grad():
        p_out = model(ins["input_ids"], output_hidden_states=True)
        anc = p_out.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        seq_p = p_out.hidden_states[-1][0, p_len-1, :].to(torch.float32).cpu().numpy()
    
    c_ids, max_s = ins["input_ids"], 0.0
    
    for i in range(min(30, t_tokens.shape[1])):
        with torch.no_grad():
            o = model(c_ids, output_hidden_states=True)
            all_l = np.concatenate([h[0, -1, :].to(torch.float32).cpu().numpy() for h in o.hidden_states])
            lgt = o.logits[0, -1, :].to(torch.float32).cpu().numpy()
            blob = np.concatenate([np.array([float(i), float(p_len)]), anc, seq_p, all_l, lgt]).tolist()
            
            try:
                res = requests.post(
                    API_URL, 
                    json={"packet": blob, "rst": (i==0 and is_first)}, 
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
                    timeout=60
                ).json()
                s = res.get("score", 0.0)
                if s > max_s: max_s = s
            except Exception as e:
                print(f"\n[!] Server connection error at token {i}: {e}")
                break
            
            c_ids = torch.cat([c_ids, t_tokens[:, i:i+1]], dim=-1)
    return max_s

def run_direct_benchmark(start_id, count):
    print(f"[*] Loading HaluEval(QA) from Hugging Face datasets...")
    dataset = load_dataset("pminervini/HaluEval", "qa", split="data")
    
    results = []
    for i in range(start_id, start_id + count):
        data = dataset[i]
        print(f"\n[{i}] Processing Question: {data['question'][:50]}...")
        
        # Audit Right Answer (Label 0)
        s_right = scan_sample(data["question"], data["right_answer"], is_first=True)
        results.append({"id": i, "type": "right", "label": 0, "score": s_right})
        print(f"    >> Right Score: {s_right:.4f}")
        
        # Audit Hallucinated Answer (Label 1)
        s_hallu = scan_sample(data["question"], data["hallucinated_answer"], is_first=True)
        results.append({"id": i, "type": "hallu", "label": 1, "score": s_hallu})
        print(f"    >> Hallu Score: {s_hallu:.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    output_name = f"halueval_results_{start_id}_to_{start_id+count-1}.csv"
    df.to_csv(output_name, index=False)
    print(f"\n[*] Scan completed. Data saved to {output_name}")

if __name__ == "__main__":
    try:
        start_id = int(input("Enter Start ID (e.g., 0): "))
        count = int(input("Enter Number of Samples to Scan: "))
        run_direct_benchmark(start_id, count)
    except ValueError:
        print("[!] Invalid input. Please enter integers only.")