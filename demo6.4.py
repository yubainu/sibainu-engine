import torch, requests, numpy as np, sys
from transformers import AutoTokenizer, AutoModelForCausalLM

API_URL = "https://yubainu-sibainu-engine.hf.space/analyze_raw" 
HF_TOKEN = ""
MODEL_ID = "google/gemma-2b-it" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)

def autonomous_audit(q):
    p = f"Question: {q}\nAnswer: "
    ins = tokenizer(p, return_tensors="pt").to(DEVICE)
    input_ids = ins["input_ids"]
    p_len = input_ids.shape[1]
    
    with torch.no_grad():
        p_out = model(input_ids, output_hidden_states=True)
        anc = p_out.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        seq_p = p_out.hidden_states[-1][0, p_len-1, :].to(torch.float32).cpu().numpy()
    
    max_s = 0.0
    print("\n--- Model Generating & Sibainu Engine Auditing ---")
    
    for i in range(20):
        with torch.no_grad():
            o = model(input_ids, output_hidden_states=True)
            
            next_token_id = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            all_l = np.concatenate([h[0, -1, :].to(torch.float32).cpu().numpy() for h in o.hidden_states])
            lgt = o.logits[0, -1, :].to(torch.float32).cpu().numpy()
            blob = np.concatenate([np.array([float(i), float(p_len)]), anc, seq_p, all_l, lgt]).tolist()
            
            try:
                res = requests.post(API_URL, json={"packet": blob, "rst": i==0}, 
                                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=60).json()
                s = res.get("score", 0.0)
                if s > max_s: max_s = s

                next_token_str = tokenizer.decode(next_token_id[0])
                print(f"Token {i} [{next_token_str}]: Score = {s:.4f}")
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                    
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                
            except Exception as e:
                print(f"\n[!] Error: {e}")
                break
    
    full_answer = tokenizer.decode(input_ids[0][p_len:], skip_special_tokens=True)
    print(f"\n--- Final Result ---")
    print(f"Full Answer: {full_answer}")
    print(f"Max Combined Score: {max_s:.4f}")
    print(f"Result: {'HIGH_RISK' if max_s >= 3.6510 else 'SAFE'}")

if __name__ == "__main__":
    q_in = input("Question: ")
    if q_in: autonomous_audit(q_in)