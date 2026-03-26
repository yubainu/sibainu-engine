import torch, numpy as np, sys, uuid, os, time, requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Configuration ---
A_U = "https://yubainu-sibainu-proxy-v2.hf.space/" 
M_I = "google/gemma-2b" 
D = "cuda" if torch.cuda.is_available() else "cpu"
TKN = str(uuid.uuid4())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Model Loading (4-bit Optimized) ---
q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

try:
    tk = AutoTokenizer.from_pretrained(M_I)
    md = AutoModelForCausalLM.from_pretrained(
        M_I, quantization_config=q_config, device_map={"": 0},
        max_memory={0: "3.5GiB"}, low_cpu_mem_usage=True, attn_implementation="eager"
    )
except Exception as e:
    sys.exit(1)

def get_gauge(score, max_val=5.0):
    filled = int(min(score / max_val * 10, 10))
    return f"[{'#' * filled}{'-' * (10 - filled)}]"

def run_proc(q):
    t = f"Question: {q}\nAnswer: "
    ins = tk(t, return_tensors="pt").to(D)
    i_d = ins["input_ids"]
    l_p = i_d.shape[1]
    
    with torch.no_grad():
        o_p = md(i_d, output_hidden_states=True)
        v_a = o_p.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        v_s = v_a
    
    max_d = 0.0
    print(f"\n[*] Monitoring Hallucination (Session: {TKN[:8]})\n")

    for i in range(20):
        try:
            with torch.no_grad():
                o = md(i_d, output_hidden_states=True)
                n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
                
           
                p_k = [float(i), float(l_p)]
                p_k.extend(v_a.tolist())
                p_k.extend(v_s.tolist())
                for ly in o.hidden_states:
                    p_k.extend(ly[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                p_k.extend(o.logits[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                
    
                cur_s = 0.0
                eng_lat = 0.0
                try:
                    payload = {"packet": p_k, "rst": i == 0, "token": TKN, "is_b": True}
                    resp = requests.post(A_U, json={"data": [payload]}, timeout=35)
                    if resp.status_code == 200:
                        res_data = resp.json()["data"][0]
                        cur_s = float(res_data.get("score", 0.0))
                    
                        eng_lat = float(res_data.get("latency_ms", 0.0))
                except: pass
                
                if cur_s > max_d: max_d = cur_s
                
                t_str = tk.decode(n_t[0]).strip()
                
                print(f"Token {i:02d} | {get_gauge(cur_s)} Dist: {cur_s:.4f} | Eng: {eng_lat:>5.1f}ms | Next: '{t_str}'")
                
                i_d = torch.cat([i_d, n_t], dim=-1)
                if n_t.item() == tk.eos_token_id: break
                if i % 5 == 0: torch.cuda.empty_cache()
        except: break
    
    print(f"\nFinal Analysis: {get_gauge(max_d)} Max Distortion: {max_d:.4f}")
    print(f"Status: {'CRITICAL' if max_d >= 3.651 else 'STABLE'}")
    print(f"Result Text: {tk.decode(i_d[0][l_p:], skip_special_tokens=True)}")

if __name__ == "__main__":
    query = input("\nQuery > ").strip()
    if query: run_proc(query)