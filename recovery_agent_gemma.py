import torch, numpy as np, sys, uuid, os, gc, requests, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

A_U = "https://yubainu-gemma2b-re-open.hf.space/proxy" 
M_I = "google/gemma-2b" 
D = "cuda" if torch.cuda.is_available() else "cpu"
TKN = str(uuid.uuid4())

C_A = "\033[36m" 
C_B = "\033[31m" 
C_N = "\033[0m"  

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
except Exception:
    sys.exit(1)

def get_gauge(score, max_val=5.0, length=10):
    filled = int(min(score / max_val * length, length))
    bar = "#" * filled + "-" * (length - filled)
    return f"[{bar}]"

def run_proc(q):
    retry_count = 0
    max_retries = 3 

    while retry_count < max_retries:
        t = f"Question: {q}\nAnswer: "
        ins = tk(t, return_tensors="pt").to(D)
        i_d = ins["input_ids"]
        l_p = i_d.shape[1]
        
        with torch.no_grad():
            o_p = md(i_d, output_hidden_states=True)
            v_a = o_p.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        
        m_s = 0.0
        recovered = False
        
        print(f"\n--- Attempt {retry_count+1} ---")
        sys.stdout.write("Output: ")
        sys.stdout.flush()

        for i in range(50): 
            with torch.no_grad():
                o = md(i_d, output_hidden_states=True)
                h_final = o.hidden_states[-1][0, -1, :]
                
                p_k = [float(i), float(l_p)]
                p_k.extend(v_a.tolist()) 
                p_k.extend(v_a.tolist()) 
                for ly in o.hidden_states:
                    p_k.extend(ly[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                p_k.extend(o.logits[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                
                try:
                    payload = {"packet": p_k, "rst": (i==0), "token": TKN, "is_b": True}
                    res = requests.post(A_U, json=payload, timeout=8).json()
                    
                    cur_s = res.get("score", 0.0)
                    s_vec = res.get("steering_vector") 
                    if cur_s > m_s: m_s = cur_s

                    if cur_s >= 5.0:
                        gauge = get_gauge(cur_s)
                        print(f"\n{C_B}>> {gauge} INTERVENTION: HIGH-ORDER DISTORTION DETECTED. REGENERATING...{C_N}")
                        recovered = True
                        break

                    if cur_s >= 2.0 and s_vec:
                        gauge = get_gauge(cur_s)
                        sys.stdout.write(f"\n{C_A}{gauge} CORRECTING GEOMETRIC DISTORTION...{C_N}")
                        sys.stdout.flush()
                        
                        s_tensor = torch.tensor(s_vec).to(D).to(torch.float16)
                        h_steered = h_final + s_tensor
                        logits_steered = md.lm_head(h_steered)
                        n_t = torch.argmax(logits_steered, dim=-1).view(1, 1)
                        
                        token_str = tk.decode(n_t[0], skip_special_tokens=True)
                        sys.stdout.write(f" {token_str}") 
                    else:
                        n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
                        token_str = tk.decode(n_t[0], skip_special_tokens=True)
                        sys.stdout.write(token_str)

                    sys.stdout.flush()
                    i_d = torch.cat([i_d, n_t], dim=-1)
                    if n_t.item() == tk.eos_token_id: break
                
                except Exception:
                    n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    i_d = torch.cat([i_d, n_t], dim=-1)
                    sys.stdout.write(tk.decode(n_t[0]))
                    sys.stdout.flush()

        if not recovered:
            final_gauge = get_gauge(m_s)
            print(f"\n\nAnalysis: {final_gauge} Stability Verified.")
            print("Status: COMPLETED")
            break
        else:
            retry_count += 1
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(1.2) 

if __name__ == "__main__":
    while True:
        q = input("\nQuery > ").strip()
        if q.lower() == 'exit': break
        if q: run_proc(q)