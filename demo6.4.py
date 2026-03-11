import torch, requests, numpy as np, sys, uuid, os, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

A_U = "https://yubainu-sibainu-engine.hf.space/analyze_raw" 
H_T = ""
M_I = "google/gemma-2b" 
D = "cuda" if torch.cuda.is_available() else "cpu"
TKN = str(uuid.uuid4())

q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

try:
    tk = AutoTokenizer.from_pretrained(M_I, token=H_T)
    md = AutoModelForCausalLM.from_pretrained(
        M_I, 
        quantization_config=q_config,
        device_map={"": 0},
        max_memory={0: "3.5GiB"},
        token=H_T,
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
except Exception as e:
    print(f"Load Error: {e}")
    sys.exit(1)

def run_proc(q):
    t = f"Question: {q}\nAnswer: "
    ins = tk(t, return_tensors="pt").to(D)
    i_d = ins["input_ids"]
    l_p = i_d.shape[1]
    
    with torch.no_grad():
        o_p = md(i_d, output_hidden_states=True)
        v_a = o_p.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        v_s = v_a
        del o_p
        torch.cuda.empty_cache()
    
    m_s = 0.0
    h = {"Authorization": f"Bearer {H_T}"} if H_T else {}
    
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
                try:
                    r_raw = requests.post(A_U, json={"packet": p_k, "rst": i==0, "token": TKN, "is_b": True}, headers=h, timeout=10)
                    cur_s = r_raw.json().get("score", 0.0)
                except:
                    pass
                
                if cur_s > m_s: m_s = cur_s
                
                t_s = tk.decode(n_t[0])
                print(f"Token {i} [{t_s}]: Score = {cur_s:.4f}")
                
                i_d = torch.cat([i_d, n_t], dim=-1)
                if n_t.item() == tk.eos_token_id: break
                
                del o
                if i % 2 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Loop Error: {e}")
            break
    
    res = tk.decode(i_d[0][l_p:], skip_special_tokens=True)
    print(f"\nResult: {res}")
    print(f"Max Score: {m_s:.4f}")
    print(f"Status: {'HIGH_RISK' if m_s >= 3.6510 else 'SAFE'}")

if __name__ == "__main__":
    q = input("Enter Question: ").strip()
    if q: run_proc(q)