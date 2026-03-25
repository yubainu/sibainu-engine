import torch, numpy as np, sys, uuid, os, gc, requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

A_U = "https://yubainu-gemma2b-re-open.hf.space/proxy" 
M_I = "google/gemma-2b" 
D = "cuda" if torch.cuda.is_available() else "cpu"
TKN = str(uuid.uuid4())

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
        M_I, 
        quantization_config=q_config,
        device_map={"": 0},
        max_memory={0: "3.5GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="eager"
    )
except Exception as e:
    sys.exit(1)

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
            del o_p
        
        m_s = 0.0
        recovered = False

        for i in range(30):
            with torch.no_grad():
                o = md(i_d, output_hidden_states=True)
                n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
                
                p_k = [float(i), float(l_p)]
                p_k.extend(v_a.tolist()) 
                p_k.extend(v_a.tolist()) 
                for ly in o.hidden_states:
                    p_k.extend(ly[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                p_k.extend(o.logits[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                
                try:
                    payload = {"packet": p_k, "rst": (i==0), "token": TKN, "is_b": True}
                    res = requests.post(A_U, json=payload, timeout=10).json()
                    
                    cur_s = res.get("score", 0.0)
                    cmd = res.get("command", "CONTINUE")
                    
                    if cur_s > m_s: m_s = cur_s
                    
                    if cmd == "RECOVER":
                        print(f"\n[RECOVER] Detected risk at Token {i}. Retrying...")
                        recovered = True
                        break

                    t_s = tk.decode(n_t[0])
                    print(f"T{i:02d} [{t_s.strip()}]: {cur_s:.4f}")
                    
                except:
                    pass
                
                i_d = torch.cat([i_d, n_t], dim=-1)
                if n_t.item() == tk.eos_token_id: break
                
                del o
                if i % 5 == 0:
                    torch.cuda.empty_cache()

        if not recovered:
            res_text = tk.decode(i_d[0][l_p:], skip_special_tokens=True)
            print(f"\nQuestion: {q}")
            print(f"Answer: {res_text}")
            print(f"Max Score: {m_s:.4f}")
            break
        else:
            retry_count += 1
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    while True:
        q = input("\nEnter Question: ").strip()
        if q.lower() == 'exit': break
        if q: run_proc(q)