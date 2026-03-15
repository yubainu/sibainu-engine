import torch, requests, numpy as np, sys, uuid, os, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

A_U = "https://yubainu-gemma2b-re.hf.space/analyze_raw" 
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

def run_proc(q, is_recovery=False):
    t = f"Question: {q}\nAnswer: "
    ins = tk(t, return_tensors="pt").to(D)
    i_d = ins["input_ids"]
    l_p = i_d.shape[1]
    
    with torch.no_grad():
        o_p = md(i_d, output_hidden_states=True)
        v_a = o_p.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        del o_p

    h = {"Authorization": f"Bearer {H_T}"}
    
    for i in range(30):
        try:
            with torch.no_grad():
                o = md(i_d, output_hidden_states=True)
                n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
                
                p_k = [float(i), float(l_p)]
                p_k.extend(v_a.tolist())
                p_k.extend(v_a.tolist())
                for ly in o.hidden_states:
                    p_k.extend(ly[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                p_k.extend(o.logits[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                
                cur_s = 0.0
                cmd = "CONTINUE"
                try:
                    r = requests.post(A_U, json={"packet": p_k, "rst": i==0, "token": TKN, "is_b": True}, headers=h, timeout=10).json()
                    cur_s = r.get("score", 0.0)
                    cmd = r.get("command", "CONTINUE")
                except:
                    pass
                
                t_s = tk.decode(n_t[0]).replace("\n", "\\n")
                print(f"Token {i:02} | Score: {cur_s:.4f} | Cmd: {cmd} | [{t_s}]")
                
                if is_recovery and cur_s > 4.5:
                    break

                if cmd == "RECOVER" and not is_recovery:
                    return "RECOVER_SIGNAL", None

                i_d = torch.cat([i_d, n_t], dim=-1)
                if n_t.item() == tk.eos_token_id: break
                
                del o
                if i % 2 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        except:
            break
    
    return "SUCCESS", tk.decode(i_d[0][l_p:], skip_special_tokens=True)

if __name__ == "__main__":
    q = input("Question: ").strip()
    if q:
        status, result = run_proc(q)
        if status == "RECOVER_SIGNAL":
            print("\n--- API COMMAND: RECOVERY ---\n")
            _, result = run_proc(q, is_recovery=True)
        print(f"\nFinal: {result}")