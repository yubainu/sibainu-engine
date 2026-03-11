import torch, requests, numpy as np, pandas as pd, sys, gc, uuid, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset

API_URL = "https://yubainu-sibainu-engine.hf.space/analyze_raw" 
HF_TOKEN = "" # Token here
MODEL_ID = "google/gemma-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

print("[*] Initializing Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tk = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
md = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config,
    device_map={"": 0},
    max_memory={0: "3.2GiB"}, # VRAM消費をさらに絞る
    token=HF_TOKEN,
    attn_implementation="eager",
    low_cpu_mem_usage=True
)

def scan_sample(q, a, s_t):
    p = f"Question: {q}\nAnswer: "
    ins = tk(p, return_tensors="pt").to(DEVICE)
    p_l = ins["input_ids"].shape[1]
    
    t_t = tk(a, return_tensors="pt").input_ids.to(DEVICE)
    if t_t[0, 0] == tk.bos_token_id: t_t = t_t[:, 1:]

    with torch.no_grad():
        p_o = md(ins["input_ids"], output_hidden_states=True)
        v_a = p_o.hidden_states[-1][0, -1, :].to(torch.float32).cpu().numpy()
        v_s = v_a
        del p_o
        torch.cuda.empty_cache()
    
    c_ids, m_s = ins["input_ids"], 0.0
    h = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    
    for i in range(min(30, t_t.shape[1])):
        try:
            with torch.no_grad():
                o = md(c_ids, output_hidden_states=True)
                pkt = [float(i), float(p_l)]
                pkt.extend(v_a.tolist())
                pkt.extend(v_s.tolist())
                for ly in o.hidden_states:
                    pkt.extend(ly[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                pkt.extend(o.logits[0, -1, :].to(torch.float32).cpu().numpy().tolist())
                
                res_raw = requests.post(
                    API_URL, 
                    json={"packet": pkt, "rst": i==0, "token": s_t, "is_b": True}, 
                    headers=h, timeout=30
                )
                r = res_raw.json()
                s = r.get("score", 0.0)
                if s > m_s: m_s = s
                
                c_ids = torch.cat([c_ids, t_t[:, i:i+1]], dim=-1)
                del o
                if i % 3 == 0: # 頻繁に解放
                    gc.collect()
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Token Error: {e}")
            break
    return m_s

def run_bench(s_id, cnt):
    print("[*] Accessing HaluEval (Streaming)...")
    # streaming=True で全ダウンロードを回避
    ds = load_dataset("pminervini/HaluEval", "qa", split="data", streaming=True)
    
    res = []
    # ストリーミングから指定範囲を抽出
    for i, d in enumerate(ds):
        if i < s_id: continue
        if i >= s_id + cnt: break
        
        print(f"[{i}] Testing: {d['question'][:40]}...")
        
        s_r = scan_sample(d["question"], d["right_answer"], str(uuid.uuid4()))
        print(f"  > Right: {s_r:.4f}")
        res.append({"id": i, "type": "right", "label": 0, "score": s_r})
        
        s_h = scan_sample(d["question"], d["hallucinated_answer"], str(uuid.uuid4()))
        print(f"  > Hallu: {s_h:.4f}")
        res.append({"id": i, "type": "hallu", "label": 1, "score": s_h})
        
        gc.collect()
        torch.cuda.empty_cache()

    out = f"halueval_{s_id}_{s_id+cnt-1}.csv"
    pd.DataFrame(res).to_csv(out, index=False)
    print(f"[*] Done. Saved to {out}")

if __name__ == "__main__":
    try:
        s = int(input("Start ID: "))
        n = int(input("Count: "))
        run_bench(s, n)
    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)