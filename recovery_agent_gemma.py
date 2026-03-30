import torch, numpy as np, sys, uuid, os, gc, requests, time, base64, hashlib, secrets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


A_U = "https://yubainu-gemma2b-re-open2.hf.space/"  # 中間サーバーのURL
M_I = "google/gemma-2b"
D = "cuda" if torch.cuda.is_available() else "cpu"


C_A, C_B, C_N = "\033[36m", "\033[31m", "\033[0m"


P = int("FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
        "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
        "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
        "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
        "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
        "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
        "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
        "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
        "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
        "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
        "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
G = 2


print("Loading Model...")
q_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
tk = AutoTokenizer.from_pretrained(M_I)
md = AutoModelForCausalLM.from_pretrained(M_I, quantization_config=q_config, device_map={"": 0})

def get_shared_key(token):
    priv_a = secrets.randbelow(P)
    pub_a = pow(G, priv_a, P)
    try:
        res = requests.post(A_U, json={"cmd": "KEY_EXCHANGE", "pub_key": str(pub_a), "token": token}, timeout=20).json()
        shared = pow(int(res["pub_key"]), priv_a, P)
        return hashlib.sha256(str(shared).encode()).digest()
    except: return None

def scramble(v, step, key):
    seed = int.from_bytes(hashlib.sha256(key + str(step).encode()).digest()[:4], 'big')
    np.random.seed(seed)
    v_f32 = v.astype(np.float32)
    mask = np.random.normal(1.0, 0.02, v.shape).astype(np.float32)
    offset = np.random.uniform(-0.05, 0.05, v.shape).astype(np.float32)
    return ((v_f32 * mask) + offset).astype(np.float16)

def get_gauge(score, length=10):
    filled = int(min(score / 5.0 * length, length))
    return f"[{'#' * filled}{'-' * (length - filled)}]"

def run_proc(q):
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        curr_t = str(uuid.uuid4())
        key = get_shared_key(curr_t)
        if not key: 
            print("Connection failed."); break

        t = f"Question: {q}\nAnswer: "
        ins = tk(t, return_tensors="pt").to(D)
        i_d = ins["input_ids"]
        p_len = i_d.shape[1]
        
        with torch.no_grad():
            o_p = md(i_d, output_hidden_states=True)
            v_anc = o_p.hidden_states[-1][0, -1, :].to(torch.float16).cpu().numpy()
            v_pai = o_p.hidden_states[-1][0, p_len-1, :].to(torch.float16).cpu().numpy()

        print(f"\n--- Attempt {retry_count+1} ---")
        sys.stdout.write("Output: "); sys.stdout.flush()
        
        is_recovered = False
        m_score = 0.0

        for i in range(30):
            with torch.no_grad():
                o = md(i_d, output_hidden_states=True)
                h_s = o.hidden_states
                top_v, _ = torch.topk(o.logits[0, -1, :].float(), 2)
                l_data = top_v.to(torch.float16).cpu().numpy().tobytes()
                
                buf = [np.array([float(i), float(p_len), float(len(h_s))], dtype=np.float16).tobytes()]
                buf.append(scramble(v_anc, i, key).tobytes())
                buf.append(scramble(v_pai, i, key).tobytes())
                for lv in h_s:
                    buf.append(scramble(lv[0, -1, :].to(torch.float16).cpu().numpy(), i, key).tobytes())
                
                try:
                    res = requests.post(A_U, json={
                        "packet_b64": base64.b64encode(b"".join(buf)).decode('utf-8'),
                        "logits_b64": base64.b64encode(l_data).decode('utf-8'),
                        "token": curr_t
                    }, timeout=15).json()
                    
                    score = res.get("score", 0.0)
                    m_score = max(m_score, score)
                    
                    
                    if res.get("status") == "RECOVER_TRIGGERED" or score >= 3.651:
                        print(f"\n{C_B}>> {get_gauge(score)} INTERVENTION: HIGH-ORDER DISTORTION DETECTED. REGENERATING...{C_N}")
                        is_recovered = True
                        break
                    
                    
                    if score >= 2.5 and "steering_vector" in res:
                        sys.stdout.write(f"\n{C_A}>> {get_gauge(score)} INTERVENTION: CORRECTING GEOMETRIC DISTORTION...{C_N} ")
                        s_v = torch.tensor(res["steering_vector"]).to(D).to(torch.float16)
                        n_t = torch.argmax(md.lm_head(h_s[-1][0, -1, :] + s_v), dim=-1).view(1, 1)
                    else:
                        n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)

                    word = tk.decode(n_t[0], skip_special_tokens=True)
                    sys.stdout.write(word); sys.stdout.flush()
                    i_d = torch.cat([i_d, n_t], dim=-1)
                    if n_t.item() == tk.eos_token_id: break
                    
                except:
                   
                    n_t = torch.argmax(o.logits[:, -1, :], dim=-1).unsqueeze(-1)
                    i_d = torch.cat([i_d, n_t], dim=-1)
                    word = tk.decode(n_t[0], skip_special_tokens=True)
                    sys.stdout.write(word); sys.stdout.flush()
                    if n_t.item() == tk.eos_token_id: break

        if not is_recovered:
            print(f"\n\nAnalysis: {get_gauge(m_score)} Stability Verified.")
            break
        else:
            retry_count += 1
            torch.cuda.empty_cache(); gc.collect()
            time.sleep(1.0)

if __name__ == "__main__":
    while True:
        query = input("\nQuery > ").strip()
        if query: run_proc(query)