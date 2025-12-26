import time
import os
import torch
from model_loader import load_codet5
from generator import generate_function_bodies

# Mock config matching the new PRESET_QUALITY in server.py
# n_candidates=12, tries=1, max_new_tokens=200
NEW_QUALITY_CONFIG = dict(
    n_candidates=12,
    tries=1,
    max_new_tokens=200,
    temperature=0.75,
    top_p=0.92
)

CHECKPOINT = os.getenv("CODET5_CHECKPOINT", "Salesforce/codet5p-220m-py")

def main():
    print("=== Quality Mode Speed Verification ===")
    print(f"Model: {CHECKPOINT}")
    print(f"Params: {NEW_QUALITY_CONFIG}")
    
    # 1. Load Model
    t0 = time.perf_counter()
    print("Loading model... (this takes time but happens only once at boot)")
    tokenizer, model, device = load_codet5(CHECKPOINT)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s on {device}")
    
    # 2. Test Generation Speed
    prefix = "def fib(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"
    
    print("\n--- Running Generation ---")
    t_gen_start = time.perf_counter()
    
    # Simulate the loop in server.py _run_round
    # Note: server.py loops 'tries' times calling generate_function_bodies, 
    # but generate_function_bodies ALSO has a 'tries' arg.
    # In server.py:
    #   bodies = generate_function_bodies(..., tries=tries, ...)
    # So we just call it once with our config.
    
    bodies = generate_function_bodies(
        tokenizer, model, device,
        code_prefix=prefix,
        n=NEW_QUALITY_CONFIG["n_candidates"],
        tries=NEW_QUALITY_CONFIG["tries"],
        max_new_tokens=NEW_QUALITY_CONFIG["max_new_tokens"],
        temperature=NEW_QUALITY_CONFIG["temperature"],
        top_p=NEW_QUALITY_CONFIG["top_p"],
    )
    
    t_gen_end = time.perf_counter()
    duration = t_gen_end - t_gen_start
    
    print(f"Generated {len(bodies)} unique candidates.")
    print(f"Duration: {duration:.2f}s")
    
    # 3. Validation
    # Client timeout is usually 60s.
    # Evaluation also takes time (approx 0.5s * candidates).
    # Estimated Eval time:
    est_eval_time = len(bodies) * 0.5 
    total_est = duration + est_eval_time
    
    print(f"\n--- Analysis ---")
    print(f"Generation Time: {duration:.2f}s")
    print(f"Est. Eval Time : {est_eval_time:.2f}s (assuming ~0.5s/cand)")
    print(f"Total Est. Time: {total_est:.2f}s")
    
    THRESHOLD = 45.0 # Safety margin below 60s
    if total_est < THRESHOLD:
        print(f"\n[PASS] Total time {total_est:.2f}s is well within 60s timeout.")
    else:
        print(f"\n[WARN] Total time {total_est:.2f}s is close to or exceeds timeout!")

if __name__ == "__main__":
    main()
