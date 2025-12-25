from model_loader import load_codet5
from generator import generate_function_bodies
from evaluator import EvalConfig, Evaluator

def main():
    tokenizer, model, device = load_codet5("Salesforce/codet5p-220m-py")

    func_name = "sum_of_squares"
    signature = "(n)"
    docstring = "Return the sum of squares from 1 to n inclusive."

    prefix = f"""# Python 3
# Generate ONLY the function body (no def, no tests, no print, no import, no I/O).
# Pure function: deterministic, must return a value.
def {func_name}{signature}:
    \"\"\"{docstring}\"\"\"
"""

    candidates = generate_function_bodies(
        tokenizer, model, device,
        code_prefix=prefix,
        n=12,
        tries=2,
        max_new_tokens=160,
        temperature=0.9,
        top_p=0.95
    )

    cfg = EvalConfig(timeout_s=3.0, repeats=2, probe_cases=6, pool_workers=1)
    ev = Evaluator(cfg)

    best, leaderboard = ev.select_best_generic(candidates, func_name, signature, docstring)
    ev.close()

    print("=== TOP 5 ===")
    for r in leaderboard[:5]:
        print(f"#{r['idx']:02d} score={r['score']:.1f} tests={r['passed']}/{r['total']} reason={r['reason']} det={r['determinism']} ms={r['avg_ms']:.1f}")

    print("\n=== BEST CODE ===")
    print(best["code"])

if __name__ == "__main__":
    main()
