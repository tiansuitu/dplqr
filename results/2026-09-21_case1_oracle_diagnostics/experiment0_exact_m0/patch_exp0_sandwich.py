from pathlib import Path
import re

p = Path(r"C:\Users\Tiansui Tu\Documents\GitHub\dplqr\results\2026-09-21_case1_oracle_diagnostics\experiment0_exact_m0\run_experiment0.py")
src = p.read_text(encoding="utf-8")
orig = src

pat = r"def oracle_sandwich\([\s\S]*?return Sigma1, Sigma2, V\n"
repl = (
    "def oracle_sandwich(X: np.ndarray, f0: float, tau: float):\n"
    "    \"\"\"Linear QR sandwich on raw X when m0 is known (not residualized Xt).\n"
    "\n"
    "    With known m0, Y-m0(Z)=X'theta+eps is ordinary linear median regression.\n"
    "    Use E[XX']; reserve Xt=X-phi*(Z) for estimated-m experiments.\n"
    "    \"\"\"\n"
    "    n = X.shape[0]\n"
    "    Xt = X.astype(np.float64)\n"
    "    Sxx = (Xt.T @ Xt) / n\n"
    "    Sigma1 = tau * (1.0 - tau) * Sxx\n"
    "    Sigma2 = f0 * Sxx\n"
    "    A = np.linalg.solve(Sigma2, np.eye(Sigma2.shape[0]))\n"
    "    V = A @ Sigma1 @ A\n"
    "    return Sigma1, Sigma2, V\n"
)
src2, n = re.subn(pat, repl, src, count=1)
print("oracle_sandwich replacements:", n)
src = src2 if n else src

pairs = [
    ("oracle_sandwich(X_tilde, F0, TAU)", "oracle_sandwich(X, F0, TAU)"),
    ("oracle_sandwich(X_tilde, f0, tau)", "oracle_sandwich(X, f0, tau)"),
    ("oracle_sandwich(data[\"X_tilde\"]", "oracle_sandwich(data[\"X\"]"),
    ("(X_tilde * score[:, None]).sum(axis=0) / sqrt_n", "(X * score[:, None]).sum(axis=0) / sqrt_n"),
    ("(X_tilde * score[:, None]).sum(axis=0) / math.sqrt(n)", "(X * score[:, None]).sum(axis=0) / math.sqrt(n)"),
    ("X_tilde * score", "X * score"),
]
for a, b in pairs:
    c = src.count(a)
    if c:
        src = src.replace(a, b)
        print(f"replaced {c}x: {a[:70]}")

if src == orig:
    print("NO CHANGES — dumping region")
    i = src.find("def oracle_sandwich")
    print(src[i:i+600])
    for line in src.splitlines():
        if "oracle_sandwich(" in line or ("S_n" in line and "X" in line):
            print("LINE:", line)
else:
    p.write_text(src, encoding="utf-8")
    print("wrote", p)

rp = Path(r"C:\Users\Tiansui Tu\Documents\GitHub\dplqr\results\2026-09-21_case1_oracle_diagnostics\experiment0_exact_m0\README.md")
txt = rp.read_text(encoding="utf-8")
if "linear_QR_raw_X" not in txt:
    txt = txt.rstrip() + "\n\n## Sandwich note (corrected 2026-09-21)\n\nWith known \(m_0\), Experiment 0 is ordinary linear median regression of \(Y-m_0(Z)\) on \(X\). Oracle SEs / \(T\) / coverage / IF diagnostics use **raw \(X\)** (not \(\\tilde X=X-\\varphi^*(Z)\)). Residualized \(\\tilde X\) is reserved for estimated-\(m\) experiments. See `SANDWICH_CORRECTION.md`.\n"
    rp.write_text(txt, encoding="utf-8")
    print("README updated")
else:
    print("README already notes sandwich")
