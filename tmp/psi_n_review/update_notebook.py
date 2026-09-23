import copy
import json
from pathlib import Path

path = Path('results/2026-09-23_psi_n_thm33/psi_n_diagnostic.ipynb')
original = path.read_text(encoding='utf-8')
backup = Path('tmp/psi_n_review/psi_n_diagnostic.before.ipynb')
if not backup.exists():
    backup.write_text(original, encoding='utf-8')
nb = json.loads(original)


def source(i):
    return ''.join(nb['cells'][i]['source'])


def put(i, value):
    nb['cells'][i]['source'] = value.splitlines(keepends=True)


for i, cell in enumerate(nb['cells']):
    cell['id'] = f'psi-n-{i:02d}'
    if cell['cell_type'] == 'markdown':
        put(i, source(i).replace(r'\(', '$').replace(r'\)', '$')
            .replace(r'\[', '$$').replace(r'\]', '$$'))

put(0, source(0).replace(
    'If the projected-score step is on track, $\\hat a$ should be near zero.',
    'The primary target is $S_n\\to_p0$. Record both $|\\hat a|$ and '
    '$\\sqrt n\,|\\hat a|$: unscaled shrinkage alone does not check the required rate. '
    'The adjustment is a supporting diagnostic; its relation to the score needs local curvature and regularity.'
).replace('**frozen-$h$**', '**frozen nuisance** $h$'))

put(2, source(2).replace('from scipy.optimize import linprog',
    'from scipy.optimize import linprog\nfrom scipy import sparse').replace(
    'THETA0 = 1.0',
    'if TAU != 0.5:\n    raise ValueError("This DGP and training loss are configured for median regression (TAU = 0.5).")\nTHETA0 = 1.0'))

put(8, source(8).replace(
    'A_eq = np.hstack([(-xt).reshape(n, 1), np.eye(n), -np.eye(n)])',
    '# a*xt + u+ - u- = r, hence u+ - u- = r - a*xt.\n'
    '    A_eq = sparse.hstack(\n'
    '        [sparse.csr_matrix(xt.reshape(n, 1)), sparse.eye(n), -sparse.eye(n)],\n'
    '        format="csr",\n'
    '    )'
).replace(
    '"abs_a_hat": float(abs(a_hat)),',
    '"abs_a_hat": float(abs(a_hat)),\n'
    '        "sqrt_n_a_hat": float(sqrt_n * a_hat),\n'
    '        "abs_sqrt_n_a_hat": float(sqrt_n * abs(a_hat)),'))

put(7, source(7) + '\nThe LP uses $a\\tilde X_i+u_i^+-u_i^-=\\hat r_i$, '
    'so it minimizes the stated residual $\\hat r_i-a\\tilde X_i$. '
    'The previous minus sign on the first LP column returned $-\\hat a$; '
    'its absolute value was unaffected. Sparse constraints avoid quadratic storage.\n')

put(11, source(11) + '\nUse a fresh output directory whenever changing the DGP, optimizer, network, '
    'or diagnostic definitions. The restart key checks only `(n, rep)`, not the configuration.\n')

put(13, '## Summary table and four plots\n\n'
    'Inspect the distribution of $|S_n|$, including upper quantiles and exceedance frequencies. '
    'Convergence in probability requires $P(|S_n|>\\epsilon)\\to0$ for every fixed '
    '$\\epsilon>0$. The thresholds below are descriptive diagnostics, not hypothesis tests.\n')

put(14, source(14).replace(
    'df = pd.read_csv(RESULTS_CSV)',
    'df = pd.read_csv(RESULTS_CSV)\n'
    '# This also permits summaries of older rows that stored only abs_a_hat.\n'
    'df["abs_sqrt_n_a_hat"] = np.sqrt(df["n"]) * df["abs_a_hat"]'
).replace(
    '"sd_abs_Sn": float(g["abs_Sn"].std(ddof=1)),',
    '"sd_abs_Sn": float(g["abs_Sn"].std(ddof=1)),\n'
    '            "mc_se_mean_abs_Sn": float(g["abs_Sn"].std(ddof=1) / math.sqrt(len(g))),\n'
    '            "median_abs_Sn": float(g["abs_Sn"].median()),\n'
    '            "q90_abs_Sn": float(g["abs_Sn"].quantile(0.9)),\n'
    '            "prob_abs_Sn_gt_0_1": float((g["abs_Sn"] > 0.1).mean()),\n'
    '            "prob_abs_Sn_gt_0_25": float((g["abs_Sn"] > 0.25).mean()),'
).replace(
    '"sd_abs_a_hat": float(g["abs_a_hat"].std(ddof=1)),',
    '"sd_abs_a_hat": float(g["abs_a_hat"].std(ddof=1)),\n'
    '            "mean_abs_sqrt_n_a_hat": float(g["abs_sqrt_n_a_hat"].mean()),\n'
    '            "sd_abs_sqrt_n_a_hat": float(g["abs_sqrt_n_a_hat"].std(ddof=1)),'
).replace(
    '    plt.close(fig)',
    '    from IPython.display import display\n'
    '    display(fig)\n'
    '    plt.close(fig)'
).replace(
    'print("Wrote figures under", FIG_DIR)',
    'fig, ax = plt.subplots(figsize=(5.5, 3.8))\n'
    '_scatter_mean(\n'
    '    ax, "n", "abs_sqrt_n_a_hat", r"$|\\sqrt{n}\\hat a|$",\n'
    '    r"$|\\sqrt{n}\\hat a|$ vs $n$", "abs_sqrt_n_ahat_vs_n.png",\n'
    ')\n\n'
    'print("Wrote figures under", FIG_DIR)'))

put(15, '## Descriptive summary (does not establish an asymptotic limit)\n')
put(16, '''s = summary.sort_values("n")
print("=== Finite-sample diagnostics for the fitted Adam estimator ===")
for label, column in [
    ("mean |sqrt(n) Psi_hat|", "mean_abs_Sn"),
    ("90th percentile of |sqrt(n) Psi_hat|", "q90_abs_Sn"),
    ("fraction |sqrt(n) Psi_hat| > 0.1", "prob_abs_Sn_gt_0_1"),
    ("fraction |sqrt(n) Psi_hat| > 0.25", "prob_abs_Sn_gt_0_25"),
    ("mean |sqrt(n) G_theta|", "mean_abs_sqrt_n_G_theta"),
    ("mean |a_hat|", "mean_abs_a_hat"),
    ("mean |sqrt(n) a_hat|", "mean_abs_sqrt_n_a_hat"),
]:
    print(label + ":", dict(zip(s["n"].astype(int), s[column].astype(float))))

if len(s) >= 2:
    first, last = s["mean_abs_Sn"].iloc[[0, -1]]
    if first > 0:
        print(f"Largest/smallest-n ratio of mean |Sn|: {last / first:.3f}")
print("A decrease is descriptive evidence over this n grid; it is not proof of convergence to zero.")
print("Check Monte Carlo uncertainty, tails, larger n, and sensitivity to optimization and network size.")
print("Small ordinary theta scores or unscaled a_hat alone do not establish the projected-score rate.")
print("This notebook does not separately evaluate the empirical-process or population Taylor remainder.")
''')

review = r'''## What this notebook can and cannot check

**Yes: it directly probes the fitted projected-score step**
$\Psi_n(\hat\zeta,\hat h)=o_p(n^{-1/2})$ for this numerical estimator.
**It does not establish the full asymptotic expansion or prove Theorem 3.3.**
At review time the notebook was unexecuted and its folder contained no Monte Carlo results.

### Why the implemented score is the right one

The supplement, pp. 4–5, defines

$$
\hat\zeta=\hat\theta-\theta_0,\qquad
\hat h(z)=\hat m(z)-m_0(z)+\hat\zeta\,\varphi^*(z).
$$

Consequently,

$$
\epsilon_i-\hat\zeta\tilde X_i-\hat h(Z_i)
=Y_i-X_i\hat\theta-\hat m(Z_i)=\hat r_i.
$$

So `Psi_hat` evaluates the stated $\Psi_n(\hat\zeta,\hat h)$ without explicitly constructing
$\hat h$. For the independent normal errors used here, $f(0\mid X,Z)$ is constant,
and the oracle $\varphi^*(Z)=E[X\mid Z]$ is the correct weighted projection.
The same formula generally needs a density-weighted projection under heteroskedasticity.
Use the **training sample** for this score: it is the empirical loss score at its fitted estimator.

### The optimization step needs checking

The ordinary parameter score and the projected score satisfy

$$
\widehat\Psi
=G_\theta+\frac1n\sum_{i=1}^n
\bigl[\tau-\mathbf1\{\hat r_i<0\}\bigr]\varphi^*(Z_i).
$$

Thus a small $G_\theta$ alone is insufficient. The frozen-$h$ move is

$$
\theta(a)=\hat\theta+a,\qquad
m_a(z)=\hat m(z)-a\varphi^*(z),\qquad
r_i(a)=\hat r_i-a\tilde X_i.
$$

The supplement's proof uses minimization along this direction. A fixed neural-network class
need not contain $m_a$, and finite-epoch Adam need not reach even the original joint minimum.
The auxiliary LP is useful for measuring the missing directional adjustment; it does not
make the original estimator a minimizer. Keep its result diagnostic rather than replacing
$\hat\theta$ with the adjusted estimate when assessing the original algorithm.

Raw $|\hat a|\to0$ is too weak for the desired score rate. Under suitable local curvature
and stochastic regularity, the expected relation is
$\hat a\approx-\widehat\Psi/J$, where

$$
J=E[f(0\mid X,Z)\tilde X^2]
=\frac{0.25}{\sqrt{2\pi}}>0
$$

for this DGP. Hence also inspect $\sqrt n\,|\hat a|$.
At exact zero residuals, the strict-indicator score selects a subgradient endpoint;
an exact quantile-regression optimum need not make that selected score exactly zero.

### What is missing for a full expansion check

With $\Psi_0$ denoting the population score and $\Psi_0(0,0)=0$, define

$$
\begin{aligned}
E_n&=\sqrt n\left\{(\Psi_n-\Psi_0)(\hat\zeta,\hat h)
                         -(\Psi_n-\Psi_0)(0,0)\right\},\\
T_n&=\sqrt n\left\{\Psi_0(\hat\zeta,\hat h)-J\hat\zeta\right\}.
\end{aligned}
$$

The exact decomposition is

$$
J\sqrt n\,(\hat\theta-\theta_0)+\sqrt n\,\Psi_n(0,0)
=S_n-E_n-T_n.
$$

The notebook measures $S_n$, but does not separately measure $E_n$ or $T_n$.
All three being $o_p(1)$ would give the usual linear expansion, with sign and normalization
derived from the score defined above. Even at the truth, $\sqrt n\,\Psi_n(0,0)$ normally
has a nondegenerate limit; it is not the score that should vanish.
A fuller experiment should record the left-hand side and, using the known DGP and a sufficiently
accurate independent integration sample, estimate the two remainder terms.

### Limits of the current experiment

- The grid $n\in\{500,1000,2000\}$ and 30 replications gives a small finite-sample comparison.
  Inspect Monte Carlo uncertainty and tail frequencies; endpoint decreases cannot certify a limit.
- The network width is fixed at `[16, 16]`, while the theorem assumes a network class that grows
  with $n$ (Assumption 4). Fixed learning rate and 500 epochs also leave optimization error uncontrolled.
- Gaussian $X-\varphi^*(Z)$ makes $X$ unbounded, outside the theorem's compact-covariate
  Assumption 2. This is a diagnostic toy model, not an exact replication of all assumptions.
- `TAU = 0.5` is essential to this normal-error DGP and the absolute-loss training code;
  a guard now rejects other values.
- `final_training_loss` is the average batch loss during the last epoch, not a reevaluation
  at the returned parameters. The projected score itself is evaluated at the returned fit.
- Restarting checks only `(n, rep)`, and `ROOT` is the kernel working directory. Run from this
  notebook's folder and use a fresh output directory for each changed configuration.

Sources: [main paper](../../Zhong%20and%20Wang%202024%20JBES.pdf), Section 3 and Theorem 3.3;
[supplement](../../Zhong%20and%20Wang%202024%20JBES%20supplement.pdf), pp. 4–6.
'''
nb['cells'].append({'cell_type': 'markdown', 'id': 'psi-n-review', 'metadata': {},
                    'source': review.splitlines(keepends=True)})
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Updated', path)
