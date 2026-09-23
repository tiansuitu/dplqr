import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile

import nbformat
import numpy as np

root = Path(__file__).resolve().parents[2]
path = root / 'results/2026-09-23_psi_n_thm33/psi_n_diagnostic.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type == 'code':
        ast.parse(cell.source)
    else:
        assert not any(d in cell.source for d in [r'\(', r'\)', r'\[', r'\]'])
        assert cell.source.count('$$') % 2 == 0

ns = {'__name__': '__main__'}
log = io.StringIO()
with contextlib.redirect_stdout(log):
    for i in [2, 4, 6, 8]:
        exec(compile(nb.cells[i].source, f'cell_{i}', 'exec'), ns)
    solve = ns['a_hat_linprog']
    loss = ns['check_loss']
    assert np.isclose(solve(np.full(3, 2.0), np.ones(3)), 2.0)
    rng = np.random.default_rng(843)
    for tau in [0.2, 0.5, 0.8]:
        for _ in range(8):
            r = rng.normal(size=31)
            x = rng.normal(size=31)
            x[0] = 0.0
            a = solve(r, x, tau)
            knots = r[x != 0] / x[x != 0]
            exact_best = min(loss(r - k*x, tau) for k in knots)
            assert np.isclose(loss(r-a*x, tau), exact_best, atol=1e-9)
            assert loss(r-a*x, tau) <= loss(r, tau) + 1e-10
            assert np.isclose(solve(r + 1.75*x, x, tau), a + 1.75, atol=1e-8)

    with tempfile.TemporaryDirectory(prefix='smoke_', dir=root / 'tmp/psi_n_review') as temp:
        run = Path(temp)
        ns.update(N_VALUES=[32, 48], Q=2, EPOCHS=2, BATCH_SIZE=16,
                  RUN_DIR=run, FIG_DIR=run / 'figures',
                  RESULTS_CSV=run / 'replication_results.csv',
                  SUMMARY_CSV=run / 'summary.csv', CONFIG_JSON=run / 'config.json')
        ns['torch'].set_num_threads(1)
        for i in [10, 12, 14, 16]:
            exec(compile(nb.cells[i].source, f'cell_{i}', 'exec'), ns)
        df = ns['df']
        assert len(df) == 4
        assert np.isfinite(df.select_dtypes('number').to_numpy()).all()
        assert np.allclose(df.Sn, np.sqrt(df.n) * df.Psi_hat)
        assert np.allclose(df.abs_sqrt_n_a_hat, np.sqrt(df.n) * df.a_hat.abs())
        assert len(list((run / 'figures').glob('*.png'))) == 4
        assert len(ns['summary']) == 2
        for n in [32, 48]:
            d = ns['generate_sample'](n, np.random.default_rng(n))
            theta_hat, m_hat, _ = ns['train_joint'](d['Y'], d['X'], d['Z'], n)
            zeta = theta_hat - ns['THETA0']
            h = m_hat - d['m0'] + zeta*d['phi']
            eps = d['Y'] - ns['THETA0']*d['X'] - d['m0']
            r = d['Y'] - theta_hat*d['X'] - m_hat
            assert np.allclose(r, eps-zeta*d['X_tilde']-h)
            score = ns['TAU'] - (r < 0)
            psi = -np.mean(score*d['X_tilde'])
            g = -np.mean(score*d['X'])
            assert np.isclose(psi, g + np.mean(score*d['phi']))
        before = ns['RESULTS_CSV'].read_bytes()
        exec(nb.cells[12].source, ns)
        assert ns['RESULTS_CSV'].read_bytes() == before

report = {
    'notebook_structure_and_code_syntax': 'passed',
    'math_delimiters': 'passed',
    'LP_sign_regression': 'passed',
    'LP_objective_and_translation_cases': 24,
    'smoke_grid': {'n': [32, 48], 'Q': 2, 'epochs': 2},
    'smoke_loop_summary_four_figures_and_restart': 'passed',
    'residual_reparameterization_and_score_identity': 'passed',
    'production_monte_carlo_run': False,
}
(root / 'tmp/psi_n_review/validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
