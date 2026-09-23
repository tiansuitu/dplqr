"""Small numerical checks; does not execute the production Monte Carlo."""
from pathlib import Path
import hashlib
import json
import contextlib
import io
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import norm
import torch

import diagnostic_helpers as d


class DiagnosticsTests(unittest.TestCase):
    def test_gaussian_population_formula_and_derivative(self):
        for delta in [-0.8, 0.0, 0.3]:
            for h in [-1.2, 0.0, 0.7]:
                numeric, _ = quad(lambda v: v*(ndtr(delta*v+h)-.5)*norm.pdf(v, scale=.5),
                                  -np.inf, np.inf, epsabs=1e-11)
                self.assertAlmostEqual(numeric, float(d.population_integrand(delta, h)), places=10)
        step = 1e-5
        derivative = (d.population_integrand(step, 0)-d.population_integrand(-step, 0))/(2*step)
        self.assertAlmostEqual(float(derivative), d.J, places=10)
        np.testing.assert_array_equal(d.population_integrand(0, np.array([-3, 0, 3])), [0, 0, 0])

    def test_lp_sign_and_objective(self):
        self.assertAlmostEqual(d.frozen_h_adjustment(np.full(3, 2.), np.ones(3)), 2.)
        rng = np.random.default_rng(22)
        for _ in range(8):
            r, v = rng.normal(size=(2, 23))
            v[0] = 0
            a = d.frozen_h_adjustment(r, v)
            best = min(d.check_loss(r-k*v) for k in r[v != 0]/v[v != 0])
            self.assertAlmostEqual(d.check_loss(r-a*v), best, places=9)

    def test_baseline_matches_original(self):
        base = Path(__file__).parent
        old = json.loads((base / 'psi_n_diagnostic_before_expansion_v2.ipynb').read_text(encoding='utf-8'))
        scope = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for i in [2, 4, 6]:
                exec(''.join(old['cells'][i]['source']), scope)
        c = d.make_config('smoke')
        c.update(batch_size=16)
        scope.update(EPOCHS=c['baseline_epochs'], BATCH_SIZE=c['batch_size'])
        seed = d.seed_for(c, 32, 1)
        original = scope['generate_sample'](32, np.random.default_rng(seed))
        data = d.generate_sample(32, seed)
        for key in ['X', 'Y', 'Z', 'm0', 'phi']:
            np.testing.assert_allclose(data[key], original[key], rtol=0, atol=1e-15)
        torch.set_num_threads(1)
        old_theta, old_m, _ = scope['train_joint'](data['Y'], data['X'], data['Z'], seed)
        baseline = d.fit_stages(data, c, seed)[0][1]
        self.assertAlmostEqual(float(baseline.theta.detach()), old_theta, places=12)
        np.testing.assert_allclose(d.predict_m(baseline, data['Z']), old_m, atol=1e-12)

    def test_signed_bias_and_intervals(self):
        import pandas as pd
        data = pd.DataFrame({'n': [100]*4, 'stage': ['test']*4,
                             'theta_error': [-.2, -.1, .1, .2]})
        row = d.bias_summary(data).iloc[0]
        self.assertAlmostEqual(row.root_n_bias, 0)
        self.assertAlmostEqual(row.root_n_mae, 1.5)
        self.assertGreater(row.ci_high, 0)
        self.assertLess(row.ci_low, 0)
        probability, lo, hi = d.wilson_interval(0, 30)
        self.assertEqual(probability, 0)
        self.assertGreater(hi, .1)
        self.assertGreaterEqual(lo, 0)

    def test_run_restart_decomposition_checkpoints_and_plots(self):
        base = Path(__file__).parent
        original_files = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (base / 'run').rglob('*') if p.is_file()}
        c = d.make_config('smoke')
        scratch = base.parents[1] / 'tmp' / 'psi_n_review'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch, prefix='expansion_test_') as tmp:
            folder, df = d.run_experiment(tmp, c)
            self.assertEqual(len(df), 8)
            self.assertLess(df.decomposition_error.abs().max(), 1e-12)
            np.testing.assert_allclose(df.R_score, df.S-df.E-df['T'], atol=1e-12)
            np.testing.assert_allclose(df.R_theta, np.sqrt(df.n)*df.theta_error+df.W/d.J, atol=1e-12)
            np.testing.assert_allclose(df.S, df.root_n_G+df.root_n_phi_score, atol=1e-12)
            self.assertTrue(np.isfinite(df.select_dtypes('number').to_numpy()).all())
            pivot = df.pivot(index=['n', 'rep'], columns='stage', values='final_training_loss')
            self.assertTrue((pivot.polished <= pivot.baseline+1e-12).all())
            # Gaussian formula implies Taylor correction has the opposite sign to delta.
            self.assertTrue((df['T']*df.theta_error <= 1e-12).all())
            tables = d.summarize(df, c)
            self.assertEqual(len(tables['bias']), 4)
            d.save_summaries(folder, tables)
            self.assertEqual(len(d.plot_diagnostics(tables, c, folder)), 3)
            results = list(folder.rglob('result.json'))
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in results}
            _, again = d.run_experiment(tmp, c)
            self.assertEqual(len(again), 8)
            for p, (content, modified) in before.items():
                self.assertEqual(content, p.read_bytes())
                self.assertEqual(modified, p.stat().st_mtime_ns)
            cp = torch.load(results[0].parent / 'baseline.pt', weights_only=True)
            model = d.JointModel(cp['hidden'])
            model.load_state_dict(cp['state_dict'])
            data = d.generate_sample(cp['n'], cp['seed'])
            row = d.diagnose(model, data, c, cp['n'], cp['rep'], 'baseline', 0)
            saved = json.loads(results[0].read_text())[0]
            for key in ['S', 'E', 'T', 'R_theta']:
                self.assertAlmostEqual(row[key], saved[key], places=11)
            altered = dict(c, learning_rate=2e-3)
            self.assertNotEqual(folder, d.run_directory(tmp, altered))
            manifest = folder / 'manifest.json'
            manifest.write_text('{}')
            with self.assertRaises(ValueError):
                d.initialize_run(tmp, c)
        for p, digest in original_files.items():
            self.assertEqual(digest, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main(verbosity=2)
