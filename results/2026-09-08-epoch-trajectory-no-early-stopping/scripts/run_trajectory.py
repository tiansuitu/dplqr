"""Continuous, unmodified DPLQR training paths; no validation selection or stopping."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.preprocessing import StandardScaler
import torch
from torchtuples import Model
import torchtuples as tt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'dqAux.py').is_file())
sys.path.insert(0, str(ROOT))
from dqAux import dqNetSparse, checkLoss, checkErrorMean

PARENT_NOTEBOOK = ROOT / 'results/2026-09-04-homoscedastic-simulation/simulate_homoscedastic.ipynb'
FIRST_PASS = PARENT_NOTEBOOK.parent / 'first-pass-run'
THETA = np.array([1.0, -1.0])
CASE_NAMES = {1: 'linear', 2: 'additive', 3: 'deep'}
CHECKPOINTS = (1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000)
REUSED = {'seed_all', 'nonlinear_truth', 'generate_covariates', 'generate_dataset', 'tensor_pair'}
SCIENCE_NODES = {}
for cell in json.loads(PARENT_NOTEBOOK.read_text(encoding='utf-8'))['cells']:
    if cell['cell_type'] == 'code':
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.FunctionDef) and node.name in REUSED:
                if node.name in SCIENCE_NODES:
                    raise RuntimeError('Duplicate original definition: ' + node.name)
                SCIENCE_NODES[node.name] = node
if set(SCIENCE_NODES) != REUSED:
    raise RuntimeError('Original DGP definitions are missing')
for node in SCIENCE_NODES.values():
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(PARENT_NOTEBOOK), 'exec'), globals())


@dataclass(frozen=True)
class Config:
    repetitions: int = 100
    cases: tuple[int, ...] = (1, 2, 3)
    n: int = 1000
    tau: float = 0.5
    max_epochs: int = 1000
    checkpoints: tuple[int, ...] = CHECKPOINTS
    test_size: int = 10000
    eval_size: int = 10000
    seed: int = 20260904
    depth: int = 2
    width: int = 32
    batch_size: int = 128
    learning_rate: float = 0.005
    threads: int = 1
    workers: int = 3

    def validate(self):
        for name in ('repetitions', 'max_epochs', 'test_size', 'eval_size', 'depth',
                     'width', 'batch_size', 'threads', 'workers'):
            if getattr(self, name) < 1:
                raise ValueError(name + ' must be positive')
        if self.n < 50 or not 0 < self.tau < 1 or self.seed < 0 or self.learning_rate <= 0:
            raise ValueError('Invalid n, tau, seed or learning rate')
        if not self.cases or len(set(self.cases)) != len(self.cases) or not set(self.cases) <= set(CASE_NAMES):
            raise ValueError('Cases must be unique members of 1,2,3')
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError('Checkpoint epochs must be unique and increasing')
        if not self.checkpoints or min(self.checkpoints) < 1 or max(self.checkpoints) != self.max_epochs:
            raise ValueError('Checkpoints must end at the requested epoch ceiling')


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def atomic_csv(frame, path):
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def array_hash(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        a = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def state_hash(state):
    return array_hash({name: value.detach().cpu().numpy() for name, value in state.items()})


def seed_scheme(config, case, replicate):
    data_seed = config.seed + case * 10_000_000 + config.n * 1000 + replicate
    init_seed = data_seed + int(round(config.tau * 1_000_000))
    return dict(data_seed=data_seed, init_seed=init_seed,
                training_seed=init_seed + 1_000_000_000,
                test_seed=data_seed + 2_000_000_000, eval_seed=data_seed + 3_000_000_000)


def make_data(config, case, replicate):
    seeds = seed_scheme(config, case, replicate)
    rng = np.random.default_rng(seeds['data_seed'])
    x, z, y, _ = generate_dataset(config.n, case, rng)
    order = rng.permutation(config.n)
    tr, va = order[:int(0.8 * config.n)], order[int(0.8 * config.n):]
    xt, zt, yt, _ = generate_dataset(config.test_size, case, np.random.default_rng(seeds['test_seed']))
    xe, ze, _, me = generate_dataset(config.eval_size, case,
                                    np.random.default_rng(seeds['eval_seed']), include_error=False)
    data = dict(x_train=x[tr], z_train=z[tr], y_train=y[tr],
                x_validation=x[va], z_validation=z[va], y_validation=y[va],
                x_test=xt, z_test=zt, y_test=yt, x_eval=xe, z_eval=ze,
                m_true_eval=me + t.ppf(config.tau, df=3))
    manifest = dict(case=case, replicate=replicate, seeds=seeds,
                    arrays_sha256=array_hash(data), shapes={k: list(v.shape) for k, v in data.items()},
                    train_validation_split='800/200 at n=1000; permutation from data_seed',
                    test_and_evaluation='Separate independent streams; fixed within replicate')
    # Verify the unchanged original training sample when a first-pass export exists.
    saved = FIRST_PASS / 'data_for_r' / f'case_{case}_n_{config.n}_rep_{replicate:04d}_train.csv.gz'
    reference = json.loads((FIRST_PASS / 'run_config.json').read_text(encoding='utf-8'))
    if saved.exists() and config.seed == reference['seed']:
        expected = np.column_stack((y[tr], x[tr], z[tr], nonlinear_truth(z[tr], case)))
        observed = pd.read_csv(saved, float_precision='round_trip').to_numpy()
        np.testing.assert_allclose(expected, observed, rtol=0, atol=1e-12)
        manifest['original_training_sample_verified'] = True
    else:
        manifest['original_training_sample_verified'] = False
    return data, seeds, manifest


def prepare_inputs(data):
    scaler = StandardScaler().fit(data['z_train'])
    inputs = {name: tensor_pair(data['x_' + name], scaler.transform(data['z_' + name]))
              for name in ('train', 'validation', 'test', 'eval')}
    return scaler, inputs


def predict_components(net, pair, x_original, theta):
    # The nonlinear branch is obtained with X=0, preserving its interpretation as a function of Z.
    # Float64 assembly exactly follows q(X,Z)=X theta+m(Z), avoiding cancellation from subtracting
    # float32 full predictions. Neither operation changes a parameter or consumes randomness.
    nuisance = net(torch.zeros_like(pair[0]), pair[1]).detach().cpu().numpy().reshape(-1).astype(float)
    prediction = x_original @ theta + nuisance
    direct = net(*pair).detach().cpu().numpy().reshape(-1)
    np.testing.assert_allclose(prediction, direct, rtol=2e-6, atol=1e-5)
    return prediction, nuisance, float(np.max(np.abs(prediction - direct)))


class TrajectoryRecorder(tt.callbacks.Callback):
    """Observe and save checkpoints; this callback never terminates or restores a fit."""
    def __init__(self, config, case, replicate, data, seeds, scaler, inputs, directory=None):
        self.config, self.case, self.replicate = config, case, replicate
        self.data, self.seeds, self.scaler, self.inputs = data, seeds, scaler, inputs
        self.directory = directory
        self.rows = []
        self.checkpoint_manifest = []

    def on_fit_start(self):
        self.epoch, self.updates, self.fit_calls = 0, 0, 1
        self.net_id, self.optimizer_id = id(self.model.net), id(self.model.optimizer.optimizer)
        self.initial_state_sha256 = state_hash(self.model.net.state_dict())
        self.last_state_sha256 = self.initial_state_sha256
        return False

    def on_batch_end(self):
        self.updates += 1
        return False

    def on_epoch_end(self):
        self.epoch += 1
        assert id(self.model.net) == self.net_id and id(self.model.optimizer.optimizer) == self.optimizer_id
        expected = self.epoch * math.ceil(len(self.data['y_train']) / self.config.batch_size)
        actual = int(self.model.optimizer.optimizer.state[self.model.net.linLinear.weight]['step'].item())
        assert self.updates == actual == expected, 'Optimizer step counter discontinuity'
        assert all(group['lr'] == self.config.learning_rate for group in self.model.optimizer.param_groups)
        if self.epoch in self.config.checkpoints:
            self.observe(actual)
        return False

    def observe(self, optimizer_steps):
        net = self.model.net
        random_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        modes = [(module, module.training) for module in net.modules()]
        parameters_before = state_hash(net.state_dict())
        try:
            net.eval()
            with torch.inference_mode():
                theta = net.linLinear.weight.detach().cpu().numpy().reshape(-1).astype(float)
                predictions, nuisance, roundoff = {}, {}, []
                for name, pair in self.inputs.items():
                    predictions[name], nuisance[name], gap = predict_components(
                        net, pair, self.data['x_' + name], theta)
                    roundoff.append(gap)
            error = theta - THETA
            linear_error = self.data['x_eval'] @ error
            nonlinear_error = nuisance['eval'] - self.data['m_true_eval']
            true_q = self.data['x_eval'] @ THETA + self.data['m_true_eval']
            q_error = predictions['eval'] - true_q
            linear_mse, nonlinear_mse, q_mse = [float(np.mean(v**2)) for v in
                                               (linear_error, nonlinear_error, q_error)]
            cross = float(np.mean(linear_error * nonlinear_error))
            residual = q_mse - (linear_mse + nonlinear_mse + 2 * cross)
            if abs(residual) > 1e-10 * (1 + q_mse + linear_mse + nonlinear_mse):
                raise ArithmeticError('Quantile-error decomposition does not balance')
            row = dict(case=self.case, case_name=CASE_NAMES[self.case], replicate=self.replicate,
                       n=self.config.n, n_train=len(self.data['y_train']),
                       n_validation=len(self.data['y_validation']), n_test=self.config.test_size,
                       n_eval=self.config.eval_size, tau=self.config.tau, epoch=self.epoch, **self.seeds,
                       theta_error_l2=float(np.linalg.norm(error)), m_l2_error=math.sqrt(nonlinear_mse),
                       q_l2_error=math.sqrt(q_mse), linear_error_mse=linear_mse,
                       nonlinear_error_mse=nonlinear_mse, cross_error_moment=cross,
                       cancellation_fraction=(-2 * cross / (linear_mse + nonlinear_mse)
                                              if linear_mse + nonlinear_mse > 0 else 0.0),
                       q_mse=q_mse, decomposition_residual=residual,
                       max_float32_prediction_gap=max(roundoff), optimizer_steps=optimizer_steps,
                       learning_rate=self.config.learning_rate, model_instance=1, fit_calls=self.fit_calls)
            for j in (1, 2):
                row.update({f'theta_hat_{j}': float(theta[j-1]), f'theta_true_{j}': float(THETA[j-1]),
                            f'theta_error_{j}': float(error[j-1]), f'abs_theta_error_{j}': float(abs(error[j-1]))})
            for name in ('train', 'validation', 'test'):
                row[name + '_check_loss'] = float(checkErrorMean(
                    predictions[name][:, None], self.data['y_' + name][:, None], tau=self.config.tau))
            if not all(np.isfinite(value) for value in row.values() if isinstance(value, (int, float))):
                raise FloatingPointError('Non-finite checkpoint diagnostic')
            self.rows.append(row)
            if self.directory is not None:
                path = self.directory / f'epoch_{self.epoch:04d}.pt'
                payload = dict(case=self.case, replicate=self.replicate, epoch=self.epoch,
                               model_state_dict=net.state_dict(), optimizer_state_dict=self.model.optimizer.state_dict(),
                               scaler_mean=torch.tensor(self.scaler.mean_), scaler_scale=torch.tensor(self.scaler.scale_),
                               theta_true=torch.tensor(THETA), seeds=self.seeds,
                               optimizer_steps=optimizer_steps, torch_rng_state=torch_before)
                temporary = path.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(path)
                self.checkpoint_manifest.append(dict(epoch=self.epoch, file=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    state_sha256=parameters_before, previous_checkpoint_state_sha256=self.last_state_sha256,
                    optimizer_steps=optimizer_steps))
            self.last_state_sha256 = parameters_before
        finally:
            for module, mode in modes:
                module.training = mode
        assert state_hash(net.state_dict()) == parameters_before, 'Observation modified live parameters'
        assert random.getstate() == random_before, 'Observation consumed Python RNG'
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        assert np.random.get_state()[0] == numpy_before[0] and np.random.get_state()[2:] == numpy_before[2:]
        assert torch.equal(torch.get_rng_state(), torch_before), 'Observation consumed Torch RNG'

    def on_fit_end(self):
        assert self.epoch == self.config.max_epochs, 'Training stopped before its fixed ceiling'
        assert [row['epoch'] for row in self.rows] == list(self.config.checkpoints)
        assert state_hash(self.model.net.state_dict()) == self.last_state_sha256, 'Final weights were restored or changed'
        return False


def fit_one(config, case, replicate, directory=None):
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    data, seeds, data_manifest = make_data(config, case, replicate)
    scaler, inputs = prepare_inputs(data)
    seed_all(seeds['init_seed'])
    net = dqNetSparse(2, 8, torch.zeros((1, 2), dtype=torch.float32),
                      [config.depth, config.width], sparseRatio=0.5)
    net.linLinear.reset_parameters()
    model = Model(net, checkLoss(tau=config.tau), device='cpu')
    model.optimizer.set_lr(config.learning_rate)
    group = model.optimizer.param_groups[0]
    assert group['betas'] == (0.9, 0.99) and group['eps'] == 1e-8 and group['weight_decay'] == 0
    assert model.optimizer.init_args['decoupled_weight_decay'] == 0
    recorder = TrajectoryRecorder(config, case, replicate, data, seeds, scaler, inputs, directory)
    # Separate from initialization: this stream controls minibatch shuffling and original dropout.
    seed_all(seeds['training_seed'])
    started = time.perf_counter()
    model.fit(inputs['train'], torch.tensor(data['y_train'][:, None], dtype=torch.float32),
              batch_size=config.batch_size, epochs=config.max_epochs, callbacks=[recorder], verbose=False,
              shuffle=True, num_workers=0, val_data=None)
    frame = pd.DataFrame(recorder.rows)
    audit = dict(case=case, replicate=replicate, status='complete', epochs_completed=recorder.epoch,
                 fit_calls=recorder.fit_calls, model_instances=1, optimizer_steps=recorder.updates,
                 initial_state_sha256=recorder.initial_state_sha256,
                 final_state_sha256=state_hash(net.state_dict()),
                 elapsed_seconds=time.perf_counter()-started,
                 checkpoint_epochs=frame.epoch.tolist(), checkpoints=recorder.checkpoint_manifest,
                 observation_rng_unchanged=True, observation_parameters_unchanged=True,
                 early_stopping=False, validation_selection=False, clipping=False,
                 optimizer=dict(wrapper='torchtuples.optim.AdamW', underlying='torch.optim.Adam',
                                betas=list(group['betas']), eps=group['eps'], decay=0, lr=group['lr']))
    if directory is not None:
        atomic_json(data_manifest, directory / 'data_manifest.json')
        atomic_csv(frame, directory / 'trajectory.csv')
        atomic_json(audit, directory / 'completion.json')
    return frame, audit


def source_identity(config):
    scientific = {name: ast.dump(node, include_attributes=False) for name, node in SCIENCE_NODES.items()}
    return dict(config={k: v for k, v in asdict(config).items() if k != 'workers'},
                science_sha256=hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest(),
                sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), HERE / 'reporting.py', ROOT / 'dqAux.py')},
                theta_true=THETA.tolist(), versions=dict(python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, torch=str(torch.__version__), torchtuples=tt.__version__))


def read_completed(directory, config, case, replicate):
    audit = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'complete' and audit['epochs_completed'] == config.max_epochs
    assert audit['case'] == case and audit['replicate'] == replicate and audit['fit_calls'] == 1
    for checkpoint in audit['checkpoints']:
        assert hashlib.sha256((directory / checkpoint['file']).read_bytes()).hexdigest() == checkpoint['sha256']
    frame = pd.read_csv(directory / 'trajectory.csv', float_precision='round_trip')
    assert frame.epoch.tolist() == list(config.checkpoints)
    assert frame.case.eq(case).all() and frame.replicate.eq(replicate).all()
    assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
    for name, value in seed_scheme(config, case, replicate).items():
        assert frame[name].eq(value).all()
    return frame, audit


def worker_job(config, case, replicate, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'completion.json').exists():
        frame, audit = read_completed(directory, config, case, replicate)
        return dict(case=case, replicate=replicate, reused=True, elapsed_seconds=audit['elapsed_seconds'])
    # An incomplete replicate restarts from its explicit seeds; no partial path is spliced in.
    _, audit = fit_one(config, case, replicate, directory)
    return dict(case=case, replicate=replicate, reused=False, elapsed_seconds=audit['elapsed_seconds'])


def run(config, output):
    from reporting import write_reports
    config.validate()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'run_config.json'
    identity = json.loads(json.dumps(source_identity(config)))
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        if old['identity'] != identity:
            raise ValueError('Output contains different settings or sources. Choose a new --output-dir.')
    elif (output / 'replicates').exists():
        raise ValueError('Existing replicates have no configuration. Choose a new output directory.')
    total = len(config.cases) * config.repetitions
    started = time.perf_counter()
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running', identity=identity,
                    configuration=asdict(config), intended_design=dict(repetitions=100, cases=[1,2,3],
                    max_epochs=1000, checkpoints=list(CHECKPOINTS), n=1000, tau=0.5),
                    output_dir=str(output), expected_replicates=total,
                    expected_raw_rows=total*len(config.checkpoints), completed_replicates=0,
                    training='One uninterrupted fit and optimizer per replicate; no early stopping or validation data in fit',
                    diagnostics='Unclipped current parameters; dropout disabled only during observation',
                    training_sample='80% estimation sample; remaining20% diagnostic validation only',
                    full_design_requested=config.repetitions >= 100 and config.max_epochs == 1000,
                    cpu_workers=config.workers, threads_per_worker=config.threads)
    atomic_json(metadata, config_path)
    jobs = list(itertools.product(config.cases, range(1, config.repetitions+1)))
    completions = []

    def record(result):
        completions.append(result)
        metadata.update(completed_replicates=len(completions), elapsed_seconds=time.perf_counter()-started,
                        last_completed=dict(case=result['case'], replicate=result['replicate']))
        atomic_json(metadata, config_path)
        atomic_csv(pd.DataFrame(completions), output / 'replicate_timings.csv')
        print(f"Completed {len(completions)}/{total}: case={result['case']} replicate={result['replicate']} "
              f"fit={result['elapsed_seconds']:.1f}s cached={result['reused']} elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)

    try:
        if config.workers == 1:
            for case, replicate in jobs:
                record(worker_job(config, case, replicate, str(output / 'replicates' / f'case_{case}_rep_{replicate:04d}')))
        else:
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
                futures = [executor.submit(worker_job, config, case, replicate,
                           str(output / 'replicates' / f'case_{case}_rep_{replicate:04d}')) for case, replicate in jobs]
                for future in as_completed(futures):
                    record(future.result())
        frames = []
        for case, replicate in jobs:
            frame, _ = read_completed(output / 'replicates' / f'case_{case}_rep_{replicate:04d}', config, case, replicate)
            frames.append(frame)
        raw = pd.concat(frames, ignore_index=True).sort_values(['case','replicate','epoch'])
        assert len(raw) == metadata['expected_raw_rows']
        assert not raw.duplicated(['case','replicate','epoch']).any()
        atomic_csv(raw, output / 'raw_epoch_trajectory.csv')
        metadata.update(status='summarizing', elapsed_seconds=time.perf_counter()-started)
        atomic_json(metadata, config_path)
        write_reports(raw, output, asdict(config))
        metadata.update(status='complete', raw_rows=len(raw), elapsed_seconds=time.perf_counter()-started,
                        completed_replicates=total)
        atomic_json(metadata, config_path)
        print(f"Complete: {len(raw)} rows, {total} continuous paths; {metadata['elapsed_seconds']:.1f}s", flush=True)
        return metadata
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(metadata, config_path)
        raise


def verify_observer_independence(output):
    # This scientific gate is small: same model/optimizer/seeds, only observation frequency changes.
    dense = Config(repetitions=1, cases=(3,), max_epochs=12, checkpoints=tuple(range(1,13)),
                   test_size=1000, eval_size=1000, workers=1)
    sparse = replace(dense, checkpoints=(12,))
    a, audit_a = fit_one(dense, 3, 1)
    b, audit_b = fit_one(sparse, 3, 1)
    assert audit_a['initial_state_sha256'] == audit_b['initial_state_sha256']
    assert audit_a['final_state_sha256'] == audit_b['final_state_sha256']
    assert audit_a['optimizer_steps'] == audit_b['optimizer_steps'] == 12*7
    pd.testing.assert_frame_equal(a.iloc[[-1]].reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
    result = dict(status='passed', final_weights_identical=True, final_checkpoint_metrics_identical=True,
                  dense_checkpoints=12, sparse_checkpoints=1, epochs_per_fit=12, optimizer_steps=84,
                  meaning='Diagnostic frequency and test/evaluation forwards do not alter the training path')
    Path(output).mkdir(parents=True, exist_ok=True)
    atomic_json(result, Path(output) / 'observer_independence.json')
    print(json.dumps(result), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repetitions', type=int, default=100)
    parser.add_argument('--cases', default='1,2,3')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--checkpoints', default=','.join(map(str,CHECKPOINTS)))
    parser.add_argument('--test-size', type=int, default=10000)
    parser.add_argument('--eval-size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260904)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=BASE)
    parser.add_argument('--verify-observer', action='store_true', help='Run the small observation-frequency invariance check first')
    parser.add_argument('--verification-only', action='store_true', help='Run the invariance check without a Monte Carlo run')
    args = parser.parse_args(argv)
    checkpoints = sorted({int(e) for e in args.checkpoints.split(',') if 0 < int(e) <= args.max_epochs} | {args.max_epochs})
    config = Config(repetitions=args.repetitions, cases=tuple(map(int,args.cases.split(','))), n=args.n,
                    tau=args.tau, max_epochs=args.max_epochs, checkpoints=tuple(checkpoints),
                    test_size=args.test_size, eval_size=args.eval_size, seed=args.seed,
                    threads=args.threads, workers=args.workers)
    config.validate()
    return args, config


def main(argv=None):
    args, config = parse_args(argv)
    if args.verify_observer or args.verification_only:
        verify_observer_independence(args.output_dir / 'verification')
    if not args.verification_only:
        return run(config, args.output_dir)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
