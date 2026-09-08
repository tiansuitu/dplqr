"""Paired early-stopping comparisons on fully observed Experiment 1 trajectories."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'dqAux.py').is_file())
EXPERIMENT1 = ROOT / 'results/2026-09-08-epoch-trajectory-no-early-stopping'
sys.path.insert(0, str(EXPERIMENT1 / 'scripts'))
import run_trajectory as original
from record_dense import fit_dense
sys.path.insert(0, str(HERE))
from stopping_rules import select_stopping_epoch

PATIENCES = (5, 15, 30, 50, 100, 200)
CEILINGS = (25, 50, 100, 200, 500, 1000)
KEYS = ['case', 'replicate', 'epoch']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(source):
    source = Path(source).resolve()
    meta = json.loads((source / 'run_config.json').read_text(encoding='utf-8'))
    if meta['status'] != 'complete':
        raise ValueError('Experiment 1 source must be a completed run')
    settings = dict(meta['configuration'])
    if settings['max_epochs'] != 1000 or settings['cases'] != [1, 2, 3]:
        raise ValueError('This analysis requires all three cases through epoch 1000')
    # Ensure we really use the same scientific code and numerical environment.
    for name, expected in meta['identity']['sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Experiment 1 scientific source changed: ' + name)
    cfg = original.Config(**{**settings, 'cases': tuple(settings['cases']),
                             'checkpoints': tuple(range(1, 1001))})
    actual_identity = original.source_identity(cfg)
    if actual_identity['science_sha256'] != meta['identity']['science_sha256']:
        raise ValueError('The original DGP function definitions changed')
    if actual_identity['versions'] != meta['identity']['versions']:
        raise ValueError('Use the same library versions as the source trajectories')
    raw = pd.read_csv(source / 'raw_epoch_trajectory.csv', float_precision='round_trip')
    if raw.duplicated(KEYS).any():
        raise ValueError('Duplicate source checkpoint rows')
    wanted = set(itertools.product(cfg.cases, range(1, cfg.repetitions + 1)))
    if set(map(tuple, raw[['case', 'replicate']].drop_duplicates().to_numpy())) != wanted:
        raise ValueError('Source run has missing case-replicate paths')
    return cfg, raw, meta


def validate_dense_frame(frame):
    assert frame.epoch.tolist() == list(range(1, 1001))
    assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
    assert frame.fit_calls.eq(1).all() and frame.model_instance.eq(1).all()
    assert (frame.optimizer_steps.to_numpy() == 7 * frame.epoch.to_numpy()).all()
    assert frame.state_sha256.str.len().eq(64).all()


def compare_original(frame, source, case, replicate, audit):
    source_path = Path(source) / 'replicates' / f'case_{case}_rep_{replicate:04d}'
    old = pd.read_csv(source_path / 'trajectory.csv', float_precision='round_trip')
    overlap = frame.set_index('epoch').loc[old.epoch].reset_index()[old.columns]
    pd.testing.assert_frame_equal(overlap, old.reset_index(drop=True), check_exact=True)
    old_audit = json.loads((source_path / 'completion.json').read_text())
    assert audit['initial_state_sha256'] == old_audit['initial_state_sha256']
    assert audit['final_state_sha256'] == old_audit['final_state_sha256']
    for checkpoint in old_audit['checkpoints']:
        assert digest(source_path / checkpoint['file']) == checkpoint['sha256']
        actual = frame.loc[frame.epoch.eq(checkpoint['epoch']), 'state_sha256'].item()
        assert actual == checkpoint['state_sha256']
    return dict(status='passed', original_checkpoint_rows_matched_exactly=len(old),
                original_weight_states_matched=len(old_audit['checkpoints']),
                initial_and_final_states_identical=True)


def dense_job(settings, source, output, case, replicate):
    config = original.Config(**{**settings, 'cases': tuple(settings['cases']),
                                'checkpoints': tuple(settings['checkpoints'])})
    directory = Path(output) / 'dense-trajectories' / f'case_{case}_rep_{replicate:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'dense_completion.json').exists():
        audit = json.loads((directory / 'dense_completion.json').read_text())
        assert digest(directory / 'trajectory.csv') == audit['trajectory_sha256']
        frame = pd.read_csv(directory / 'trajectory.csv', float_precision='round_trip')
        validate_dense_frame(frame)
        comparison = compare_original(frame, source, case, replicate, audit)
        return dict(case=case, replicate=replicate, reused=True, seconds=audit['elapsed_seconds'], **comparison)
    # ONE replay for this original case/replicate, shared by every stopping rule.
    frame, audit = fit_dense(config, case, replicate, directory)
    validate_dense_frame(frame)
    comparison = compare_original(frame, source, case, replicate, audit)
    audit.update(comparison=comparison, source_run=str(source),
                 trajectory_sha256=digest(directory / 'trajectory.csv'))
    original.atomic_json(audit, directory / 'dense_completion.json')
    return dict(case=case, replicate=replicate, reused=False, seconds=audit['elapsed_seconds'], **comparison)


def oracle_row(path):
    record = {key: path.iloc[0][key] for key in ('case', 'case_name', 'replicate', 'n', 'tau')}
    for name, metric in [('theta', 'theta_error_l2'), ('q', 'q_l2_error'),
                         ('m', 'm_l2_error'), ('test', 'test_check_loss'),
                         ('theta_1', 'abs_theta_error_1'), ('theta_2', 'abs_theta_error_2'),
                         ('validation', 'stopping_validation_check_loss')]:
        best = path.loc[path[metric].idxmin()]
        record['e_' + name] = int(best.epoch)
        record['oracle_' + metric] = float(best[metric])
    return record


def analyze_paths(dense):
    rows, oracles = [], []
    for (case, replicate), path in dense.groupby(['case', 'replicate'], sort=True):
        path = path.sort_values('epoch').reset_index(drop=True)
        validate_dense_frame(path)
        # The decision function's ONLY data input is validation loss. Oracle and
        # truth columns are inaccessible to it, and computed separately below.
        losses = path.stopping_validation_check_loss.to_numpy()
        rules = [('patience', f'patience_{p}', p, 1000) for p in PATIENCES]
        rules += [('max_epochs', f'max_epochs_{cap}', 15, cap) for cap in CEILINGS]
        decisions = [(analysis, name, p, cap, select_stopping_epoch(losses, p, cap))
                     for analysis, name, p, cap in rules]
        best_index = int(np.argmin(losses))
        decisions.append(('reference', 'no_early_stopping', 0, 1000,
                          dict(stop_epoch=1000, selected_epoch=1000, stopping_reason='max_epochs',
                               best_validation_epoch=best_index + 1,
                               best_validation_loss=float(losses[best_index]), restore_best=False, min_delta=0.0)))
        oracle = oracle_row(path)
        oracles.append(oracle)
        for analysis, rule_id, patience, cap, decision in decisions:
            selected = path.iloc[decision['selected_epoch'] - 1].to_dict()
            selected['empirical_validation_check_loss'] = selected['validation_check_loss']
            selected['validation_check_loss'] = selected['stopping_validation_check_loss']
            selected['training_check_loss'] = selected['train_check_loss']
            selected['selected_state_sha256'] = selected.pop('state_sha256')
            selected.update({'analysis': analysis, 'rule_id': rule_id, 'patience': patience, 'max_epochs': cap,
                             **decision, **{k: v for k, v in oracle.items() if k not in selected}})
            selected.update(theta_epoch_gap=decision['selected_epoch'] - oracle['e_theta'],
                            abs_theta_epoch_gap=abs(decision['selected_epoch'] - oracle['e_theta']),
                            theta_regret=selected['theta_error_l2'] - oracle['oracle_theta_error_l2'],
                            theta_squared_regret=selected['theta_error_l2']**2 - oracle['oracle_theta_error_l2']**2,
                            q_regret=selected['q_l2_error'] - oracle['oracle_q_l2_error'])
            assert selected['theta_regret'] >= 0 and selected['q_regret'] >= 0
            rows.append(selected)
    return pd.DataFrame(rows), pd.DataFrame(oracles)


def verify_rule_pairing(raw, dense):
    """Validate selected quantities, restored states and cap plateaus without refitting."""
    reference = dense.set_index(KEYS)
    for _, row in raw.iterrows():
        selected = reference.loc[(row.case, row.replicate, row.selected_epoch)]
        assert row.selected_state_sha256 == selected.state_sha256
        for name in ['theta_hat_1', 'theta_hat_2', 'theta_error_l2', 'q_l2_error', 'm_l2_error', 'test_check_loss']:
            assert row[name] == selected[name]
        assert row.selected_epoch <= row.stop_epoch <= row.max_epochs
        if row.restore_best:
            assert row.selected_epoch == row.best_validation_epoch
            assert row.validation_check_loss == row.best_validation_loss
    plateau_checks = 0
    for _, part in raw[raw.analysis.eq('max_epochs')].groupby(['case', 'replicate']):
        part = part.sort_values('max_epochs')
        for position in range(len(part) - 1):
            current = part.iloc[position]
            if current.stopping_reason == 'early_stopping':
                later = part.iloc[position + 1:]
                assert later.stop_epoch.eq(current.stop_epoch).all()
                assert later.selected_epoch.eq(current.selected_epoch).all()
                assert later.selected_state_sha256.eq(current.selected_state_sha256).all()
                plateau_checks += len(later)
    return dict(status='passed', selected_rows_checked=len(raw), plateau_pairs_checked=plateau_checks,
                every_rule_uses_same_path=True, oracle_inputs_excluded_from_selector=True)


def run(args):
    source = args.source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config, old_raw, source_meta = load_source(source)
    if config.n != 1000:
        raise ValueError('This diagnostic currently preserves the original n=1000 design')
    pairs = sorted(set(map(tuple, old_raw[['case', 'replicate']].to_numpy())))
    if args.stage == 'smoke':
        pairs = pairs[:2]
    sources = [Path(__file__), HERE / 'stopping_rules.py', EXPERIMENT1 / 'scripts/record_dense.py']
    identity = dict(source=str(source), source_run_config_sha256=digest(source / 'run_config.json'),
                    source_raw_sha256=digest(source / 'raw_epoch_trajectory.csv'),
                    configuration=asdict(config),
                    instrumentation={str(p.relative_to(ROOT)): digest(p) for p in sources})
    identity = json.loads(json.dumps(identity))
    cache_path = output / 'dense_identity.json'
    if cache_path.exists():
        if json.loads(cache_path.read_text()) != identity:
            raise ValueError('Inputs or dense observer changed. Use a fresh output directory.')
    else:
        original.atomic_json(identity, cache_path)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='running', stage=args.stage,
                    source_run=str(source), source_configuration=source_meta['configuration'],
                    dense_epochs=list(range(1, 1001)), repetitions_per_case=config.repetitions,
                    expected_paths=len(pairs), completed_paths=0, workers=args.workers,
                    patience_grid=list(PATIENCES), max_epoch_grid=list(CEILINGS), default_patience=15,
                    early_stopping_convention='Strict decrease, min_delta=0, earliest tied best, restore best even at cap',
                    validation_criterion='Unweighted mean of sequential float32 batch check losses, batch size128',
                    retrospective=True, independently_refit_per_rule=False,
                    trajectories_sufficient_initially=False,
                    extension='Deterministic replay once per original replicate with every-epoch observer; original checkpoints matched',
                    identity=identity)
    meta_path = output / ('smoke_run_config.json' if args.stage == 'smoke' else 'run_config.json')
    original.atomic_json(metadata, meta_path)
    started = time.perf_counter()
    results = []
    if args.stage != 'analyze':
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(dense_job, asdict(config), str(source), str(output), int(c), int(r)) for c, r in pairs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                metadata.update(completed_paths=len(results), elapsed_seconds=time.perf_counter() - started)
                original.atomic_json(metadata, meta_path)
                print(f"Verified {len(results)}/{len(pairs)} paths: case={result['case']} replicate={result['replicate']} cached={result['reused']}", flush=True)
    frames = []
    for case, replicate in pairs:
        directory = output / 'dense-trajectories' / f'case_{case}_rep_{replicate:04d}'
        audit = json.loads((directory / 'dense_completion.json').read_text())
        assert digest(directory / 'trajectory.csv') == audit['trajectory_sha256']
        frame = pd.read_csv(directory / 'trajectory.csv', float_precision='round_trip')
        validate_dense_frame(frame)
        compare_original(frame, source, case, replicate, audit)
        frames.append(frame)
    dense = pd.concat(frames, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    raw, oracles = analyze_paths(dense)
    validation = verify_rule_pairing(raw, dense)
    target = output / 'smoke-test' if args.stage == 'smoke' else output
    target.mkdir(parents=True, exist_ok=True)
    original.atomic_csv(dense, target / 'dense_epoch_trajectory.csv')
    original.atomic_csv(raw, target / 'early_stopping_results.csv')
    original.atomic_csv(oracles, target / 'oracle_epochs.csv')
    original.atomic_json(validation, target / 'pairing_validation.json')
    if args.stage != 'smoke':
        from reporting import write_reports
        write_reports(raw, oracles, target, metadata)
    metadata.update(status='complete', completed_paths=len(pairs), dense_rows=len(dense),
                    selected_rows=len(raw), elapsed_seconds=time.perf_counter() - started)
    original.atomic_json(metadata, meta_path)
    print(f"Complete: {len(pairs)} paths, {len(dense)} epoch rows, {len(raw)} stopping-rule rows; {metadata['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=EXPERIMENT1 / 'pilot-10')
    parser.add_argument('--output-dir', type=Path, default=BASE)
    parser.add_argument('--stage', choices=['smoke', 'all', 'analyze'], default='all')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        run(args)
    except BaseException as error:
        meta_path = args.output_dir / ('smoke_run_config.json' if args.stage == 'smoke' else 'run_config.json')
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            if metadata.get('status') == 'running':
                metadata.update(status='failed', error=repr(error))
                original.atomic_json(metadata, meta_path)
        raise


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
