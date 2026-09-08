"""Independently verify Experiment 2 outputs without training a network.

Run ``python scripts/verify_outputs.py OUTPUT_DIR`` after the full analysis, or
add ``--smoke`` to audit its two-path smoke output. Selection is recomputed from
prefix minima; no analysis or reporting functions are imported by this auditor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = next(p for p in BASE.parents if (p / 'dqAux.py').is_file())
EXPERIMENT1 = ROOT / 'results/2026-09-08-epoch-trajectory-no-early-stopping'
sys.path.insert(0, str(EXPERIMENT1 / 'scripts'))
import run_trajectory as rt

PATIENCES = (5, 15, 30, 50, 100, 200)
CEILINGS = (25, 50, 100, 200, 500, 1000)
KEYS = ['case', 'replicate', 'epoch']
METRICS = ('selected_epoch', 'stop_epoch', 'theta_error_l2', 'training_check_loss',
           'validation_check_loss', 'test_check_loss', 'm_l2_error', 'q_l2_error',
           'theta_epoch_gap', 'abs_theta_epoch_gap', 'theta_regret',
           'theta_squared_regret', 'q_regret', 'best_validation_loss')
ORACLES = {'theta': 'theta_error_l2', 'q': 'q_l2_error', 'm': 'm_l2_error',
           'test': 'test_check_loss', 'theta_1': 'abs_theta_error_1',
           'theta_2': 'abs_theta_error_2', 'validation': 'stopping_validation_check_loss'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=1e-11, rtol=1e-11):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol,
                               equal_nan=True, err_msg=label)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_decision(losses, patience, cap):
    """Use strict record minima and distance to last record, independently of the selector."""
    values = np.asarray(losses[:cap], dtype=float)
    running_min = np.minimum.accumulate(values)
    is_record = np.r_[True, values[1:] < running_min[:-1]]
    epochs = np.arange(1, cap + 1)
    last_record = np.maximum.accumulate(np.where(is_record, epochs, 0))
    triggers = np.flatnonzero(epochs - last_record >= patience)
    stop = int(triggers[0] + 1) if len(triggers) else cap
    best = int(np.argmin(values[:stop])) + 1
    return dict(stop_epoch=stop, selected_epoch=best, best_validation_epoch=best,
                best_validation_loss=float(values[best - 1]), restore_best=True,
                min_delta=0.0, stopping_reason='early_stopping' if len(triggers) else 'max_epochs')


def verify_dense(output, dense, source, config, expected_pairs):
    require(not dense.duplicated(KEYS).any(), 'Duplicate dense epochs')
    pairs = set(dense[['case', 'replicate']].itertuples(index=False, name=None))
    require(pairs == set(expected_pairs), 'Dense path set differs from intended source paths')
    require(len(dense) == 1000 * len(pairs), 'Dense grid is incomplete')
    require(np.isfinite(dense.select_dtypes(include='number').to_numpy()).all(), 'Nonfinite dense metrics')
    error = dense[['theta_hat_1', 'theta_hat_2']].to_numpy() - np.array([1., -1.])
    close(dense.theta_error_l2, np.linalg.norm(error, axis=1), 'Theta vector error')
    for j in (1, 2):
        require(dense[f'theta_true_{j}'].eq((1., -1.)[j-1]).all(), 'Wrong coefficient truth')
        close(dense[f'theta_error_{j}'], error[:, j-1], 'Signed coefficient error')
        close(dense[f'abs_theta_error_{j}'], abs(error[:, j-1]), 'Absolute coefficient error')
    close(dense.m_l2_error**2, dense.nonlinear_error_mse, 'Nuisance absolute L2')
    close(dense.q_l2_error**2, dense.q_mse, 'Quantile absolute L2')
    close(dense.q_mse, dense.linear_error_mse + dense.nonlinear_error_mse + 2*dense.cross_error_moment,
          'Prediction decomposition', atol=1e-9)
    counts = dict(n=config.n, n_train=int(.8*config.n), n_validation=int(.2*config.n),
                  n_test=config.test_size, n_eval=config.eval_size, tau=config.tau,
                  learning_rate=config.learning_rate, model_instance=1, fit_calls=1)
    for name, expected in counts.items():
        require(dense[name].eq(expected).all(), f'Wrong fixed configuration {name}')
    matched_rows = archives = 0
    for (case, replicate), part in dense.groupby(['case', 'replicate'], sort=True):
        part = part.sort_values('epoch').reset_index(drop=True)
        require(part.epoch.tolist() == list(range(1, 1001)), 'Missing integer epoch')
        require(part.optimizer_steps.eq(part.epoch * 7).all(), 'Optimizer steps are discontinuous')
        folder = output / 'dense-trajectories' / f'case_{case}_rep_{replicate:04d}'
        prior = source / 'replicates' / folder.name
        audit, old_audit = read_json(folder / 'dense_completion.json'), read_json(prior / 'completion.json')
        require(audit['status'] == 'complete' and audit['epochs_completed'] == 1000, 'Path incomplete')
        require(audit['fit_calls'] == 1 and audit['model_instances'] == 1 and audit['optimizer_steps'] == 7000,
                'Path changed its model, fit, or optimizer')
        for flag in ('early_stopping', 'validation_selection', 'clipping'):
            require(audit[flag] is False, 'Training uses ' + flag)
        for flag in ('observation_rng_unchanged', 'observation_parameters_unchanged'):
            require(audit[flag] is True, 'Dense observer changed trajectory')
        require(audit['checkpoint_epochs'] == list(range(1,1001)), 'Not every epoch was observed')
        require(digest(folder / 'trajectory.csv') == audit['trajectory_sha256'], 'Dense CSV changed')
        pd.testing.assert_frame_equal(part, read_csv(folder / 'trajectory.csv'), check_exact=True)
        old = read_csv(prior / 'trajectory.csv')
        overlap = part.set_index('epoch').loc[old.epoch].reset_index()[old.columns]
        pd.testing.assert_frame_equal(overlap, old.reset_index(drop=True), check_exact=True)
        matched_rows += len(old)
        require(read_json(folder / 'data_manifest.json') == read_json(prior / 'data_manifest.json'),
                'Data, seed, or split hashes differ from Experiment 1')
        for key in ('initial_state_sha256', 'final_state_sha256', 'optimizer'):
            require(audit[key] == old_audit[key], 'Original path mismatch: ' + key)
        seeds = rt.seed_scheme(config, int(case), int(replicate))
        for key, value in seeds.items():
            require(part[key].eq(value).all(), 'Seed mismatch: ' + key)
        states = part.set_index('epoch').state_sha256
        old_checkpoints = {row['epoch']: row for row in old_audit['checkpoints']}
        require(set(old_checkpoints) == {row['epoch'] for row in audit['checkpoints']}, 'Archived checkpoint sets differ')
        for checkpoint in audit['checkpoints']:
            epoch = checkpoint['epoch']
            original = old_checkpoints[epoch]
            require(digest(folder / checkpoint['file']) == checkpoint['sha256'], 'Replayed archive hash mismatch')
            require(digest(prior / original['file']) == original['sha256'], 'Original archive hash mismatch')
            require(checkpoint['state_sha256'] == states.loc[epoch] == original['state_sha256'],
                    'Replayed weight state differs from original')
            previous = states.loc[epoch-1] if epoch > 1 else audit['initial_state_sha256']
            require(checkpoint['previous_checkpoint_state_sha256'] == previous, 'Incorrect preceding dense state')
            archives += 1
        require(states.loc[1000] == audit['final_state_sha256'], 'Final weights were restored or replaced')
    return dict(paths=len(pairs), dense_rows=len(dense), original_rows_matched_exactly=matched_rows,
                original_state_fingerprints_matched=archives, data_manifests_identical=True,
                every_epoch_observed=True, uninterrupted_fit_and_optimizer=True)


def verify_selection(raw, oracles, dense):
    require(np.isfinite(raw.select_dtypes(include='number').to_numpy()).all(), 'Nonfinite selected metrics')
    require(not raw.duplicated(['case','replicate','analysis','rule_id']).any(), 'Duplicate selected rules')
    require(not oracles.duplicated(['case','replicate']).any(), 'Duplicate oracle paths')
    rules = [('patience',f'patience_{p}',p,1000) for p in PATIENCES]
    rules += [('max_epochs',f'max_epochs_{cap}',15,cap) for cap in CEILINGS]
    rules += [('reference','no_early_stopping',0,1000)]
    expected_rule_keys = {(a,r,p,c) for a,r,p,c in rules}
    require(len(raw) == dense[['case','replicate']].drop_duplicates().shape[0]*13, 'Incorrect number of selected rows')
    plateau_pairs = 0
    for (case, replicate), path in dense.groupby(['case','replicate'], sort=True):
        path = path.sort_values('epoch').reset_index(drop=True)
        group = raw.loc[raw.case.eq(case) & raw.replicate.eq(replicate)]
        require(set(group[['analysis','rule_id','patience','max_epochs']].itertuples(index=False, name=None)) == expected_rule_keys,
                'Stopping-rule design is incomplete')
        oracle = oracles.loc[oracles.case.eq(case) & oracles.replicate.eq(replicate)]
        require(len(oracle) == 1, 'Oracle absent or duplicated')
        oracle = oracle.iloc[0]
        for name, metric in ORACLES.items():
            index = int(np.argmin(path[metric].to_numpy()))
            require(oracle['e_'+name] == index+1, 'Oracle is not earliest global minimizer: '+name)
            close(oracle['oracle_'+metric], path.iloc[index][metric], 'Oracle minimum '+metric, atol=0, rtol=0)
        for row in group.itertuples(index=False):
            decision = independent_decision(path.stopping_validation_check_loss.to_numpy(), int(row.patience), int(row.max_epochs)) if row.analysis != 'reference' else dict(
                stop_epoch=1000, selected_epoch=1000, best_validation_epoch=int(np.argmin(path.stopping_validation_check_loss))+1,
                best_validation_loss=float(path.stopping_validation_check_loss.min()), restore_best=False,
                min_delta=0., stopping_reason='max_epochs')
            values = row._asdict()
            for key, expected in decision.items():
                require(values[key] == expected, f'Incorrect stopping decision {case}/{replicate}/{row.rule_id}/{key}')
            selected = path.iloc[row.selected_epoch-1]
            for name in path.columns:
                target = {'state_sha256':'selected_state_sha256', 'validation_check_loss':'empirical_validation_check_loss'}.get(name,name)
                require(values[target] == selected[name], 'Selected estimate did not come from shared path: '+target)
            require(row.training_check_loss == selected.train_check_loss, 'Wrong training loss alias')
            require(row.validation_check_loss == selected.stopping_validation_check_loss, 'Wrong stopping loss alias')
            for name in ORACLES:
                require(values['e_'+name] == oracle['e_'+name], 'Selected row has inconsistent oracle')
            close(row.theta_epoch_gap, row.selected_epoch-oracle.e_theta, 'Epoch gap', atol=0, rtol=0)
            close(row.abs_theta_epoch_gap, abs(row.theta_epoch_gap), 'Absolute epoch gap', atol=0, rtol=0)
            close(row.theta_regret, row.theta_error_l2-oracle.oracle_theta_error_l2, 'Theta regret')
            close(row.theta_squared_regret, row.theta_error_l2**2-oracle.oracle_theta_error_l2**2, 'Squared theta regret')
            close(row.q_regret, row.q_l2_error-oracle.oracle_q_l2_error, 'Prediction regret')
        caps = group.loc[group.analysis.eq('max_epochs')].sort_values('max_epochs')
        for row in caps.itertuples():
            if row.stopping_reason == 'early_stopping':
                later = caps.loc[caps.max_epochs.gt(row.max_epochs)]
                require(later.selected_state_sha256.eq(row.selected_state_sha256).all(), 'Epoch-cap plateau changed weights')
                require(later.selected_epoch.eq(row.selected_epoch).all() and later.stop_epoch.eq(row.stop_epoch).all(),
                        'Epoch cap changed an already-triggered stopping rule')
                plateau_pairs += len(later)
    return dict(selected_rows=len(raw), independent_prefix_minimum_decisions_match=True,
                shared_selected_metrics_and_weights_match=True, global_oracle_minima_verified=True,
                regrets_verified=True, unchanged_cap_plateau_pairs=plateau_pairs)


def stats(values):
    a = np.asarray(values,dtype=float)
    sd = a.std(ddof=1) if len(a)>1 else np.nan
    return dict(mean=a.mean(),median=np.median(a),sd=sd,mcse=sd/math.sqrt(len(a)),
                q25=np.quantile(a,.25),q75=np.quantile(a,.75))


def verify_summaries(output, raw, oracles):
    summaries = pd.concat([read_csv(output/f'monte_carlo_summary_{name}.csv')
                           for name in ('patience','max_epochs','reference')],ignore_index=True)
    keys = ['case','analysis','rule_id','patience','max_epochs']
    groups = raw.groupby(keys,sort=True)
    require(len(summaries) == len(groups) and not summaries.duplicated(keys).any(), 'Summary grid incorrect')
    for key, part in groups:
        mask = np.ones(len(summaries),dtype=bool)
        for name,value in zip(keys,key):
            mask &= summaries[name].eq(value).to_numpy()
        row = summaries.loc[mask].iloc[0]
        require(row.repetitions == len(part), 'Monte Carlo repetitions miscounted')
        for metric in METRICS:
            for name,value in stats(part[metric]).items():
                close(row[f'{name}_{metric}'],value,'MC summary '+metric+'/'+name)
        for j in (1,2):
            theta = part[f'theta_hat_{j}'].to_numpy()
            error = theta-(1.,-1.)[j-1]
            expected = dict(true=(1.,-1.)[j-1],mean=theta.mean(),bias=error.mean(),
                            sd=theta.std(ddof=1),bias_mcse=theta.std(ddof=1)/math.sqrt(len(theta)),
                            rmse=np.sqrt(np.mean(error**2)))
            for name,value in expected.items():
                close(row[f'theta_{j}_{name}'],value,'Coefficient MC '+name)
        close(row.theta_vector_rmse,np.sqrt(np.mean(part.theta_error_l2**2)),'Vector RMSE')
        gap = part.theta_epoch_gap.to_numpy()
        proportions = dict(early_stopping=part.stopping_reason.eq('early_stopping').mean(),
                           max_epochs=part.stopping_reason.eq('max_epochs').mean(),
                           before_theta_oracle=np.mean(gap<0),equal_theta_oracle=np.mean(gap==0),
                           after_theta_oracle=np.mean(gap>0),within_10_epochs=np.mean(abs(gap)<=10),
                           within_25_epochs=np.mean(abs(gap)<=25))
        for name,value in proportions.items():
            close(row['proportion_'+name],value,'MC proportion '+name)
    oracle_summary = read_csv(output/'oracle_epoch_summary.csv')
    require(len(oracle_summary) == oracles.case.nunique(), 'Oracle MC case count incorrect')
    for case, part in oracles.groupby('case'):
        row = oracle_summary.loc[oracle_summary.case.eq(case)].iloc[0]
        for metric in ['e_theta','e_q','e_m','e_test',*[name for name in part if name.startswith('oracle_')]]:
            for name,value in stats(part[metric]).items():
                close(row[f'{name}_{metric}'],value,'Oracle MC '+metric+'/'+name)
    return dict(stopping_rule_summary_rows=len(summaries),oracle_summary_rows=len(oracle_summary),
                bias_sd_rmse_epoch_statistics_and_probabilities_recomputed=True)


def verify_reloaded_criterion(output, dense, config):
    """Reconstruct three fixed archived epochs per case and call torchtuples' scorer itself."""
    from torchtuples import Model
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checks = []
    for case in sorted(dense.case.unique()):
        replicate = int(dense.loc[dense.case.eq(case),'replicate'].min())
        data,seeds,_ = rt.make_data(config,int(case),replicate)
        scaler,inputs = rt.prepare_inputs(data)
        folder = output/'dense-trajectories'/f'case_{case}_rep_{replicate:04d}'
        manifest = read_json(folder/'data_manifest.json')
        require(manifest['arrays_sha256'] == rt.array_hash(data) and manifest['seeds'] == seeds,
                'Reconstructed data differs from Experiment 1')
        net = rt.dqNetSparse(2,8,torch.zeros((1,2),dtype=torch.float32),
                            [config.depth,config.width],sparseRatio=.5)
        model = Model(net,rt.checkLoss(tau=config.tau),device='cpu')
        target = torch.tensor(data['y_validation'][:,None],dtype=torch.float32)
        for epoch in (1,100,1000):
            state = torch.load(folder/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
            net.load_state_dict(state['model_state_dict'])
            net.eval()
            row = dense.loc[dense.case.eq(case) & dense.replicate.eq(replicate) & dense.epoch.eq(epoch)].iloc[0]
            require(rt.state_hash(net.state_dict()) == row.state_sha256,'Loaded state does not match dense row')
            close(state['scaler_mean'].numpy(),scaler.mean_,'Scaler mean',atol=0,rtol=0)
            close(state['scaler_scale'].numpy(),scaler.scale_,'Scaler scale',atol=0,rtol=0)
            require(all(int(item['step']) == 7*epoch for item in state['optimizer_state_dict']['state'].values()),
                    'Saved Adam steps are discontinuous')
            actual = model.score_in_batches(inputs['validation'],target,batch_size=128)['loss']
            close(actual,row.stopping_validation_check_loss,'Actual torchtuples validation criterion',atol=0,rtol=0)
            # torchtuples' scorer switches the model back to train mode on exit.
            net.eval()
            theta = net.linLinear.weight.detach().numpy().ravel().astype(float)
            close(theta,row[['theta_hat_1','theta_hat_2']].to_numpy(float),'Loaded coefficients',atol=0,rtol=0)
            with torch.inference_mode():
                x,z = inputs['eval']
                nuisance = net(torch.zeros_like(x),z).numpy().ravel().astype(float)
                merror = nuisance-data['m_true_eval']
                qerror = data['x_eval']@(theta-np.array([1.,-1.]))+merror
                close(np.sqrt(np.mean(merror**2)),row.m_l2_error,'Loaded nuisance L2')
                close(np.sqrt(np.mean(qerror**2)),row.q_l2_error,'Loaded quantile L2')
            checks.append(dict(case=int(case),replicate=replicate,epoch=epoch,
                               torchtuples_validation_criterion_exact=True,saved_metrics_reconstructed=True))
    return dict(checkpoints_reconstructed=len(checks),checks=checks)


def verify(output, smoke=False):
    output = Path(output).resolve()
    target = output/'smoke-test' if smoke else output
    started = time.perf_counter()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),training_performed=False,
                  auditor_sha256=digest(__file__),stage='smoke' if smoke else 'full_analysis')
    try:
        metadata = read_json(output/('smoke_run_config.json' if smoke else 'run_config.json'))
        require(metadata['status'] == 'complete','Run has not completed')
        source = Path(metadata['source_run'])
        source_metadata = read_json(source/'run_config.json')
        config = rt.Config(**source_metadata['configuration'])
        pairs = list(itertools.product(config.cases,range(1,config.repetitions+1)))
        if smoke:
            pairs = pairs[:2]
        for path,expected in source_metadata['identity']['sources'].items():
            require(digest(ROOT/path) == expected,'Experiment 1 source changed: '+path)
        for path,expected in metadata['identity']['instrumentation'].items():
            require(digest(ROOT/path) == expected,'Dense instrumentation changed: '+path)
        require(digest(source/'run_config.json') == metadata['identity']['source_run_config_sha256'],
                'Source run configuration changed')
        require(digest(source/'raw_epoch_trajectory.csv') == metadata['identity']['source_raw_sha256'],
                'Original raw checkpoints changed')
        dense,raw,oracles = [read_csv(target/name) for name in
                            ('dense_epoch_trajectory.csv','early_stopping_results.csv','oracle_epochs.csv')]
        result['dense'] = verify_dense(output,dense,source,config,pairs)
        result['selection'] = verify_selection(raw,oracles,dense)
        result['summaries'] = {'status':'not_requested_for_smoke'} if smoke else verify_summaries(output,raw,oracles)
        result['reloaded_metrics'] = verify_reloaded_criterion(output,dense,config)
        result.update(status='passed',elapsed_seconds=time.perf_counter()-started)
    except BaseException as error:
        result.update(status='failed',error=repr(error),elapsed_seconds=time.perf_counter()-started)
        rt.atomic_json(result,target/'verification.json')
        raise
    rt.atomic_json(result,target/'verification.json')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir',type=Path,nargs='?',default=BASE)
    parser.add_argument('--smoke',action='store_true')
    arguments = parser.parse_args()
    verify(arguments.output_dir,arguments.smoke)
