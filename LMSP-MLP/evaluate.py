"""
Evaluation script for LMSP-MLP
Results are saved to checkpoints/<exp>/evaluation/ and results/<exp>/test/

Usage:
    python evaluate.py --checkpoint_dir lmsp_mlp_pred24_20260421_xxxxxx
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from models.lmsp_mlp import LMSP_MLP
from data.dataset    import create_dataloaders


# ---------------------------------------------------------------------------
# Efficiency Metrics  (Params / FLOPs / Inference latency)
# ---------------------------------------------------------------------------

def _count_params(model: torch.nn.Module) -> Dict[str, Any]:
    """Count parameters. Reports total in G (1e9) and MB (fp32)."""
    total     = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        'params_total':        total,
        'params_trainable':    trainable,
        'params_M':            float(total / 1e6),    # M for display
        'params_G':            float(total / 1e9),    # G (matches V8 format)
        'params_total_mb_fp32': float(total * 4 / 1024 / 1024),
    }


def _make_benchmark_input(config: dict, device: torch.device,
                          batch_size: int, n_exo: int) -> Dict[str, torch.Tensor]:
    """Build a dummy input batch for benchmarking."""
    L = config.get('input_len', 168)
    return {
        'endo_hist':    torch.zeros(batch_size, L, 3,     device=device),
        'exo_hist':     torch.zeros(batch_size, L, n_exo, device=device),
        'hour':         torch.zeros(batch_size, L,        device=device),
        'month':        torch.ones( batch_size, L,        device=device),
        'is_weekdays':  torch.ones( batch_size, L,        device=device),
        'is_holidays':  torch.zeros(batch_size, L,        device=device),
    }


def _estimate_flops_thop(model: torch.nn.Module,
                         model_input: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Estimate MACs/FLOPs via thop (best-effort). Returns values in M (1e6)."""
    try:
        from thop import profile
    except Exception as e:
        return {'thop_available': False, 'thop_error': f'import failed: {e}',
                'macs': None, 'flops': None, 'flops_M': None}
    try:
        macs, params = profile(model, inputs=(model_input,), verbose=False)
        flops   = float(macs) * 2.0          # 1 MAC ~= 2 FLOPs
        flops_M = flops / 1e6
        return {
            'thop_available': True,
            'macs':           float(macs),
            'flops':          flops,
            'flops_M':        flops_M,
            'thop_params':    int(params) if params is not None else None,
        }
    except Exception as e:
        return {'thop_available': False, 'thop_error': f'profile failed: {e}',
                'macs': None, 'flops': None, 'flops_M': None}


def _benchmark_latency(model: torch.nn.Module,
                       model_input: Dict[str, torch.Tensor],
                       device: torch.device,
                       warmup: int = 20,
                       iters:  int = 100) -> Dict[str, Any]:
    """Forward-only latency benchmark. Results in ms."""
    model.eval()
    batch_size = int(model_input['endo_hist'].shape[0])

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = model(model_input)

    times_ms = []
    with torch.no_grad():
        if device.type == 'cuda':
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            for _ in range(max(1, iters)):
                starter.record()
                _ = model(model_input)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(float(starter.elapsed_time(ender)))
        else:
            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                _ = model(model_input)
                times_ms.append(float((time.perf_counter() - t0) * 1000.0))

    arr  = np.asarray(times_ms, dtype=np.float64)
    mean = float(arr.mean())
    mean_per_sample = float(mean / max(1, batch_size))
    return {
        'device':                  str(device),
        'batch_size':              batch_size,
        'warmup':                  warmup,
        'iters':                   iters,
        'mean_ms_per_batch':       mean,
        'p50_ms_per_batch':        float(np.percentile(arr, 50)),
        'p95_ms_per_batch':        float(np.percentile(arr, 95)),
        'mean_ms_per_sample':      mean_per_sample,
        'throughput_samples_per_s': float(1000.0 / mean_per_sample) if mean_per_sample > 0 else None,
    }


def compute_efficiency_metrics(model: torch.nn.Module,
                               config: dict,
                               device: torch.device,
                               n_exo: int,
                               compute_flops:      bool = True,
                               benchmark_latency:  bool = True,
                               latency_batch_size: int  = 1,
                               latency_warmup:     int  = 20,
                               latency_iters:      int  = 100) -> Dict[str, Any]:
    """Compute Params + FLOPs + Inference latency."""
    eff: Dict[str, Any] = {}
    eff.update(_count_params(model))

    if compute_flops:
        inp = _make_benchmark_input(config, device=device, batch_size=1, n_exo=n_exo)
        eff.update(_estimate_flops_thop(model, inp))

    if benchmark_latency:
        inp = _make_benchmark_input(config, device=device,
                                    batch_size=latency_batch_size, n_exo=n_exo)
        eff['latency'] = _benchmark_latency(model, inp, device,
                                             warmup=latency_warmup, iters=latency_iters)
    return eff


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mae   = float(np.abs(pred - target).mean())
    rmse  = float(np.sqrt(((pred - target) ** 2).mean()))
    denom = (np.abs(pred) + np.abs(target)) / 2.0
    smape = float((np.abs(pred - target) / (denom + 1e-8)).mean() * 100)
    mape  = float((np.abs((pred - target) / (target + 1e-8))).mean() * 100)
    return {'MAE': mae, 'RMSE': rmse, 'sMAPE': smape, 'MAPE': mape}


def build_metrics_payload(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """predictions / targets: (N, H, 3)  columns: ele, cool, heat"""
    def task(i):
        m = compute_metrics(predictions[:, :, i], targets[:, :, i])
        return m['MAE'], m['RMSE'], m['sMAPE'], m['MAPE']

    e_mae, e_rmse, e_smape, e_mape = task(0)
    c_mae, c_rmse, c_smape, c_mape = task(1)
    h_mae, h_rmse, h_smape, h_mape = task(2)

    avg_mae   = float(np.mean([e_mae,   c_mae,   h_mae]))
    avg_rmse  = float(np.mean([e_rmse,  c_rmse,  h_rmse]))
    avg_smape = float(np.mean([e_smape, c_smape, h_smape]))
    avg_mape  = float(np.mean([e_mape,  c_mape,  h_mape]))

    return {
        'electric_mae':   e_mae,   'electric_rmse':   e_rmse,
        'electric_smape': e_smape, 'electric_mape':   e_mape,
        'cooling_mae':    c_mae,   'cooling_rmse':     c_rmse,
        'cooling_smape':  c_smape, 'cooling_mape':     c_mape,
        'heating_mae':    h_mae,   'heating_rmse':     h_rmse,
        'heating_smape':  h_smape, 'heating_mape':     h_mape,
        'avg_mae':        avg_mae, 'avg_rmse':         avg_rmse,
        'avg_smape':      avg_smape, 'avg_mape':       avg_mape,
    }


def build_report(m: dict, label: str) -> str:
    lines = [
        f"EVALUATION RESULTS  [{label}]", "",
        "Overall Performance:",
        f"  MAE:   {m['avg_mae']:.2f}",
        f"  RMSE:  {m['avg_rmse']:.2f}",
        f"  sMAPE: {m['avg_smape']:.2f}",
        f"  MAPE:  {m['avg_mape']:.2f}", "",
        "Electric (KW):",
        f"  MAE:   {m['electric_mae']:.2f}",
        f"  RMSE:  {m['electric_rmse']:.2f}",
        f"  sMAPE: {m['electric_smape']:.2f}", "",
        "Cooling (CHWTON):",
        f"  MAE:   {m['cooling_mae']:.2f}",
        f"  RMSE:  {m['cooling_rmse']:.2f}",
        f"  sMAPE: {m['cooling_smape']:.2f}", "",
        "Heating (HTmmBTU):",
        f"  MAE:   {m['heating_mae']:.2f}",
        f"  RMSE:  {m['heating_rmse']:.2f}",
        f"  sMAPE: {m['heating_smape']:.2f}",
    ]

    # ── Efficiency section (only if present) ──────────────────────────────
    eff = m.get('efficiency')
    if isinstance(eff, dict) and eff:
        lines.append("")
        lines.append("Model Efficiency:")
        if 'params_M' in eff:
            lines.append(f"  Params:    {eff['params_M']:.3f} M  "
                         f"({eff.get('params_total_mb_fp32', 0):.1f} MB fp32)")
        if eff.get('thop_available') is True and eff.get('flops_M') is not None:
            lines.append(f"  FLOPs:     {eff['flops_M']:.2f} M  (MACs={eff['macs']/1e6:.2f} M)")
        elif 'thop_error' in eff:
            lines.append(f"  FLOPs:     N/A ({eff['thop_error']})")
        lat = eff.get('latency')
        if isinstance(lat, dict) and lat:
            lines.append(f"  Inference: {lat.get('mean_ms_per_sample', 0):.3f} ms/sample  "
                         f"(batch={lat.get('batch_size', 1)}, "
                         f"device={lat.get('device', '?')})")
            if lat.get('throughput_samples_per_s'):
                lines.append(f"  Throughput:{lat['throughput_samples_per_s']:.1f} samples/s")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def resolve_checkpoint(checkpoint_dir_arg: str) -> str:
    for candidate in [
        checkpoint_dir_arg,
        os.path.join(checkpoint_dir_arg, 'best_model.pth'),
        os.path.join('checkpoints', checkpoint_dir_arg, 'best_model.pth'),
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot find best_model.pth for '{checkpoint_dir_arg}'"
    )


def resolve_output_dirs(ckpt_path: str, ckpt_dir_arg: str, primary_dir: str):
    parts = Path(os.path.abspath(ckpt_path)).parts
    if 'checkpoints' in parts:
        idx = parts.index('checkpoints')
        exp_name = parts[idx + 1] if idx + 1 < len(parts) else Path(ckpt_dir_arg).name
    else:
        exp_name = Path(ckpt_dir_arg).name

    primary   = Path(primary_dir)
    canonical = Path('..') / 'results' / exp_name / 'test'
    return exp_name, [primary, canonical]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(model, test_loader, device):
    model.eval()
    all_preds, all_tgts = [], []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred  = model(batch)['predictions'].cpu().numpy()
        tgt   = batch['endo_future'].cpu().numpy()
        all_preds.append(pred)
        all_tgts.append(tgt)
    return np.concatenate(all_preds, 0), np.concatenate(all_tgts, 0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(predictions, targets, save_dir, n_samples=5):
    names = ['Electric (KW)', 'Cooling (CHWTON)', 'Heating (HTmmBTU)']
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(n_samples, len(predictions))):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        for j, (ax, name) in enumerate(zip(axes, names)):
            ax.plot(targets[i, :, j],     label='Ground Truth',  linewidth=2)
            ax.plot(predictions[i, :, j], label='LMSP-MLP Pred', linewidth=2,
                    linestyle='--')
            ax.set_title(f'{name} — Sample {i+1}')
            ax.set_xlabel('Time Step (h)')
            ax.set_ylabel('Load')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'), dpi=150)
        plt.close()
    print(f"Figures saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(predictions, targets, metrics, output_dirs, label):
    report = build_report(metrics, label)
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / 'metrics.json').write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        (d / 'evaluation_report.txt').write_text(report, encoding='utf-8')
        np.savez(d / 'predictions.npz', predictions=predictions, targets=targets)
        print(f"Results → {d}")


def save_summary(args, exp_name, ckpt_path, config, metrics, output_dirs):
    summary = {
        'experiment_name': exp_name,
        'model':           'LMSP-MLP',
        'checkpoint_path': str(Path(ckpt_path).resolve()),
        'evaluated_at':    datetime.now().isoformat(timespec='seconds'),
        'data_path':       args.data_path,
        'input_len':       config.get('input_len', 168),
        'pred_len':        config.get('pred_len',   24),
        'batch_size':      args.batch_size,
        'metrics':         metrics,
    }
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)
        (d / 'summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"Summary  → {d / 'summary.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load config ───────────────────────────────────────────────────────
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    if not os.path.exists(config_path):
        config_path = os.path.join('checkpoints', args.checkpoint_dir, 'config.json')
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"Config: {config_path}")
    else:
        print("Warning: config.json not found, using defaults")

    # ── Resolve checkpoint ────────────────────────────────────────────────
    ckpt_path         = resolve_checkpoint(args.checkpoint_dir)
    ckpt_abs          = args.checkpoint_dir if os.path.isdir(args.checkpoint_dir) \
                        else os.path.join('checkpoints', args.checkpoint_dir)
    primary_eval_dir  = os.path.join(ckpt_abs, 'evaluation')
    exp_name, out_dirs = resolve_output_dirs(ckpt_path, args.checkpoint_dir,
                                             primary_eval_dir)
    print(f"Experiment : {exp_name}")

    # ── Data ──────────────────────────────────────────────────────────────
    _, _, test_loader, n_exo = create_dataloaders(
        data_path               = args.data_path,
        batch_size              = args.batch_size,
        input_len               = config.get('input_len', 168),
        pred_len                = config.get('pred_len',   24),
        use_feature_engineering = True,
        num_workers             = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = LMSP_MLP(
        input_len      = config.get('input_len',      168),
        pred_len       = config.get('pred_len',        24),
        d_model        = config.get('d_model',         256),
        d_exo          = config.get('d_exo',            64),
        n_exo          = n_exo,
        patch_size     = config.get('patch_size',       24),
        sparse_periods = [24, 168],
        sparse_d_model = config.get('sparse_d_model',  128),
        n_patch_layers = config.get('n_patch_layers',    2),
        lag_hours      = [0, 1, 2, 3, 6, 12, 24],
        decomp_kernel  = config.get('decomp_kernel',    25),
        dropout        = config.get('dropout',         0.1),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded: {ckpt_path}  "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_mae {ckpt.get('val_mae', float('nan')):.2f}, "
          f"params={n_params/1e6:.3f}M)")

    # ── Efficiency metrics ─────────────────────────────────────────────────
    print("\nComputing efficiency metrics...")
    efficiency = compute_efficiency_metrics(
        model, config, device, n_exo,
        compute_flops       = args.compute_flops,
        benchmark_latency   = args.benchmark_latency,
        latency_batch_size  = args.latency_batch_size,
        latency_warmup      = args.latency_warmup,
        latency_iters       = args.latency_iters,
    )
    if efficiency:
        print(f"  Params:    {efficiency.get('params_M', 0):.3f} M")
        if efficiency.get('thop_available'):
            print(f"  FLOPs:     {efficiency.get('flops_M', 0):.2f} M")
        else:
            print(f"  FLOPs:     N/A ({efficiency.get('thop_error', 'thop not installed')})")
        lat = efficiency.get('latency', {})
        if lat:
            print(f"  Inference: {lat.get('mean_ms_per_sample', 0):.3f} ms/sample  "
                  f"({lat.get('throughput_samples_per_s', 0):.1f} samples/s)")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    predictions, targets = run_evaluation(model, test_loader, device)
    metrics = build_metrics_payload(predictions, targets)

    # Attach efficiency into metrics payload
    if efficiency:
        metrics['efficiency'] = efficiency

    # ── Print ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"Overall: MAE={metrics['avg_mae']:.2f}  "
          f"RMSE={metrics['avg_rmse']:.2f}  "
          f"sMAPE={metrics['avg_smape']:.2f}%")
    print(f"Elec:   MAE={metrics['electric_mae']:.2f}")
    print(f"Cool:   MAE={metrics['cooling_mae']:.2f}")
    print(f"Heat:   MAE={metrics['heating_mae']:.2f}")
    print("="*55)

    # ── Save ──────────────────────────────────────────────────────────────
    save_results(predictions, targets, metrics, out_dirs, 'LMSP-MLP')
    save_summary(args, exp_name, ckpt_path, config, metrics, out_dirs)
    visualise(predictions, targets, str(out_dirs[0]))

    print(f"\nDone.  avg_mae={metrics['avg_mae']:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate LMSP-MLP')
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--data_path',
                        default='../data/2019-2022-full-feature-final_dataset.csv')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)

    # ── Efficiency control ────────────────────────────────────────────────
    parser.add_argument('--compute_flops', type=lambda x: x.lower() != 'false',
                        default=True,
                        help='Estimate FLOPs via thop (best-effort). Default: True')
    parser.add_argument('--benchmark_latency', type=lambda x: x.lower() != 'false',
                        default=True,
                        help='Benchmark forward latency. Default: True')
    parser.add_argument('--latency_batch_size', type=int, default=1,
                        help='Batch size for latency benchmark (default 1)')
    parser.add_argument('--latency_warmup', type=int, default=20,
                        help='Warmup iterations for latency benchmark')
    parser.add_argument('--latency_iters',  type=int, default=100,
                        help='Timed iterations for latency benchmark')

    args = parser.parse_args()
    main(args)
