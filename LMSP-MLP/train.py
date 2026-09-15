"""
Training script for LMSP-MLP
Usage:
    python train.py --pred_len 24
    python train.py --pred_len 48 --d_model 256 --epochs 100
"""

import sys
import torch
import torch.optim as optim
import numpy as np
import random
import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

# Set to True to suppress tqdm bars (auto-detected when stdout is not a TTY)
QUIET = not sys.stdout.isatty()

from models.lmsp_mlp import LMSP_MLP
from data.dataset    import create_dataloaders


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mae  = torch.abs(pred - target).mean()
    mse  = ((pred - target) ** 2).mean()
    rmse = torch.sqrt(mse)
    denom  = (torch.abs(pred) + torch.abs(target)) / 2
    smape  = (torch.abs(pred - target) / (denom + 1e-8)).mean() * 100
    return {'mae': mae.item(), 'rmse': rmse.item(), 'smape': smape.item()}


# ---------------------------------------------------------------------------
# Adaptive task weights
# ---------------------------------------------------------------------------

def compute_adaptive_weights(train_loader, device, num_samples: int = 100) -> list:
    print("\n" + "="*55)
    print("Computing adaptive task weights…")
    scales = []
    for i, batch in enumerate(train_loader):
        if i >= num_samples:
            break
        ef = batch['endo_future'].to(device)
        scales.append(ef.abs().mean(dim=(0, 1)))           # (3,)
    avg = torch.stack(scales).mean(0)
    print(f"  KW      avg: {avg[0]:.1f}")
    print(f"  CHWTON  avg: {avg[1]:.1f}")
    print(f"  HTmmBTU avg: {avg[2]:.1f}")
    weights = 1.0 / (avg + 1e-6)
    weights = (weights / weights.sum() * 3.0)
    print(f"  Weights: {weights.tolist()}")
    print("="*55 + "\n")
    return weights.tolist()


# ---------------------------------------------------------------------------
# Train / Validate one epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, task_weights, device):
    model.train()
    total_loss = 0.0
    metrics    = {'mae': 0.0, 'rmse': 0.0, 'smape': 0.0}
    w = torch.tensor(task_weights, device=device)   # (3,)

    pbar = tqdm(loader, desc='Train', leave=False, disable=QUIET)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch['endo_future']                # (B, H, 3)

        optimizer.zero_grad()
        out  = model(batch)
        pred = out['predictions']                    # (B, H, 3)

        # Weighted MAE loss (balance KW / CHWTON / HTmmBTU magnitudes)
        per_task_mae = torch.abs(pred - target).mean(dim=(0, 1))  # (3,)
        loss = (w * per_task_mae).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        m = compute_metrics(pred, target)
        for k in metrics:
            metrics[k] += m[k]
        pbar.set_postfix(loss=f"{loss.item():.2f}", mae=f"{m['mae']:.2f}")

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


@torch.no_grad()
def validate(model, loader, task_weights, device):
    model.eval()
    total_loss = 0.0
    metrics    = {'mae': 0.0, 'rmse': 0.0, 'smape': 0.0}
    w = torch.tensor(task_weights, device=device)

    for batch in tqdm(loader, desc='Val  ', leave=False, disable=QUIET):
        batch  = {k: v.to(device) for k, v in batch.items()}
        target = batch['endo_future']
        pred   = model(batch)['predictions']

        per_task_mae = torch.abs(pred - target).mean(dim=(0, 1))
        loss = (w * per_task_mae).sum()
        total_loss += loss.item()

        m = compute_metrics(pred, target)
        for k in metrics:
            metrics[k] += m[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    global QUIET
    if args.quiet or not sys.stdout.isatty():
        QUIET = True

    set_seed(args.seed)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name  = f"lmsp_mlp_pred{args.pred_len}_{timestamp}"

    ckpt_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n" + "="*55)
    print(f"Experiment : {exp_name}")
    print(f"Device     : {device}")
    print(f"pred_len   : {args.pred_len}")
    print(f"input_len  : {args.input_len}")
    print("="*55 + "\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, n_exo = create_dataloaders(
        data_path               = args.data_path,
        batch_size              = args.batch_size,
        input_len               = args.input_len,
        pred_len                = args.pred_len,
        use_feature_engineering = True,
        num_workers             = args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  "
          f"Val: {len(val_loader)}  Test: {len(test_loader)}")
    print(f"n_exo = {n_exo}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = LMSP_MLP(
        input_len      = args.input_len,
        pred_len       = args.pred_len,
        d_model        = args.d_model,
        d_exo          = args.d_exo,
        n_exo          = n_exo,
        patch_size     = args.patch_size,
        sparse_periods = [24, 168],
        sparse_d_model = args.sparse_d_model,
        n_patch_layers = args.n_patch_layers,
        lag_hours      = [0, 1, 2, 3, 6, 12, 24],
        decomp_kernel  = args.decomp_kernel,
        dropout        = args.dropout,
    ).to(device)

    info = model.count_parameters()
    print(f"Parameters: {info['total']:,} ({info['total_M']:.3f} M)\n")

    # ── Adaptive task weights ─────────────────────────────────────────────
    task_weights = compute_adaptive_weights(train_loader, device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # ── Config snapshot ───────────────────────────────────────────────────
    config = vars(args)
    config.update({'experiment_name': exp_name, 'n_exo': n_exo,
                   'task_weights': task_weights, 'parameters': info})
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_mae  = float('inf')
    patience_cnt  = 0
    history       = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    log_path      = os.path.join(ckpt_dir, 'training_log.txt')

    print("="*55)
    print("Starting training …")
    print("="*55 + "\n")

    with open(log_path, 'w') as log_f:
        for epoch in range(1, args.epochs + 1):
            if not QUIET:
                print(f"Epoch {epoch}/{args.epochs}")
            tr_loss, tr_m = train_epoch(model, train_loader, optimizer, task_weights, device)
            vl_loss, vl_m = validate(model, val_loader, task_weights, device)
            scheduler.step(vl_m['mae'])

            history['train_loss'].append(tr_loss)
            history['train_mae'].append(tr_m['mae'])
            history['val_loss'].append(vl_loss)
            history['val_mae'].append(vl_m['mae'])

            msg = (f"Epoch {epoch:03d} | "
                   f"TrLoss {tr_loss:.4f} TrMAE {tr_m['mae']:.2f} | "
                   f"VlLoss {vl_loss:.4f} VlMAE {vl_m['mae']:.2f}")
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()

            if vl_m['mae'] < best_val_mae:
                best_val_mae = vl_m['mae']
                patience_cnt = 0
                torch.save({
                    'epoch':            epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae':          best_val_mae,
                    'config':           config,
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                print(f"  [OK] New best  VlMAE={best_val_mae:.2f}\n")
            else:
                patience_cnt += 1
                print(f"  Patience {patience_cnt}/{args.patience}\n")
                if patience_cnt >= args.patience:
                    print("Early stopping.")
                    break

    with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("="*55)
    print("Training complete.")
    print(f"Best Val MAE : {best_val_mae:.2f}")
    print(f"Checkpoint   : {ckpt_dir}")
    print("="*55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LMSP-MLP')

    # Data
    parser.add_argument('--data_path',   default='../data/2019-2022-full-feature-final_dataset.csv')
    parser.add_argument('--input_len',   type=int,   default=168)
    parser.add_argument('--pred_len',    type=int,   default=24)
    parser.add_argument('--num_workers', type=int,   default=0)

    # Model
    parser.add_argument('--d_model',        type=int,   default=256)
    parser.add_argument('--d_exo',          type=int,   default=64)
    parser.add_argument('--patch_size',     type=int,   default=24)
    parser.add_argument('--sparse_d_model', type=int,   default=128)
    parser.add_argument('--n_patch_layers', type=int,   default=2)
    parser.add_argument('--decomp_kernel',  type=int,   default=25)
    parser.add_argument('--dropout',        type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.01)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--quiet',         action='store_true',
                        help='Suppress tqdm bars (auto-set when stdout is not a TTY)')

    args = parser.parse_args()
    main(args)
