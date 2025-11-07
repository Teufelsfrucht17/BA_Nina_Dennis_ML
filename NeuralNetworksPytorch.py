from __future__ import annotations

import argparse
from pathlib import Path
import copy
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import GloablVariableStorage
from Dataprep2 import finalrunner
from createScoreModels import createscore


class SimpleNet(nn.Module):
    """Ein kleines MLP mit zwei Hidden-Layern."""

    def __init__(self, in_features: int = 1, hidden1: int = 32, hidden2: int = 16, out_features: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_xy(sheet: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Lädt X, Y aus Dataprep2.finalrunner(sheet) und konvertiert zu Tensoren.

    Die Helper aus Dataprep2 haben zuletzt X und Y vertauscht. Wir prüfen daher
    die Spaltennamen und drehen die Rückgabe bei Bedarf zurück. Zusätzlich
    liefern wir die Feature-Namen für Logging/Serialisierung.
    """
    X_df, Y_df = finalrunner(sheet)

    def _normalize(cols: list[str]) -> set[str]:
        return {c.lower() for c in cols}

    feature_cols = {"momentum", "change_dax", "change_vdax"}
    target_cols = {"change"}

    x_cols = _normalize(list(X_df.columns))
    y_cols = _normalize(list(Y_df.columns))

    x_has_features = feature_cols.issubset(x_cols)
    y_has_target = target_cols.issubset(y_cols)

    if not (x_has_features and y_has_target):
        swapped_x_has_features = feature_cols.issubset(y_cols)
        swapped_y_has_target = target_cols.issubset(x_cols)
        if swapped_x_has_features and swapped_y_has_target:
            X_df, Y_df = Y_df, X_df
            x_cols, y_cols = y_cols, x_cols
        else:
            raise ValueError(
                "Dataprep2.finalrunner liefert unerwartete Spalten: "
                f"X={list(X_df.columns)}, Y={list(Y_df.columns)}"
            )

    X_t = torch.tensor(X_df.values, dtype=torch.float32)
    Y_t = torch.tensor(Y_df.values, dtype=torch.float32)

    return X_t, Y_t, list(X_df.columns)


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Berechne das mittlere R² (mehrere Targets werden gemittelt)."""
    if y_true.numel() == 0:
        return float("nan")

    y_true = y_true.float()
    y_pred = y_pred.float()

    mean_true = y_true.mean(dim=0, keepdim=True)
    ss_tot = ((y_true - mean_true) ** 2).sum(dim=0)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)

    # Verhindert Division durch Null bei konstantem Target
    eps = torch.finfo(torch.float32).eps
    r2_per_target = 1.0 - ss_res / (ss_tot + eps)
    return float(r2_per_target.mean().item())


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Gibt das Basis-Modell zurück (entfernt nn.DataParallel-Hülle)."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_model_for_inference(
    model_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict, torch.device]:
    """Lädt ein gespeichertes SimpleNet inkl. Meta-Daten für Inferenz.

    - Handhabt 'module.'-Prefix aus DataParallel-Checkpoints automatisch.
    - Platziert Modell aufs gewünschte Device und setzt eval().
    - Gibt (modell, meta, device) zurück.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    map_loc = device if device.type == "cuda" else torch.device("cpu")
    ckpt = torch.load(model_path, map_location=map_loc)

    base = SimpleNet(
        in_features=ckpt["in_features"],
        hidden1=ckpt.get("hidden1", 32),
        hidden2=ckpt.get("hidden2", 16),
        out_features=ckpt["out_features"],
    )

    state = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    base.load_state_dict(state)
    base.to(device)

    model = base
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base)
    model.eval()
    return model, ckpt, device


def predict_df(
    model: nn.Module,
    meta: dict,
    X_df: pd.DataFrame,
    device: torch.device | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    """Berechnet Vorhersagen für ein DataFrame mit Feature-Spalten.

    Erwartet die selben Spalten wie im Training (meta['feature_names']).
    Standardisiert per meta['feature_mean'] und meta['feature_std'].
    """
    if not isinstance(X_df, pd.DataFrame):
        raise TypeError("X_df muss ein pandas.DataFrame sein")

    feature_names = meta["feature_names"]
    missing = [c for c in feature_names if c not in X_df.columns]
    if missing:
        raise ValueError(f"Fehlende Feature-Spalten: {missing}")

    X_np = X_df[feature_names].values
    X_t = torch.tensor(X_np, dtype=torch.float32)
    mean = torch.tensor(meta["feature_mean"], dtype=torch.float32).unsqueeze(0)
    std = torch.tensor(meta["feature_std"], dtype=torch.float32).unsqueeze(0).clamp_min(1e-8)
    X_t = (X_t - mean) / std

    ds = TensorDataset(X_t)
    num_workers = min(4, os.cpu_count() or 0)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device is not None and device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    preds = []
    m = model
    m.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device is not None and device.type == "cuda")):
        for (xb,) in loader:
            if device is not None:
                xb = xb.to(device, non_blocking=True)
            out = m(xb)
            preds.append(out.detach().cpu())
    return torch.cat(preds, dim=0)


def train_model(
    sheet: int = 3,
    epochs: int = 300,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    weight_decay: float = 0.0,
    patience: int = 20,
    min_delta: float = 0.0,
    model_out: str | Path = "data_output/simple_net.pt",
) -> dict:
    """Trainiert ein kleines Netz auf X->Y und speichert Gewichte.

    Features werden anhand des Trainingssegments standardisiert. Rückgabe
    enthält Metriken der letzten Epoche.
    """
    # Device wählen (CUDA wenn verfügbar)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sheet {sheet}] Device: {device}")
    if device.type == "cuda":
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Daten laden
    X, Y, feature_names = load_xy(sheet)
    n_samples = X.shape[0]

    # Chronologischer Train/Val-Split (keine Zufallsdurchmischung)
    val_size = int(n_samples * val_split)
    train_size = n_samples - val_size

    if train_size <= 0:
        raise ValueError("Trainingssplit ergibt keine Trainingsdaten. val_split zu groß?")

    # Feature-Standardisierung nach Trainingsschnitt
    train_slice = slice(0, train_size)
    train_mean = X[train_slice].mean(dim=0, keepdim=True)
    train_std = X[train_slice].std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    X = (X - train_mean) / train_std

    dataset = TensorDataset(X, Y)

    if val_size > 0:
        train_indices = torch.arange(0, train_size)
        val_indices = torch.arange(train_size, n_samples)
        train_ds = torch.utils.data.Subset(dataset, train_indices.tolist())
        val_ds = torch.utils.data.Subset(dataset, val_indices.tolist())
    else:
        train_ds = dataset
        val_ds = None

    # DataLoader mit CUDA-optimierten Parametern
    num_workers = min(4, os.cpu_count() or 0)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
        if val_ds is not None else None
    )

    # Modell + Optimierer
    model = SimpleNet(in_features=X.shape[1], out_features=Y.shape[1])
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"DataParallel aktiv: {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    # Falls das Kriterium Parameter/Buffer hätte, aufs Device verschieben
    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Mixed Precision (AMP) für schnellere GPU-Trainingsläufe
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_r2 = None
    val_r2 = None

    best_state: dict | None = None
    best_metrics: dict | None = None
    best_epoch: int | None = None
    best_val_loss: float | None = None
    epochs_since_improve = 0
    last_epoch = 0

    # Training Loop
    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        running_loss = 0.0
        train_preds: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(xb)
                loss = criterion(preds, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            train_preds.append(preds.detach().cpu())
            train_targets.append(yb.detach().cpu())

        train_loss = running_loss / train_size if train_size > 0 else float("nan")
        train_r2 = None
        if train_targets:
            train_preds_t = torch.cat(train_preds, dim=0)
            train_targets_t = torch.cat(train_targets, dim=0)
            train_r2 = _r2_score(train_targets_t, train_preds_t)

        # Validation
        val_loss = None
        val_r2 = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_preds: list[torch.Tensor] = []
            val_targets: list[torch.Tensor] = []
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_running += loss.item() * xb.size(0)
                    val_preds.append(preds.detach().cpu())
                    val_targets.append(yb.detach().cpu())
            val_loss = val_running / val_size if val_size > 0 else None
            if val_targets:
                val_preds_t = torch.cat(val_preds, dim=0)
                val_targets_t = torch.cat(val_targets, dim=0)
                val_r2 = _r2_score(val_targets_t, val_preds_t)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            if val_loss is not None:
                r2_str = f" - train_r2={train_r2:.4f}" if train_r2 is not None else ""
                val_r2_str = f" - val_r2={val_r2:.4f}" if val_r2 is not None else ""
                print(
                    f"Epoch {epoch:4d}/{epochs} - train_loss={train_loss:.6f}{r2_str}"
                    f" - val_loss={val_loss:.6f}{val_r2_str}"
                )
            else:
                r2_str = f" - train_r2={train_r2:.4f}" if train_r2 is not None else ""
                print(f"Epoch {epoch:4d}/{epochs} - train_loss={train_loss:.6f}{r2_str}")

        if val_loss is not None:
            target_metric = val_loss
            improved = False
            if best_val_loss is None or target_metric < (best_val_loss - min_delta):
                best_val_loss = target_metric
                improved = True
            if improved:
                best_state = copy.deepcopy(_unwrap_model(model).state_dict())
                best_epoch = epoch
                best_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_r2": train_r2,
                    "val_r2": val_r2,
                    "best_epoch": epoch,
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if patience > 0 and epochs_since_improve >= patience:
                    print(
                        f"Frühes Stoppen nach Epoche {epoch} (keine Verbesserung über {patience} Epochen)."
                    )
                    break

    # Modell speichern
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        _unwrap_model(model).load_state_dict(best_state)
        metrics = best_metrics.copy() if best_metrics is not None else {}
        if best_epoch is not None:
            metrics.setdefault("best_epoch", best_epoch)
    else:
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "best_epoch": last_epoch,
        }

    torch.save({
        "state_dict": _unwrap_model(model).state_dict(),
        "in_features": X.shape[1],
        "out_features": Y.shape[1],
        "hidden1": 32,
        "hidden2": 16,
        "sheet": sheet,
        "feature_names": feature_names,
        "feature_mean": train_mean.squeeze(0).tolist(),
        "feature_std": train_std.squeeze(0).tolist(),
        "best_epoch": metrics.get("best_epoch", last_epoch),
    }, model_out)

    print(f"Gespeichert: {model_out} | Metrics: {metrics}")
    return metrics


def main(sheet:int | None,report: pd.DataFrame | None = None)->pd.DataFrame:
    parser = argparse.ArgumentParser(description="Train simples PyTorch-Netz auf X/Y aus Dataprep2")
    parser.add_argument("--sheet", type=int, default=sheet, help="Sheet-Index für das Wertpapier (Default: 3)")
    parser.add_argument("--epochs", type=int, default=300, help="Anzahl maximaler Epochen (Default: 300)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch-Größe (Default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Lernrate (Default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2-Regularisierung/Weight Decay (Default: 0)")
    parser.add_argument("--patience", type=int, default=20, help="Frühes Stoppen nach X erfolglosen Validierungs-Epochen (Default: 20)")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimaler Validierungsverlust-Rückgang für Verbesserung (Default: 0)")
    parser.add_argument("--model_out", type=str, default=str("data_output/NN/simple_net"+str(sheet)+".pt"), help="Pfad zum Speichern des Modells")

    args = parser.parse_args()
    metric = train_model(
        sheet=args.sheet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        model_out=args.model_out,
    )
    report.loc[len(report)] = [
        "Neural Network",
        sheet,
        metric["train_r2"],
        metric["val_r2"],
        metric["best_epoch"],
        "N/A",
    ]
    return report


def runNN()->pd.DataFrame:
    report = createscore()
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            report = main(i, report)
    except Exception as e:
        print(f"Ridge run failed: {e}")

    return report
