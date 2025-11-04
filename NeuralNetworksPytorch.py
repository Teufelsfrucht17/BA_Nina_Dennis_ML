from __future__ import annotations

import argparse
from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import GloablVariableStorage
from Dataprep2 import finalrunner


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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds is not None else None

    # Modell + Optimierer
    model = SimpleNet(in_features=X.shape[1], out_features=Y.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
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
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_running += loss.item() * xb.size(0)
                    val_preds.append(preds.cpu())
                    val_targets.append(yb.cpu())
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
                best_state = copy.deepcopy(model.state_dict())
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
        model.load_state_dict(best_state)
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
        "state_dict": model.state_dict(),
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


def main(sheet:int | None):
    parser = argparse.ArgumentParser(description="Train simples PyTorch-Netz auf X/Y aus Dataprep2")
    parser.add_argument("--sheet", type=int, default=sheet, help="Sheet-Index für das Wertpapier (Default: 3)")
    parser.add_argument("--epochs", type=int, default=300, help="Anzahl maximaler Epochen (Default: 300)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch-Größe (Default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Lernrate (Default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2-Regularisierung/Weight Decay (Default: 0)")
    parser.add_argument("--patience", type=int, default=20, help="Frühes Stoppen nach X erfolglosen Validierungs-Epochen (Default: 20)")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimaler Validierungsverlust-Rückgang für Verbesserung (Default: 0)")
    parser.add_argument("--model_out", type=str, default=str("data_output/simple_net"+str(sheet)+".pt"), help="Pfad zum Speichern des Modells")

    args = parser.parse_args()
    train_model(
        sheet=args.sheet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        model_out=args.model_out,
    )


if __name__ == "__main__":
    try:
        for i in range(len(GloablVariableStorage.Portfolio)):
            main(i)
    except Exception as e:
        print(f"Ridge run failed: {e}")
