"""Comparacion de activaciones para el target GDS_R3."""

import argparse
import os
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import CognitiveMultiLabelDataset, load_dataframe  # noqa: E402
from evaluation import compute_multilabel_metrics  # noqa: E402
from models import ShallowMultiLabelNet  # noqa: E402
from preprocessing import make_stratified_outer_folds, prepare_xy  # noqa: E402
from uncertainty import threshold_with_tiebreak  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "dataset" / "deterioro_cognitivo.sav"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "activation_comparison.csv"

TARGET_COL = "GDS_R3"
ACTIVATIONS = ["relu", "tanh", "leaky_relu"]
RESULT_COLUMNS = [
    "activation",
    "fold",
    "hamming",
    "f1_macro",
    "f1_micro",
    "exact_match",
]

THRESHOLD = 0.5
OUTER_FOLDS = 5
HIDDEN_DIM = 16
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 32
EPOCHS = 80


def ensure_results_dir() -> None:
    """Crea la carpeta de resultados si no existe."""

    os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed: int) -> None:
    """Fija semillas para reproducibilidad."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str | None = None) -> torch.device:
    """Usa CUDA si esta disponible; si no, CPU."""

    if device_name is None or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero no hay GPU disponible.")

    return device


def build_data_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Construye un DataLoader reproducible."""

    dataset = CognitiveMultiLabelDataset(X, Y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    activation: str,
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> nn.Module:
    """Entrena un modelo con la activacion indicada."""

    set_seed(seed)
    train_loader = build_data_loader(
        X_train,
        Y_train,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    model = ShallowMultiLabelNet(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_dim=Y_train.shape[1],
        activation=activation,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice probabilidades con sigmoide y devuelve y_true."""

    model.eval()
    all_probs = []
    all_true = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_true.append(yb.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_true, axis=0)


def evaluate_model(
    model: nn.Module,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    """Evalua con sigmoide, umbral 0.5 y regla de desempate."""

    eval_loader = build_data_loader(
        X_eval,
        Y_eval,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    probs, y_true = predict_probabilities(model, eval_loader, device)
    y_pred = threshold_with_tiebreak(probs, threshold=THRESHOLD)
    return compute_multilabel_metrics(y_true, y_pred)


def summarize_by_activation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula promedios por activacion."""

    return (
        results_df.groupby("activation", as_index=False)[
            ["hamming", "f1_macro", "f1_micro", "exact_match"]
        ]
        .mean()
        .sort_values("activation")
    )


def run_activation_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    seed: int = 42,
    epochs: int = EPOCHS,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Ejecuta 5 folds externos para ReLU, Tanh y LeakyReLU en GDS_R3."""

    set_seed(seed)
    ensure_results_dir()
    device = device or resolve_device()

    dataframe = load_dataframe(data_path)
    X, y_int, Y = prepare_xy(dataframe, TARGET_COL)
    outer_splits = make_stratified_outer_folds(
        y_int,
        n_splits=OUTER_FOLDS,
        seed=seed,
    )
    rows = []

    print(f"Target: {TARGET_COL}")
    print(f"Dispositivo: {device}")
    print(
        "Config: "
        f"hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, lr={LEARNING_RATE}, "
        f"weight_decay={WEIGHT_DECAY}, batch_size={BATCH_SIZE}, epochs={epochs}"
    )

    for activation in ACTIVATIONS:
        print(f"\nActivacion: {activation}")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            fold_seed = seed + fold_idx * 100_000
            model = train_model(
                X_train=X[train_idx],
                Y_train=Y[train_idx],
                activation=activation,
                hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                device=device,
                seed=fold_seed,
            )
            metrics = evaluate_model(
                model=model,
                X_eval=X[test_idx],
                Y_eval=Y[test_idx],
                batch_size=BATCH_SIZE,
                device=device,
                seed=fold_seed,
            )

            row = {
                "activation": activation,
                "fold": fold_idx,
                "hamming": metrics["hamming_loss"],
                "f1_macro": metrics["f1_macro"],
                "f1_micro": metrics["f1_micro"],
                "exact_match": metrics["exact_match"],
            }
            rows.append(row)

            print(
                f"  Fold {fold_idx}: "
                f"hamming={row['hamming']:.4f}, "
                f"f1_macro={row['f1_macro']:.4f}, "
                f"f1_micro={row['f1_micro']:.4f}, "
                f"exact_match={row['exact_match']:.4f}"
            )

    results_df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results_df.to_csv(RESULTS_PATH, index=False)

    summary_df = summarize_by_activation(results_df)
    print("\nResumen promedio por activacion")
    print(summary_df.to_string(index=False))
    print(f"\nResultados guardados en: {RESULTS_PATH}")

    return results_df


def build_argument_parser() -> argparse.ArgumentParser:
    """Argumentos para ejecutar la comparacion de activaciones."""

    parser = argparse.ArgumentParser(
        description="Compara ReLU, Tanh y LeakyReLU en GDS_R3."
    )
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="Ruta al dataset .sav o .csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Epocas de entrenamiento por modelo.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo a usar: auto, cpu o cuda.",
    )
    return parser


def main() -> None:
    """Punto de entrada del script."""

    parser = build_argument_parser()
    args = parser.parse_args()

    run_activation_experiment(
        data_path=args.data_path,
        seed=args.seed,
        epochs=args.epochs,
        device=resolve_device(args.device),
    )


if __name__ == "__main__":
    main()
