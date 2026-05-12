"""Validacion cruzada anidada para los experimentos multilabel del laboratorio."""

from itertools import product
import argparse
import json
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

from config import TARGET_COLUMNS  # noqa: E402
from data_loader import CognitiveMultiLabelDataset, load_dataframe  # noqa: E402
from evaluation import compute_multilabel_metrics  # noqa: E402
from models import ShallowMultiLabelNet  # noqa: E402
from preprocessing import (  # noqa: E402
    make_stratified_inner_folds,
    make_stratified_outer_folds,
    prepare_xy,
)
from src.uncertainty import mc_dropout_predict, threshold_with_tiebreak  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "dataset" / "deterioro_cognitivo.sav"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "results_multilabel_experiments.csv"
SUMMARY_PATH = RESULTS_DIR / "summary_by_target.csv"
FINAL_SUMMARY_PATH = RESULTS_DIR / "final_summary.txt"
THRESHOLD = 0.5
OUTER_FOLDS = 5
INNER_FOLDS = 3
MC_SAMPLES = 50

RESULT_COLUMNS = [
    "target",
    "fold",
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "hamming_loss",
    "exact_match",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "f1_macro",
    "uncertainty_mean",
]

SUMMARY_COLUMNS = [
    "target",
    "mean_hamming",
    "std_hamming",
    "mean_exact",
    "mean_f1_micro",
    "mean_f1_macro",
    "mean_precision_micro",
    "mean_recall_micro",
    "mean_uncertainty",
]


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
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    device: torch.device,
    seed: int,
) -> nn.Module:
    """Entrena un modelo nuevo con BCEWithLogitsLoss y Adam."""

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
    """Predice probabilidades con sigmoide y devuelve tambien y_true."""

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
    """Evalua con sigmoide + umbral 0.5 + regla de desempate."""

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


def hyperparameter_grid(smoke_test: bool = False) -> list[dict]:
    """Devuelve el grid de hiperparametros para la busqueda interna."""

    if smoke_test:
        hidden_dims = [32]
        dropouts = [0.3]
        learning_rates = [1e-2]
        weight_decays = [0.0]
        batch_sizes = [32]
    else:
        hidden_dims = [16, 32, 64]
        dropouts = [0.2, 0.3, 0.5]
        learning_rates = [1e-2, 1e-3]
        weight_decays = [0.0, 1e-4]
        batch_sizes = [32]

    return [
        {
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        }
        for hidden_dim, dropout, lr, weight_decay, batch_size in product(
            hidden_dims,
            dropouts,
            learning_rates,
            weight_decays,
            batch_sizes,
        )
    ]


def summarize_target(target_col: str, fold_rows: list[dict]) -> dict:
    """Resume las metricas de los 5 folds externos para un target."""

    return {
        "target": target_col,
        "mean_hamming": float(np.mean([row["hamming_loss"] for row in fold_rows])),
        "std_hamming": float(np.std([row["hamming_loss"] for row in fold_rows])),
        "mean_exact": float(np.mean([row["exact_match"] for row in fold_rows])),
        "mean_f1_micro": float(np.mean([row["f1_micro"] for row in fold_rows])),
        "mean_f1_macro": float(np.mean([row["f1_macro"] for row in fold_rows])),
        "mean_precision_micro": float(
            np.mean([row["precision_micro"] for row in fold_rows])
        ),
        "mean_recall_micro": float(
            np.mean([row["recall_micro"] for row in fold_rows])
        ),
        "mean_uncertainty": float(
            np.mean([row["uncertainty_mean"] for row in fold_rows])
        ),
    }


def save_uncertainty_examples(
    target_col: str,
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    mean_probs: np.ndarray,
    std_probs: np.ndarray,
) -> None:
    """Guarda 10 ejemplos concretos de prediccion con incertidumbre MC Dropout."""

    ensure_results_dir()
    example_count = min(10, len(mean_probs))
    y_pred_threshold = threshold_with_tiebreak(
        mean_probs[:example_count],
        threshold=THRESHOLD,
    )

    rows = []
    for idx in range(example_count):
        rows.append(
            {
                "indice_muestra": int(sample_indices[idx]),
                "y_true": json.dumps(y_true[idx].tolist()),
                "mean_probs": json.dumps(mean_probs[idx].tolist()),
                "std_probs": json.dumps(std_probs[idx].tolist()),
                "y_pred_threshold": json.dumps(y_pred_threshold[idx].tolist()),
            }
        )

    output_path = RESULTS_DIR / f"uncertainty_examples_{target_col}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _most_frequent_value(results_df: pd.DataFrame, column: str) -> tuple[object, int]:
    """Devuelve el valor mas frecuente y su frecuencia."""

    counts = results_df[column].value_counts()
    return counts.index[0], int(counts.iloc[0])


def _format_hparam(value: object) -> str:
    """Formatea hiperparametros para el resumen de texto."""

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        return f"{float(value):g}"

    return str(value)


def write_final_summary(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    outer_folds: int,
    inner_folds: int,
    grid_size: int,
    mc_samples: int,
) -> None:
    """Genera results/final_summary.txt con el resumen ejecutivo."""

    ensure_results_dir()

    best_target = summary_df.loc[summary_df["mean_hamming"].idxmin()]
    worst_target = summary_df.loc[summary_df["mean_hamming"].idxmax()]
    lowest_uncertainty = summary_df.loc[summary_df["mean_uncertainty"].idxmin()]
    highest_uncertainty = summary_df.loc[summary_df["mean_uncertainty"].idxmax()]
    total_selections = len(results_df)

    hidden_dim, hidden_dim_count = _most_frequent_value(results_df, "hidden_dim")
    dropout, dropout_count = _most_frequent_value(results_df, "dropout")
    lr, lr_count = _most_frequent_value(results_df, "lr")
    weight_decay, weight_decay_count = _most_frequent_value(
        results_df,
        "weight_decay",
    )

    lines = [
        "=" * 64,
        "RESUMEN FINAL — Laboratorio 02 Deep Learning",
        "=" * 64,
        "",
        "Configuración:",
        f"- {outer_folds} folds externos × {inner_folds} folds internos "
        "(StratifiedKFold)",
        f"- {grid_size} configuraciones por fold interno",
        f"- {mc_samples} pasadas Monte Carlo Dropout",
        "- BCEWithLogitsLoss + sigmoide + umbral 0.5 con desempate",
        "",
        f"Resultados por target (promedio sobre {outer_folds} folds externos):",
        "",
        "Target     Hamming    F1-macro   F1-micro   ExactMatch  Uncertainty",
        "--------   --------   --------   --------   ----------  -----------",
    ]

    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.target:<8}   "
            f"{row.mean_hamming:>8.4f}   "
            f"{row.mean_f1_macro:>8.4f}   "
            f"{row.mean_f1_micro:>8.4f}   "
            f"{row.mean_exact:>10.4f}  "
            f"{row.mean_uncertainty:>11.4f}"
        )

    lines.extend(
        [
            "",
            "Mejor target (menor Hamming):    "
            f"{best_target['target']}  ({best_target['mean_hamming']:.4f})",
            "Peor target (mayor Hamming):     "
            f"{worst_target['target']}  ({worst_target['mean_hamming']:.4f})",
            "Target con menor incertidumbre:  "
            f"{lowest_uncertainty['target']}  "
            f"({lowest_uncertainty['mean_uncertainty']:.4f})",
            "Target con mayor incertidumbre:  "
            f"{highest_uncertainty['target']}  "
            f"({highest_uncertainty['mean_uncertainty']:.4f})",
            "",
            "Hiperparámetros más frecuentemente seleccionados:",
            "- hidden_dim: "
            f"{_format_hparam(hidden_dim)} "
            f"({hidden_dim_count} de {total_selections} selecciones)",
            "- dropout: "
            f"{_format_hparam(dropout)} "
            f"({dropout_count} de {total_selections})",
            "- lr: "
            f"{_format_hparam(lr)} "
            f"({lr_count} de {total_selections})",
            "- weight_decay: "
            f"{_format_hparam(weight_decay)} "
            f"({weight_decay_count} de {total_selections})",
            "",
            "=" * 64,
        ]
    )

    FINAL_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment_for_target(
    target_col: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    seed: int = 42,
    epochs: int = 80,
    device: torch.device | None = None,
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    mc_samples: int = MC_SAMPLES,
    grid: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Ejecuta validacion cruzada anidada 5x3 para un target."""

    set_seed(seed)
    device = device or resolve_device()

    dataframe = load_dataframe(data_path)
    X, y_int, Y = prepare_xy(dataframe, target_col)
    outer_splits = make_stratified_outer_folds(
        y_int,
        n_splits=outer_folds,
        seed=seed,
    )
    if grid is None:
        grid = hyperparameter_grid()
    fold_rows = []

    print(f"\nTarget: {target_col}")
    print(f"Dispositivo: {device}")

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        X_train_outer = X[outer_train_idx]
        Y_train_outer = Y[outer_train_idx]
        y_int_train_outer = y_int[outer_train_idx]
        X_test_outer = X[outer_test_idx]
        Y_test_outer = Y[outer_test_idx]

        inner_splits = make_stratified_inner_folds(
            y_int_train_outer,
            n_splits=inner_folds,
            seed=seed + fold_idx,
        )

        best_config = None
        best_mean_inner_hamming = float("inf")

        for config_idx, config in enumerate(grid):
            inner_hamming_values = []

            for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(
                inner_splits,
                start=1,
            ):
                model_seed = seed + fold_idx * 10_000 + config_idx * 100 + inner_idx
                model = train_model(
                    X_train=X_train_outer[inner_train_idx],
                    Y_train=Y_train_outer[inner_train_idx],
                    hidden_dim=config["hidden_dim"],
                    dropout=config["dropout"],
                    lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    batch_size=config["batch_size"],
                    epochs=epochs,
                    device=device,
                    seed=model_seed,
                )

                metrics = evaluate_model(
                    model=model,
                    X_eval=X_train_outer[inner_val_idx],
                    Y_eval=Y_train_outer[inner_val_idx],
                    batch_size=config["batch_size"],
                    device=device,
                    seed=model_seed,
                )
                inner_hamming_values.append(metrics["hamming_loss"])

            mean_inner_hamming = float(np.mean(inner_hamming_values))

            if mean_inner_hamming < best_mean_inner_hamming:
                best_mean_inner_hamming = mean_inner_hamming
                best_config = config

        final_seed = seed + fold_idx * 100_000
        final_model = train_model(
            X_train=X_train_outer,
            Y_train=Y_train_outer,
            hidden_dim=best_config["hidden_dim"],
            dropout=best_config["dropout"],
            lr=best_config["lr"],
            weight_decay=best_config["weight_decay"],
            batch_size=best_config["batch_size"],
            epochs=epochs,
            device=device,
            seed=final_seed,
        )

        test_loader = build_data_loader(
            X_test_outer,
            Y_test_outer,
            batch_size=best_config["batch_size"],
            shuffle=False,
            seed=final_seed,
        )
        probs, y_true = predict_probabilities(final_model, test_loader, device)
        y_pred = threshold_with_tiebreak(probs, threshold=THRESHOLD)
        metrics = compute_multilabel_metrics(y_true, y_pred)

        mean_probs, std_probs, y_true_mc = mc_dropout_predict(
            model=final_model,
            loader=test_loader,
            mc_samples=mc_samples,
            device=device,
        )
        uncertainty_mean = float(std_probs.mean())

        if fold_idx == outer_folds:
            save_uncertainty_examples(
                target_col=target_col,
                sample_indices=outer_test_idx[:10],
                y_true=y_true_mc[:10],
                mean_probs=mean_probs[:10],
                std_probs=std_probs[:10],
            )

        row = {
            "target": target_col,
            "fold": fold_idx,
            "hidden_dim": best_config["hidden_dim"],
            "dropout": best_config["dropout"],
            "lr": best_config["lr"],
            "weight_decay": best_config["weight_decay"],
            "batch_size": best_config["batch_size"],
            "hamming_loss": metrics["hamming_loss"],
            "exact_match": metrics["exact_match"],
            "precision_micro": metrics["precision_micro"],
            "recall_micro": metrics["recall_micro"],
            "f1_micro": metrics["f1_micro"],
            "f1_macro": metrics["f1_macro"],
            "uncertainty_mean": uncertainty_mean,
        }
        fold_rows.append(row)

        print(
            f"  Fold {fold_idx}: "
            f"best_hamming_interno={best_mean_inner_hamming:.4f}, "
            f"hamming_test={metrics['hamming_loss']:.4f}, "
            f"f1_micro={metrics['f1_micro']:.4f}"
        )

    return fold_rows, summarize_target(target_col, fold_rows)


def run_all_experiments(
    seed: int = 42,
    epochs: int = 80,
    data_path: str | Path = DEFAULT_DATA_PATH,
    targets: list[str] | None = None,
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    mc_samples: int = MC_SAMPLES,
    grid: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta los targets solicitados y guarda los CSV finales."""

    ensure_results_dir()
    device = resolve_device()
    targets_to_run = TARGET_COLUMNS if targets is None else targets
    if grid is None:
        grid = hyperparameter_grid()
    all_rows = []
    summary_rows = []

    for target_col in targets_to_run:
        target_rows, target_summary = run_experiment_for_target(
            target_col=target_col,
            data_path=data_path,
            seed=seed,
            epochs=epochs,
            device=device,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            mc_samples=mc_samples,
            grid=grid,
        )
        all_rows.extend(target_rows)
        summary_rows.append(target_summary)

    results_df = pd.DataFrame(all_rows, columns=RESULT_COLUMNS)
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    results_df.to_csv(RESULTS_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    write_final_summary(
        summary_df=summary_df,
        results_df=results_df,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        grid_size=len(grid),
        mc_samples=mc_samples,
    )

    print("\nResumen por target")
    print(summary_df.to_string(index=False))

    return results_df, summary_df


def build_argument_parser() -> argparse.ArgumentParser:
    """Argumentos minimos para ejecutar el laboratorio."""

    parser = argparse.ArgumentParser(
        description="Validacion cruzada anidada 5x3 para todos los targets."
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
        default=80,
        help="Epocas de entrenamiento por modelo.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Ejecuta una prueba rapida: GDS_R1, 2 folds externos, "
            "2 folds internos, 1 configuracion, 10 epocas y 5 muestras MC."
        ),
    )
    return parser


def main() -> None:
    """Punto de entrada del script."""

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.smoke_test:
        run_all_experiments(
            seed=args.seed,
            epochs=10,
            data_path=args.data_path,
            targets=["GDS_R1"],
            outer_folds=2,
            inner_folds=2,
            mc_samples=5,
            grid=hyperparameter_grid(smoke_test=True),
        )
        print("SMOKE TEST OK")
        return

    run_all_experiments(
        seed=args.seed,
        epochs=args.epochs,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
