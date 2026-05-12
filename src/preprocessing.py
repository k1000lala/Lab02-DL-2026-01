"""Funciones para preparar X/Y y folds estratificados del laboratorio."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import TARGET_COLUMNS
from data_loader import build_input_matrix


def validate_target_column(dataframe: pd.DataFrame, target_col: str) -> None:
    """Verifica que el target pedido sea valido y exista en el dataset."""

    if target_col not in TARGET_COLUMNS:
        raise ValueError(
            f"Target invalido: {target_col}. Debe ser uno de {TARGET_COLUMNS}."
        )

    if target_col not in dataframe.columns:
        raise ValueError(f"La columna objetivo {target_col} no existe en el dataset.")


def encode_target_as_indices(
    dataframe: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """Codifica el target original como indices enteros 0..K-1."""

    validate_target_column(dataframe, target_col)

    y_raw = dataframe[target_col].astype(int)
    y_merged = y_raw.copy()

    while True:
        counts = y_merged.value_counts().sort_index()
        rare_classes = counts[counts < 5]

        if rare_classes.empty:
            break

        rare_class = int(rare_classes.index[0])
        rare_count = int(rare_classes.iloc[0])
        classes = counts.index.tolist()

        if len(classes) < 2:
            break

        class_position = classes.index(rare_class)

        if class_position < len(classes) - 1:
            target_class = int(classes[class_position + 1])
        else:
            target_class = int(classes[class_position - 1])

        print(
            f"[WARN] target {target_col}: clase {rare_class} "
            f"({rare_count} muestras) fusionada con clase {target_class}"
        )
        y_merged = y_merged.replace(rare_class, target_class)

    classes = sorted(y_merged.unique().tolist())
    class_to_idx = {class_value: idx for idx, class_value in enumerate(classes)}

    y_int = y_merged.map(class_to_idx).to_numpy(dtype=np.int64)

    return y_int, classes, class_to_idx


def one_hot_encode_target(y_int: np.ndarray, n_classes: int) -> np.ndarray:
    """Convierte indices enteros 0..K-1 a matriz one-hot float32."""

    y_int = np.asarray(y_int, dtype=np.int64)
    Y = np.zeros((len(y_int), n_classes), dtype=np.float32)
    Y[np.arange(len(y_int)), y_int] = 1.0
    return Y


def encode_target_as_one_hot(
    dataframe: pd.DataFrame,
    target_name: str,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Toma una sola columna objetivo y la convierte a formato one-hot.

    Ejemplo:
    Si la clase original es 2 y el experimento tiene tres clases,
    entonces el vector queda como [0, 1, 0].
    """

    y_int, classes, class_to_idx = encode_target_as_indices(dataframe, target_name)
    Y = one_hot_encode_target(y_int, n_classes=len(classes))
    return Y, classes, class_to_idx


def prepare_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepara X, y_int e Y one-hot multilabel para un experimento."""

    X = build_input_matrix(df, feature_cols)
    y_int, classes, _ = encode_target_as_indices(df, target_col)
    Y = one_hot_encode_target(y_int, n_classes=len(classes))
    return X, y_int, Y


def prepare_experiment_data(
    dataframe: pd.DataFrame,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[int, int]]:
    """Prepara X e Y para un experimento puntual."""

    X = build_input_matrix(dataframe)
    Y, classes, class_to_idx = encode_target_as_one_hot(dataframe, target_name)
    return X, Y, classes, class_to_idx


def _validate_stratified_folds(y_int: np.ndarray, n_splits: int) -> np.ndarray:
    """Valida y prepara las etiquetas enteras para StratifiedKFold."""

    if n_splits < 2:
        raise ValueError("n_splits debe ser mayor o igual que 2.")

    y_int = np.asarray(y_int, dtype=np.int64)

    if y_int.ndim != 1:
        raise ValueError("y_int debe ser un vector 1D con clases enteras.")

    unique_labels, counts = np.unique(y_int, return_counts=True)

    if len(unique_labels) < 2:
        raise ValueError(
            "Se requieren al menos dos clases distintas para realizar validacion."
        )

    if counts.min() < n_splits:
        raise ValueError(
            "No se puede crear una validacion estratificada con "
            f"{n_splits} folds porque la clase menos frecuente solo tiene "
            f"{counts.min()} muestras."
        )

    return y_int


def _make_stratified_folds(
    y_int: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Crea folds estratificados usando el target entero 0..K-1."""

    y_int = _validate_stratified_folds(y_int, n_splits)
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    dummy_inputs = np.zeros(len(y_int), dtype=np.float32)
    return list(splitter.split(dummy_inputs, y_int))


def make_stratified_outer_folds(
    y_int: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Crea los folds externos estratificados del laboratorio."""

    return _make_stratified_folds(y_int, n_splits=n_splits, seed=seed)


def make_stratified_inner_folds(
    y_int_train: np.ndarray,
    n_splits: int = 3,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Crea los folds internos estratificados dentro del train externo."""

    return _make_stratified_folds(y_int_train, n_splits=n_splits, seed=seed)


def split_for_validation(
    Y: np.ndarray,
    n_splits: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compatibilidad: crea folds estratificados desde una matriz one-hot."""

    Y = np.asarray(Y, dtype=np.int64)

    if Y.ndim != 2:
        raise ValueError("Y debe ser una matriz 2D para construir folds estratificados.")

    label_counts = Y.sum(axis=1)
    if np.any(label_counts != 1):
        raise ValueError("Cada muestra debe activar exactamente una etiqueta en Y.")

    y_int = np.argmax(Y, axis=1)
    return _make_stratified_folds(y_int, n_splits=n_splits, seed=random_state)
