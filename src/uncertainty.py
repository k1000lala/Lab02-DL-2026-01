"""
Monte Carlo Dropout para estimación de incertidumbre predictiva.

Idea: durante el entrenamiento, Dropout regulariza la red. Si se mantiene
Dropout activo durante inferencia y se hacen múltiples pasadas, cada pasada
genera una predicción ligeramente distinta. La media estima la probabilidad
y la desviación estándar estima la incertidumbre.
"""
import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """Pone TODAS las capas Dropout en modo train(), dejando el resto en eval().
    
    Esto permite mantener Dropout activo durante la inferencia para Monte Carlo
    Dropout, sin afectar otras capas (ej: BatchNorm si las hubiera).
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model, loader, mc_samples=50, device="cpu"):
    """Hace n pasadas hacia adelante con Dropout activo y devuelve media y std.
    
    Parameters
    ----------
    model : nn.Module
        Red entrenada.
    loader : DataLoader
        DataLoader sobre el conjunto a predecir.
    mc_samples : int
        Número de pasadas Monte Carlo (default 50, según pauta del laboratorio).
    device : str
        'cpu' o 'cuda'.
    
    Returns
    -------
    mean_probs : np.ndarray, shape (N, K)
        Probabilidad media por clase, después de aplicar sigmoide.
    std_probs : np.ndarray, shape (N, K)
        Desviación estándar por clase (incertidumbre por etiqueta).
    y_true : np.ndarray, shape (N, K)
        Etiquetas verdaderas one-hot.
    """
    import numpy as np
    
    enable_dropout_during_inference(model)
    
    all_means = []
    all_stds = []
    all_true = []
    
    for xb, yb in loader:
        xb = xb.to(device)
        # Acumular las mc_samples pasadas para este batch
        batch_samples = []
        for _ in range(mc_samples):
            logits = model(xb)
            probs = torch.sigmoid(logits)
            batch_samples.append(probs.unsqueeze(0))
        batch_samples = torch.cat(batch_samples, dim=0)  # (mc, B, K)
        
        mean_b = batch_samples.mean(dim=0).cpu().numpy()  # (B, K)
        std_b = batch_samples.std(dim=0).cpu().numpy()    # (B, K)
        
        all_means.append(mean_b)
        all_stds.append(std_b)
        all_true.append(yb.numpy())
    
    mean_probs = np.concatenate(all_means, axis=0)
    std_probs = np.concatenate(all_stds, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    
    return mean_probs, std_probs, y_true


def threshold_with_tiebreak(mean_probs, threshold=0.5):
    """Aplica umbral con regla de desempate.
    
    Si todas las probabilidades de una muestra están por debajo del umbral,
    se activa la clase de mayor probabilidad. Esto evita predicciones con
    cero clases activas (que rompen métricas como exact match).
    """
    import numpy as np
    
    preds = (mean_probs >= threshold).astype(np.int32)
    # Para muestras con cero predicciones, activar la clase de mayor prob
    zero_rows = preds.sum(axis=1) == 0
    if zero_rows.any():
        argmax_idx = mean_probs[zero_rows].argmax(axis=1)
        preds[zero_rows, argmax_idx] = 1
    return preds
