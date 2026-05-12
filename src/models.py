"""Modelos del laboratorio."""

import torch
import torch.nn as nn


class ShallowMultiLabelNet(nn.Module):
    """
    Red neuronal poco profunda.

    Tiene:
    - una capa de entrada,
    - una capa oculta,
    - dropout,
    - y una capa de salida con tantas neuronas como clases tenga el experimento.
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 32,
        dropout: float = 0.3,
        output_dim: int = 3,
    ) -> None:
        """Inicializa las capas de la red neuronal multilabel."""

        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Calcula los logits de salida para un batch de entradas."""

        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits
