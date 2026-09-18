"""Neural-network components used by the publication experiment."""

from __future__ import annotations

import torch
from torch import nn


class AttentionLSTMActor(nn.Module):
    """LSTM actor used to produce Bernoulli action probabilities.

    Attention is applied over the sequence dimension so the actor can learn
    which historical states are most relevant to the current action.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.unit_input = input_size
        self.unit_hidden = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Keep the original parameter name so published .pth files remain
        # loadable with this refactored module.
        self.w_omiga = nn.Parameter(torch.randn(hidden_size, 1))
        self.output_projection = (
            nn.Identity()
            if hidden_size == output_size
            else nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_output, _ = self.lstm(inputs)
        hidden = torch.tanh(lstm_output)
        weight = self.w_omiga.unsqueeze(0).expand(hidden.size(0), -1, -1)
        scores = torch.bmm(hidden, weight) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=hidden.dtype, device=hidden.device)
        )
        # ``scores`` has shape (batch, sequence_length, 1).  Normalizing over
        # the last dimension would therefore always return one and disable
        # the attention module.  Normalize over time instead.
        attention_weights = torch.softmax(scores, dim=1)
        context = torch.sum(lstm_output * attention_weights, dim=1)
        return torch.sigmoid(self.output_projection(context))


class MLP(nn.Module):
    """Fully connected critic network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.net_layers = nn.ModuleList()
        self.net_layers.append(
            nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        )
        for _ in range(max(0, num_layers - 1)):
            self.net_layers.append(
                nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
            )
        self.net_layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer in self.net_layers:
            output = layer(output)
        return output


# Backwards-compatible names used by the original script.
LSTMNetClass = AttentionLSTMActor
NetClass = MLP
