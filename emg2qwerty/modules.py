# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
from torch import Tensor


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(self.out_dim, num_features)
        self.layer_norm = nn.LayerNorm(num_features)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, C)
        x, _ = self.lstm(inputs)  # (T, N, out_dim)
        x = self.proj(x)  # (T, N, C)
        x = x + inputs  # residual
        return self.layer_norm(x) 

class GRUEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            # By default, PyTorch GRU expects input as (seq_len, batch, features).
            # If your input is (batch, seq_len, features), add batch_first=True here.
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(self.out_dim, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (seq_len, batch_size, num_features) by default for nn.GRU
        x, _ = self.gru(inputs)  # (seq_len, batch_size, out_dim)
        x = self.proj(x)  # Project back to (seq_len, batch_size, num_features)
        x = x + inputs  # Residual connection
        return self.layer_norm(x)

class TransformerEncoderLayer(nn.Module):
    """
    Simplified TransformerEncoderLayer
    """

    def __init__(
        self,
        num_features: int,
        nhead: int=8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(num_features,nhead, dropout=dropout, batch_first=False)  # CHANGED
        self.linear1 = nn.Linear(num_features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, num_features)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    """
    TransformerEncoder with a stack of TransformerEncoderLayer
    """

    def __init__(
      self,
      num_features: int,
      nhead: int=8,
      num_layers: int=2,
      dim_feedforward: int = 2048,
      dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(num_features, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output



class Hybrid_CNN_LSTMEncoder(nn.Module):
    """A hybrid encoder combining TDSConvEncoder and LSTMEncoder.

    Args:
        num_features (int): `num_features` for an input of shape (T, N, num_features).
        tds_block_channels (Sequence[int]): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        tds_kernel_width (int): The kernel size of the temporal convolutions for TDSConvEncoder.
        hidden_size (int): The number of features in the hidden state of the LSTM.
        num_layers (int): Number of recurrent layers for LSTM.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
        dropout (float): If non-zero, introduces a Dropout layer on the outputs
            of each LSTM layer except the last one.
    """

    def __init__(
        self,
        num_features: int,
        tds_block_channels: Sequence[int] = (24, 24, 24, 24),
        tds_kernel_width: int = 32,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tds_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=tds_block_channels,
            kernel_width=tds_kernel_width,
        )

        self.lstm_encoder = LSTMEncoder(
            num_features=num_features, # Output features of TDSConvEncoder are input features for LSTMEncoder
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # First, pass through the TDSConvEncoder
        tds_output = self.tds_encoder(inputs)

        # Then, pass the output of TDSConvEncoder to the LSTMEncoder
        lstm_output = self.lstm_encoder(tds_output)

        return lstm_output



class Hybrid_CNN_GRUEncoder(nn.Module):
    """A hybrid encoder combining TDSConvEncoder and GRUEncoder.

    Args:
        num_features (int): `num_features` for an input of shape (T, N, num_features).
        tds_block_channels (Sequence[int]): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        tds_kernel_width (int): The kernel size of the temporal convolutions for TDSConvEncoder.
        gru_hidden_size (int): The number of features in the hidden state of the GRU.
        gru_num_layers (int): Number of recurrent layers for GRU.
        gru_bidirectional (bool): If True, becomes a bidirectional GRU.
        gru_dropout (float): If non-zero, introduces a Dropout layer on the outputs
            of each GRU layer except the last one.
    """

    def __init__(
        self,
        num_features: int,
        tds_block_channels: Sequence[int] = (24, 24, 24, 24),
        tds_kernel_width: int = 32,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tds_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=tds_block_channels,
            kernel_width=tds_kernel_width,
        )

        self.gru_encoder = GRUEncoder(
            num_features=num_features, # Output features of TDSConvEncoder are input features for GRUEncoder
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # First, pass through the TDSConvEncoder
        tds_output = self.tds_encoder(inputs)

        # Then, pass the output of TDSConvEncoder to the GRUEncoder
        gru_output = self.gru_encoder(tds_output)

        return gru_output


class Hybrid_CNN_TransformerEncoder(nn.Module):
    """A hybrid encoder combining TDSConvEncoder and TransformerEncoder.

    Args:
        num_features (int): `num_features` for an input of shape (T, N, num_features).
        tds_block_channels (Sequence[int]): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        tds_kernel_width (int): The kernel size of the temporal convolutions for TDSConvEncoder.
        transformer_nhead (int): The number of attention heads in the TransformerEncoder.
        transformer_num_layers (int): The number of TransformerEncoderLayer modules.
        transformer_dim_feedforward (int): The dimension of the feedforward network model in TransformerEncoder.
        transformer_dropout (float): The dropout value for TransformerEncoder.
    """

    def __init__(
        self,
        num_features: int,
        tds_block_channels: Sequence[int] = (24, 24, 24, 24),
        tds_kernel_width: int = 32,
        transformer_nhead: int = 8,
        num_layers: int = 2,
        transformer_dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.tds_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=tds_block_channels,
            kernel_width=tds_kernel_width,
        )

        self.transformer_encoder = TransformerEncoder(
            num_features=num_features, # Output features of TDSConvEncoder are input features for TransformerEncoder
            nhead=transformer_nhead,
            num_layers=num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # First, pass through the TDSConvEncoder
        tds_output = self.tds_encoder(inputs)
        transformer_output = self.transformer_encoder(tds_output)  # CHANGED

        return transformer_output  # CHANGED

class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)

class Hybrid_CNN_RNNEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        conv_channels: int = 256,
        hidden_size: int = 384,
        num_layers: int = 2,
        bidirectional: bool = True,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=conv_channels,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rnn = nn.RNN(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, C)
        residual = inputs

        # (T, N, C) -> (N, C, T)
        x = inputs.permute(1, 2, 0)

        # CNN over time
        x = self.conv(x)  # (N, conv_channels, T)

        # (N, conv_channels, T) -> (T, N, conv_channels)
        x = x.permute(2, 0, 1)

        # RNN over time
        x, _ = self.rnn(x)  # (T, N, out_dim)

        # back to original feature dim
        x = self.proj(x)  # (T, N, C)

        # residual + norm
        x = x + residual
        return self.layer_norm(x)