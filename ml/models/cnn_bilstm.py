import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class CNNBiLSTMModelConfig:
    """Configuration for CNN-BiLSTM model.
    
    Args:
        in_channels: Number of ECG channels (e.g. 2 for two-lead ECG)
        base_out_channels: Out channels for first conv layer
        base_kernel_size: Kernel size for second conv layer
        dropout_rate: Dropout rate to use in the model
        adaptive_pool_size: Sequence length the BiLSTM operates over
        hidden_size: Hidden size for BiLSTM layer
        num_layers: Number of stacked LSTM layers
        lstm_dropout: Dropout rate between LSTM layers
        num_classes: Number of output units (default: 1 for binary classification with BCEWithLogitsLoss)
    """
    in_channels: int
    base_out_channels: int
    base_kernel_size: int
    dropout_rate: float
    adaptive_pool_size: int
    hidden_size: int = 64
    num_layers: int = 2
    lstm_dropout: float = 0.3
    num_classes: int = 1


class CNNBiLSTMModel(nn.Module):
    """
    CNN-BiLSTM model with residual blocks for 1D sequence classification.
    
    Architecture:
    - 3 residual blocks 
    - Pooling layer (4x downsampling)
    - 3 residual blocks 
    - Pooling layer (4x downsampling)
    - 3 residual blocks
    - AdaptiveAvgPool1d (adaptive downsampling)
    - BiLSTM
    - Concatenated mean + max pool
    - 2 fully connected layers
    """
    def __init__(self, config: CNNBiLSTMModelConfig):
        super().__init__()
        self.config = config

        # set out channels for each group
        c1 = config.base_out_channels
        c2 = c1 * 2
        c3 = c2 * 2

        # set kernel sizes for each group
        k1 = config.base_kernel_size + 2
        k2 = config.base_kernel_size
        k3 = config.base_kernel_size - 2

        # set paddings for each group
        p1 = (k1 - 1) // 2
        p2 = (k2 - 1) // 2
        p3 = (k3 - 1) // 2
        
        # group 1: 3 blocks with c1 channels (kernel_size=k1)
        self.block1 = ResidualBlock(in_channels=config.in_channels, out_channels=c1, kernel_size=k1, padding=p1)
        self.block2 = ResidualBlock(in_channels=c1, out_channels=c1, kernel_size=k1, padding=p1)
        self.block3 = ResidualBlock(in_channels=c1, out_channels=c1, kernel_size=k1, padding=p1)

        # pooling and dropout 1
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(p=config.dropout_rate)

        # group 2: 3 blocks with c2 channels (kernel_size=k2)
        self.block4 = ResidualBlock(in_channels=c1, out_channels=c2, kernel_size=k2, padding=p2)
        self.block5 = ResidualBlock(in_channels=c2, out_channels=c2, kernel_size=k2, padding=p2)
        self.block6 = ResidualBlock(in_channels=c2, out_channels=c2, kernel_size=k2, padding=p2)

        # pooling and dropout 2
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(p=config.dropout_rate)

        # group 3: 3 blocks with c3 channels (kernel_size=k3)
        self.block7 = ResidualBlock(in_channels=c2, out_channels=c3, kernel_size=k3, padding=p3)
        self.block8 = ResidualBlock(in_channels=c3, out_channels=c3, kernel_size=k3, padding=p3)
        self.block9 = ResidualBlock(in_channels=c3, out_channels=c3, kernel_size=k3, padding=p3)

        # pooling and dropout 3
        self.pool3 = nn.AdaptiveAvgPool1d(output_size=config.adaptive_pool_size)
        self.dropout3 = nn.Dropout(p=config.dropout_rate)
        
        # BiLSTM (operates across CNN time steps)
        # input: (batch, adaptive_pool_size, c3)
        self.lstm = nn.LSTM(
            input_size=c3,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.num_layers > 1 else 0
        )
        self.lstm_dropout = nn.Dropout(p=config.lstm_dropout)
        
        # fully connected layers (BiLSTM output is 4 * hidden_size due to mean + max pooling)
        lstm_output_size = config.hidden_size * 4
        self.fc1 = nn.Linear(lstm_output_size, 32)
        self.fc2 = nn.Linear(32, config.num_classes)
    
    def _forward_features(self, x):
        """
        Forward pass through convolutional blocks only.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Feature tensor of shape (batch_size, c3, adaptive_pool_size)
        """
        # group 1
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # group 2
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # group 3
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        return x

    def forward(self, x):
        """
        Forward pass through the entire network.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
            Note: Returns logits (not probabilities) for use with BCEWithLogitsLoss
        """
        # CNN features: (batch, c3, adaptive_pool_size)
        x = self._forward_features(x)
        
        # reshape for LSTM: (batch, adaptive_pool_size, c3)
        x = x.transpose(1, 2)
        
        # BiLSTM across time steps
        lstm_out, (h_n, c_n) = self.lstm(x)

        # concatenate mean and max 
        mean_pool = lstm_out.mean(dim=1)         # (batch, 2 * hidden_size)
        max_pool = lstm_out.max(dim=1).values    # (batch, 2 * hidden_size)
        x = torch.cat([mean_pool, max_pool], dim=1)  # (batch, 4 * hidden_size)
        
        x = self.lstm_dropout(x)
        
        # fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

    def predict_proba(self, x):
        """
        Get probability predictions (for inference).
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding)

        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None
    
    def forward(self, x_in):
        """
        Forward pass through residual block.
        
        Args:
            x_in: Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, length)
        """
        # main path
        x = self.conv1(x_in)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        # shortcut connection
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(x_in)
        else:
            shortcut = x_in
        
        # add residual and apply ReLU
        x = x + shortcut
        x = F.relu(x)
        
        return x