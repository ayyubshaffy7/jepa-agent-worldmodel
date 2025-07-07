from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from torch import nn
import random

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class VICRegEncoderWithoutFlattening(nn.Module):
    def __init__(self, spatial_dim=128, final_channels=32):
        super(VICRegEncoderWithoutFlattening, self).__init__()
        
        # ResNet Backbone for Spatial Features
        
        # Modify the first convolutional layer to accept 2 channels
        self.conv1 = nn.Conv2d(2, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape to merge sequence and batch dimensions
        x = x.view(batch_size * seq_len, channels, height, width)
        #print(f"Input Shape After Reshape: {x.shape}")
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool(x)
        # print("Conv3")
        # print(x.shape)

        x = self.conv4(x)
        # print("Conv4")
        # print(x.shape)
        return x.reshape(batch_size, seq_len, 32, 4, 4)
        
        # return spatial_features

class CNNPredictor(nn.Module):
    def __init__(self, input_channels, action_dim, hidden_channels, output_channels):
        super().__init__()

        # Action to Feature Map
        self.action_to_map = nn.Sequential(
            nn.Linear(action_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )

        # Convolutional Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_channels)
        )

        # Output Layer
        self.output_layer = nn.Conv2d(hidden_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, encoded_feature, action):
        """
        Args:
        - encoded_feature: Tensor of shape (batch_size, channels, h, w).
        - action: Tensor of shape (batch_size, action_dim).

        Returns:
        - predicted_feature: Tensor of shape (batch_size, channels, h, w).
        """
        batch_size, channels, h, w = encoded_feature.shape

        # Map action to spatial feature map
        action_map = self.action_to_map(action)  # Shape: (batch_size, hidden_channels)
        action_map = action_map.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)  # Shape: (batch_size, hidden_channels, h, w)

        # Concatenate encoded features and action map
        x = torch.cat([encoded_feature, action_map], dim=1)  # Shape: (batch_size, channels + hidden_channels, h, w)

        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Final prediction
        predicted_feature = self.output_layer(x)  # Shape: (batch_size, channels, h, w)
        return predicted_feature


class JEPA(nn.Module):
    def __init__(self, encoder, predictor, output_dim=1024, target_encoder=None):
        """
        Args:
        - encoder (nn.Module): Main encoder for observations.
        - predictor (nn.Module): Predictor module for state transitions.
        - target_encoder (nn.Module, optional): Separate target encoder for ground-truth observations. 
          If None, assumes encoder and target_encoder share weights.
        """
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = target_encoder if target_encoder else encoder  # Use shared encoder by default
        self.repr_dim = output_dim

    def forward(self, states, actions, epoch=None, num_epochs=None, teacher_forcing=False, train=False):
        """
        Args:
        - states (Tensor): Observation sequence of shape (batch_size, trajectory_len, C, H, W).
        - actions (Tensor): Action sequence of shape (batch_size, trajectory_len - 1, action_dim).
        - epoch (int, optional): Current training epoch for scheduled teacher forcing.
        - num_epochs (int, optional): Total number of epochs for scheduled teacher forcing.
        - teacher_forcing (bool): Whether to use teacher-forcing (ground-truth states as inputs to the predictor).

        Returns:
        - flattened_representations (Tensor): Flattened states, shape (batch_size, trajectory_len, output_dim_flattened).
        """
        # Encode all observations into representations
        encoded_obs = self.encoder(states)  # Shape: (batch_size, trajectory_len, channels, H, W)
    
        # Separate initial state (s_x) and target future states (s_y)
        s_x = encoded_obs[:, 0]  # Initial state: (batch_size, channels, H, W)
        s_y = encoded_obs[:, 1:]  # Target future states: (batch_size, trajectory_len - 1, channels, H, W)
    
        # Predict future states using the predictor
        s_pred_list = []
        s_prev = s_x  # Start from the initial state
    
        for t in range(actions.size(1)):
            s_prev = self.predictor(s_pred_list[-1] if s_pred_list else s_prev, actions[:, t])
            s_pred_list.append(s_prev)
        s_pred = torch.stack(s_pred_list, dim=1)        

        # Flatten s_x and s_pred spatial dimensions
        batch_size, channels, h, w = s_x.shape
        s_x_flattened = s_x.view(batch_size, -1)  # Shape: (batch_size, channels * H * W)

        batch_size, seq_len_minus_1, channels, h, w = s_pred.shape
        s_pred_flattened = s_pred.view(batch_size, seq_len_minus_1, -1)  # Shape: (batch_size, seq_len - 1, channels * H * W)

        # Concatenate s_x with s_pred along the temporal dimension
        s_x_flattened = s_x_flattened.unsqueeze(1)  # Shape: (batch_size, 1, channels * H * W)
        flattened_representations = torch.cat([s_x_flattened, s_pred_flattened], dim=1)  # Shape: (batch_size, trajectory_len, channels * H * W)

        if train:
            return s_x, s_y, s_pred
        return flattened_representations


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
