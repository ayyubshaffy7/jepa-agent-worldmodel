import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import numpy as np
from models import *


# Dataset class for observations and actions
class JEPADataset(Dataset):
    def __init__(self, observations, actions):
        """
        Args:
        - observations (numpy.ndarray): Shape (num_trajectories, trajectory_len, C, H, W).
        - actions (numpy.ndarray): Shape (num_trajectories, trajectory_len - 1, action_dim).
        """
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = self.observations[idx]  # Observation sequence
        act = self.actions[idx]  # Action sequence
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32)

# Compute JEPA Loss
class JEPALoss(nn.Module):
    def __init__(self, lambda_pred=25.0,lambda_var=25.0, lambda_cov=1.0):
        """
        Args:
        - distance_function (str): Distance function to use ('mse' or 'cosine').
        - lambda_var (float): Weight for variance regularization.
        - lambda_cov (float): Weight for covariance regularization.
        """
        super().__init__()
        self.lambda_pred=lambda_pred
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    def forward(self, s_pred, s_y, s_input, s_target):
        """
        Args:
        - s_pred (Tensor): Predicted states, shape (batch_size, trajectory_len - 1, state_dim).
        - s_y (Tensor): Target states (ground truth), shape (batch_size, trajectory_len - 1, state_dim).
        - s_input (Tensor): Encoded input observations, shape (batch_size, trajectory_len, state_dim).
        - s_target (Tensor): Encoded target observations, shape (batch_size, trajectory_len, state_dim).

        Returns:
        - loss (Tensor): Total JEPA loss.
        """
        # Prediction Loss
        pred_loss = F.mse_loss(s_pred, s_y)  # Mean squared error
        # Variance and Covariance Regularization for Input and Target Representations
        var_loss_input = self.variance_loss(s_input)
        var_loss_target = self.variance_loss(s_target)
        cov_loss_input = self.covariance_loss(s_input,32*4*4)
        cov_loss_target = self.covariance_loss(s_target,32*4*4)

        # Total Loss
        total_loss = (
            self.lambda_pred*pred_loss
            + self.lambda_var * (var_loss_input + var_loss_target)
            + self.lambda_cov * (cov_loss_input + cov_loss_target)
        )
        return total_loss, pred_loss, var_loss_input + var_loss_target, cov_loss_input + cov_loss_target
        
    def variance_loss(self, embeddings, gamma=1.0, epsilon=1e-4):
        """
        Computes variance loss to prevent collapse.
        
        Args:
        - embeddings (Tensor): Encoded representations, shape (batch_size, state_dim).
        - gamma (float): Threshold for the standard deviation (default: 1.0).
        - epsilon (float): Small constant for numerical stability (default: 1e-4).
        
        Returns:
        - var_loss (Tensor): Variance loss (scalar).
        """
        # Compute variance along the batch dimension (dim=0)
        std = torch.sqrt(embeddings.var(dim=0, unbiased=False) + epsilon)  # Shape: [state_dim]
            
        # Apply hinge loss (penalize std < gamma)
        var_loss = torch.mean(F.relu(gamma - std))  # Penalize dimensions with std < gamma
        
        return var_loss

    def covariance_loss(self, embeddings, dim):
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        # Compute covariance matrix (batch_size, state_dim)
        cov_matrix = (embeddings.T @ embeddings) / (embeddings.size(0) - 1)
        # Extract off-diagonal elements and compute their squared sum
        off_diag_loss = self.off_diagonal(cov_matrix).pow(2).sum()
        # Normalize by embedding dimension
        cov_loss = off_diag_loss / dim
        return cov_loss
    
    def off_diagonal(self, matrix):
        n, _ = matrix.size()
        return matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()
if __name__ == "__main__":
    # Load data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observations = np.load("/scratch/DL24FA/train/states.npy", mmap_mode="r")  # Shape: (num_trajectories, trajectory_len, C, H, W)
    actions = np.load("/scratch/DL24FA/train/actions.npy")  # Shape: (num_trajectories, trajectory_len - 1, action_dim)
    dataset = JEPADataset(observations, actions)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=1)
    
    # Initialize Components
# Initialize Components
    encoder = VICRegEncoderWithoutFlattening(spatial_dim=128, final_channels=32).to(device)
    predictor = CNNPredictor(input_channels=32, action_dim=2, hidden_channels=64, output_channels=32).to(device)
    model= JEPA(encoder=encoder, predictor=predictor, output_dim=32).to(device)
    # model.load_state_dict(torch.load('model.pth', weights_only=True))

    # Training Setup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    #model.load_state_dict(torch.load('model.pth', weights_only=True))
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,  # Learning rate
        momentum=0.9,
        weight_decay=1e-3  # Regularization
    )
    # Loss Function
    loss_fn = JEPALoss(lambda_pred=15.0,lambda_var=15.0, lambda_cov=1.0)
    loss_history = {
        "epoch": [],
        "prediction_loss": [],
        "variance_loss": [],
        "covariance_loss": [],
        "total_loss": []
    }

    # Training Loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_pred_loss, epoch_var_loss, epoch_cov_loss, epoch_total_loss = 0.0, 0.0, 0.0, 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Load data
            obs_batch, act_batch = batch
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)

            # Forward pass
            s_x, s_y, s_pred = model(obs_batch, act_batch, train=True)

            # Flatten embeddings for variance and covariance losses

            batch_size, seq_len, channels, height, width = s_y.shape
            s_input = s_x.reshape(batch_size, -1)  # Flatten spatial dimensions: [batch_size, 32 * 5 * 5]
            s_target = s_y.reshape(batch_size * seq_len, -1)  # Combine batch and seq_len dimensions

            # Compute losses
            total_loss, pred_loss, var_loss, cov_loss = loss_fn(s_pred, s_y, s_input, s_target)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update epoch losses
            epoch_pred_loss += pred_loss.item()  # Subtract regularization
            epoch_var_loss += var_loss.item()
            epoch_cov_loss += cov_loss.item()
            epoch_total_loss += total_loss.item()



        # Compute average losses for the epoch
        avg_pred_loss = epoch_pred_loss / len(dataloader)
        avg_var_loss = epoch_var_loss / len(dataloader)
        avg_cov_loss = epoch_cov_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)

        # Log losses
        loss_history["epoch"].append(epoch + 1)
        loss_history["prediction_loss"].append(avg_pred_loss)
        loss_history["variance_loss"].append(avg_var_loss)
        loss_history["covariance_loss"].append(avg_cov_loss)
        loss_history["total_loss"].append(avg_total_loss)

        # Print epoch losses
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Prediction Loss: {avg_pred_loss:.4f}")
        print(f"  Variance Loss: {avg_var_loss:.4f}")
        print(f"  Covariance Loss: {avg_cov_loss:.4f}")
        print(f"  Total Loss: {avg_total_loss:.4f}")
    torch.save(model.state_dict(), 'model.pth')