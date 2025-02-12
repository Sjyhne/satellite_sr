import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicLosses(nn.Module):
    """Basic regression loss functions."""
    
    @staticmethod
    def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Error (L1) Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Average absolute difference between predictions and targets
        """
        return torch.abs(pred - target).mean()
    
    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error (L2) Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Average squared difference between predictions and targets
        """
        return torch.square(pred - target)
    
    @staticmethod
    def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Root Mean Squared Error Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Square root of average squared difference between predictions and targets
        """
        return torch.sqrt(torch.square(pred - target).mean())


class AdvancedLosses(nn.Module):
    """Advanced regression loss functions."""
    
    @staticmethod
    def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """Huber Loss (Smooth L1).
        
        Combines MSE and MAE - less sensitive to outliers than MSE.
        
        Args:
            pred: Predicted values
            target: Target values
            delta: Threshold for switching between L1 and L2 loss
            
        Returns:
            Huber loss value
        """
        abs_diff = torch.abs(pred - target)
        quadratic = torch.min(abs_diff, torch.tensor(delta))
        linear = abs_diff - quadratic
        return (0.5 * quadratic.pow(2) + delta * linear).mean()
    
    @staticmethod
    def logcosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Log-Cosh Loss.
        
        Smooth approximation of MAE - combines the best properties of MSE and MAE.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Log-cosh loss value
        """
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff)))
    
    @staticmethod
    def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
        """Quantile Loss.
        
        Used for quantile regression - predicting specific percentiles.
        
        Args:
            pred: Predicted values
            target: Target values
            quantile: Desired quantile (0.5 = median)
            
        Returns:
            Quantile loss value
        """
        diff = target - pred
        return torch.mean(torch.max(quantile * diff, (quantile - 1) * diff))


class WeightedLosses(nn.Module):
    """Weighted variants of common loss functions."""
    
    @staticmethod
    def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, 
                         weights: torch.Tensor) -> torch.Tensor:
        """Weighted Mean Squared Error Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            weights: Weight for each prediction
            
        Returns:
            Weighted MSE loss value
        """
        return (weights * torch.square(pred - target)).mean()
    
    @staticmethod
    def weighted_mae_loss(pred: torch.Tensor, target: torch.Tensor, 
                         weights: torch.Tensor) -> torch.Tensor:
        """Weighted Mean Absolute Error Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            weights: Weight for each prediction
            
        Returns:
            Weighted MAE loss value
        """
        return (weights * torch.abs(pred - target)).mean()


class RelativeLosses(nn.Module):
    """Relative error loss functions."""
    
    @staticmethod
    def relative_l1_loss(pred: torch.Tensor, target: torch.Tensor, 
                        eps: float = 1e-8) -> torch.Tensor:
        """Relative L1 Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            eps: Small constant to avoid division by zero
            
        Returns:
            Relative L1 loss value
        """
        return (torch.abs(pred - target) / (torch.abs(target) + eps)).mean()
    
    @staticmethod
    def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, 
                        eps: float = 1e-8) -> torch.Tensor:
        """Relative L2 Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            eps: Small constant to avoid division by zero
            
        Returns:
            Relative L2 loss value
        """
        return (torch.square(pred - target) / (torch.square(target) + eps)).mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (differentiable variant of L1 loss)."""
    
    def __init__(self, eps: float = 1e-6):
        """Initialize Charbonnier loss.
        
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Charbonnier loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Charbonnier loss value
        """
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


class CombinedLoss(nn.Module):
    """Combination of multiple loss functions."""
    
    def __init__(self, alpha: float = 0.5):
        """Initialize combined loss.
        
        Args:
            alpha: Weight for combining L1 and L2 losses
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined L1-L2 loss.
        
        Args:
            pred: Predicted values
            target: Target values
            
        Returns:
            Weighted combination of L1 and L2 losses
        """
        l1 = torch.abs(pred - target).mean()
        l2 = torch.square(pred - target).mean()
        return self.alpha * l1 + (1 - self.alpha) * l2
