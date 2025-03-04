import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class BasicLosses(nn.Module):
    """Basic regression loss functions."""
    
    @staticmethod
    def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Mean Absolute Error (L1) Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            mask: Optional boolean mask for each prediction
            
        Returns:
            Average absolute difference between predictions and targets
        """
        if mask is None:
            return torch.abs(pred - target).mean()
        else:
            return (torch.abs(pred - target) * mask).mean()
    
    @staticmethod
    def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Mean Squared Error (L2) Loss.
        
        Args:
            pred: Predicted values
            target: Target values
            mask: Optional boolean mask for each prediction
            
        Returns:
            Average squared difference between predictions and targets
        """
        if mask is None: 
            return torch.square(pred - target).mean()
        else:
            return (torch.square(pred - target) * mask).mean()
    
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


class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG features."""
    
    def __init__(self, layers: list = None, weights: list = None, resize: bool = True):
        """Initialize perceptual loss.
        
        Args:
            layers: List of VGG layers to extract features from
            weights: Weights for each layer's contribution
            resize: Whether to resize inputs to match VGG input size
        """
        super().__init__()
        
        if layers is None:
            self.layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        else:
            self.layers = layers
            
        if weights is None:
            self.weights = [1.0/len(self.layers)] * len(self.layers)
        else:
            self.weights = weights
            
        self.resize = resize
        
        # Load pretrained VGG model
        vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg_layers = vgg.features
        self.vgg_layers.eval()
        
        # Freeze parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
            
        # Register normalization for VGG input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Map layer names to indices
        self.layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
        }
    
    def _normalize(self, x):
        """Normalize input to match VGG expected range."""
        return (x - self.mean) / self.std
    
    def _get_features(self, x):
        """Extract features from specified VGG layers."""
        if x.shape[1] == 1:  # If grayscale, repeat to make 3 channels
            x = x.repeat(1, 3, 1, 1)
            
        if self.resize and (x.shape[2] != 224 or x.shape[3] != 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        x = self._normalize(x)
        
        features = {}
        for name, idx in self.layer_map.items():
            if name in self.layers:
                for i in range(idx + 1):
                    x = self.vgg_layers[i](x)
                features[name] = x
                
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss.
        
        Args:
            pred: Predicted values (images)
            target: Target values (images)
            
        Returns:
            Perceptual loss value
        """
        # Ensure values are in [0, 1] range
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.clamp(pred, 0, 1)
        if target.min() < 0 or target.max() > 1:
            target = torch.clamp(target, 0, 1)
            
        pred_features = self._get_features(pred)
        target_features = self._get_features(target)
        
        loss = 0.0
        for i, layer in enumerate(self.layers):
            layer_loss = F.mse_loss(pred_features[layer], target_features[layer])
            loss += self.weights[i] * layer_loss
            
        return loss


class GradientDifferenceLoss(nn.Module):
    """Gradient Difference Loss (GDL) for image sharpness."""
    
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Initialize Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_x.weight.data = sobel_x
        self.sobel_y.weight.data = sobel_y
        
        # Freeze the weights
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
        # Note: We'll move the filters to the correct device in the forward pass
    
    def forward(self, pred, target):
        """
        Calculate gradient difference loss between prediction and target.
        
        Args:
            pred: Predicted image tensor [B, C, H, W] or [B, H, W, C]
            target: Target image tensor [B, C, H, W] or [B, H, W, C]
            
        Returns:
            Gradient difference loss
        """
        # Move filters to the same device as input
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        # Check if tensors are in [B, H, W, C] format and convert to [B, C, H, W]
        if pred.shape[-1] in [1, 2, 3, 4] and len(pred.shape) == 4:
            pred = pred.permute(0, 3, 1, 2)
        if target.shape[-1] in [1, 2, 3, 4] and len(target.shape) == 4:
            target = target.permute(0, 3, 1, 2)
        
        # Ensure both tensors have the same number of channels
        if pred.shape[1] != target.shape[1]:
            # If pred has 3 channels and target has 4 (RGGB), convert target to RGB
            if pred.shape[1] == 3 and target.shape[1] == 4:
                # Average the two green channels
                R = target[:, 0:1]
                G = (target[:, 1:2] + target[:, 2:3]) / 2
                B = target[:, 3:4]
                target = torch.cat([R, G, B], dim=1)
            # If pred has 4 channels and target has 3, handle this case too
            elif pred.shape[1] == 4 and target.shape[1] == 3:
                # Convert RGB to RGGB (duplicate green channel)
                R = target[:, 0:1]
                G = target[:, 1:2]
                B = target[:, 2:3]
                target = torch.cat([R, G, G, B], dim=1)
        
        batch_size, channels = pred.shape[0], pred.shape[1]
        
        # Process each channel separately
        grad_diff_x = 0
        grad_diff_y = 0
        
        for c in range(channels):
            # Extract single channel
            pred_c = pred[:, c:c+1]
            target_c = target[:, c:c+1]
            
            # Calculate gradients
            pred_grad_x = self.sobel_x(pred_c)
            pred_grad_y = self.sobel_y(pred_c)
            target_grad_x = self.sobel_x(target_c)
            target_grad_y = self.sobel_y(target_c)
            
            # Calculate gradient differences
            grad_diff_x += torch.abs(pred_grad_x - target_grad_x)
            grad_diff_y += torch.abs(pred_grad_y - target_grad_y)
        
        # Average over channels
        grad_diff_x = grad_diff_x / channels
        grad_diff_y = grad_diff_y / channels
        
        return torch.mean(grad_diff_x + grad_diff_y)
