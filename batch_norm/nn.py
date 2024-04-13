import torch

class Linear:
    """
        Initialize the linear layer with random weights and optional bias.

        Args:
            fan_in (int): Number of input neurons.
            fan_out (int): Number of output neurons.
            bias (bool): Whether to include bias or not.
    """
    def __init__(self, fan_in, fan_out, g, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator= g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1d:
    """
        Initialize the batch normalization layer.

        Args:
            dim (int): Number of input features.
            eps (float): Small value to avoid division by zero.
            momentum (float): Momentum for updating running statistics.
    """
    def __init__(self, dim, eps=1e-5, momentum=0.5):
        self.eps = eps 
        self.momentum = momentum
        
        self.training = True

        # Parameters trained with backprop
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # buffers --> trained with a running momentum update
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # Forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        # Normalize to unit variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        # For the plot and analysis
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
            return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    
class Tanh:
    """Tanh activation function implementation."""
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []