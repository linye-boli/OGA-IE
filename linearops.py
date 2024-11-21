import torch

class MCIntegralOperator:
    def __init__(self, K, beta):
        self.K = K 
        self.beta = beta
        self.n = K.shape[1]
    
    def map(self, u):
        return self.beta / self.n * (self.K @ u)
    
    def to(self, device):
        self.K = self.K.to(device)

class CBSOperator:
    def __init__(self, gamma, V, g0):
        self.V = V 
        self.g0 = g0 
        self.gamma = gamma

    def map(self, u):
        return self.gamma * torch.fft.ifftn(self.g0 * torch.fft.fftn(self.V*u)) - self.gamma * u + u

    def to(self, device):
        self.V = self.V.to(device)
        self.g0 = self.g0.to(device)
        self.gamma = self.gamma.to(device)
        