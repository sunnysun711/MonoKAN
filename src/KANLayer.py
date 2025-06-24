import torch
import torch.nn as nn
import numpy as np
from src.spline import *

class KANLayer(nn.Module):
    """
    KANLayer class
    

    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        num: int
            the number of grid intervals
        k: int
            the piecewise polynomial order of splines
    """
    def __init__(self, 
                 in_dim:int=3, 
                 out_dim:int=2, 
                 num:int=5, 
                 k:int=3, 
                 include_bias:bool=True,
                 grid_range: list[float]=[-1., 1.],
                 device:str | torch.device='cpu'
                 ):
        """initilize a KANLayer

        :param in_dim: input dimension, defaults to 3
        :type in_dim: int, optional
        :param out_dim: output dimension, defaults to 2
        :type out_dim: int, optional
        :param num: the number of grid intervals = G, defaults to 5
        :type num: int, optional
        :param k: the piecewise polynomial order of splines, defaults to 3
        :type k: int, optional
        :param include_bias: whether to include bias in each spline, defaults to True
        :type include_bias: bool, optional
        :param grid_range: the range of grid points, defaults to [-1., 1.]
        :type grid_range: list[float], optional
        :param device: the device to run the model, defaults to 'cpu'
        :type device: str | torch.device, optional
        
        :return: self
        :rtype: KANLayer
        
        Example
        -------
        >>> kan_layer = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device='cuda')
        >>> kan_layer.in_dim, kan_layer.out_dim, kan_layer.num, kan_layer.k, kan_layer.device
        (3, 2, 5, 3, device(type='cuda'))
        """
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1).repeat(self.in_dim, 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        
        noise_scale = 0.5
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))  # initialize coefficients to avoid zero and singularity issues
        
        self.scale_sp = nn.Parameter(torch.ones(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)).requires_grad_(True)
        if include_bias:
            self.scale_base = nn.Parameter(torch.rand(self.in_dim, self.out_dim)*2-1) * 1/np.sqrt(self.in_dim).requires_grad_(True)
            self.base_fun = nn.SiLU()
        else:
            self.scale_base = None
            self.base_fun = nn.Identity()
        self.to(self.device)
        return
    
    def forward(self, x: torch.Tensor):
        """Forward pass of the KANLayer

        :param x: 2D torch.Tensor with shape (batch_size, in_dim)
        :type x: torch.Tensor
        
        :return: 
        :rtype: 
        """
        bs = x.shape[0]
        
        preacts = x[:, None, :].clone().expand(bs, self.out_dim, self.in_dim)
        # preacts = x.repeat(bs, self.out_dim, 1)  # can I replace this with expand?
        
        base = self.base_fun(x)  # bs, in_dim
        y = coef2curve(self.coef, self.grid, coef=self.coef, k=self.k)  # bs, num+1, out_dim
        pass