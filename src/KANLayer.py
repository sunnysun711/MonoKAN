import torch
import torch.nn as nn
import numpy as np
from src.spline import extend_grid, curve2coef, coef2curve
from src.utils import sparse_mask

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
        device: torch.device
            the device to run the model
    """
    def __init__(self, 
                 in_dim:int=3, 
                 out_dim:int=2, 
                 num:int=5, 
                 k:int=3, 
                 include_basis:bool=True,
                 sparse_init:bool=False,
                 grid_range: list[float]=[-1., 1.],
                 monotonic_dims_dirs: list[tuple[int, int]] | None = None,
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
        :param sparse_init: whether to initialize the coefficients with sparse mask, defaults to False.
            If True, the coefficients will be initialized with a sparse mask, which can help to reduce the number of parameters.
        :type sparse_init: bool, optional
        :param include_basis: whether to include basis in each spline, defaults to True, including a linear basis in the output (instead of just the splines).
            To speed up the convergence, it is recommended to include the basis.
        :type include_basis: bool, optional
        :param grid_range: the range of grid points, defaults to [-1., 1.]
        :type grid_range: list[float], optional
        :param monotonic_dims_dirs: the dimensions of monotonic input dimensions and their directions, defaults to None
            If provided, it should be a list of tuples, where each tuple contains two integers: (dimension index, direction).
            
            The direction can be either 1 (increasing) or -1 (decreasing).
            
            For example, [(0, 1), (2, -1)] means that the first dimension is increasing and the third dimension is decreasing.
        :type monotonic_dims_dirs: list[tuple[int, int]] | None, optional
        :param device: the device to run the model, defaults to 'cpu'
        :type device: str | torch.device, optional
        
        :return: self
        :rtype: KANLayer
        
        Example
        -------
        >>> from src.KANLayer import KANLayer
        >>> kan_layer = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device='cuda')
        >>> kan_layer.in_dim, kan_layer.out_dim, kan_layer.num, kan_layer.k, kan_layer.device
        (3, 2, 5, 3, device(type='cuda'))
        
        >>> kan = KANLayer(in_dim=3, out_dim=2, num=5, k=3, monotonic_dims_dirs=[(0, 1), (2, -1)])
        >>> kan.monotonic_dims_dirs
        [(0, 1), (2, -1)]
        """
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.monotonic_dims_dirs = monotonic_dims_dirs or []  # default to empty list if not provided
        self.monotonic_dims = [d[0] for d in self.monotonic_dims_dirs]
        
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1).repeat(self.in_dim, 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        
        noise_scale = 0.5
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num
        # Initialize B-spline coefficients by fitting the (possibly noisy) curve values to the B-spline basis using least squares
        self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_sp_raw = nn.Parameter(torch.ones(self.in_dim, self.out_dim) / np.sqrt(self.in_dim)).requires_grad_(True)
        
        self.include_basis = include_basis
        if include_basis:
            self.scale_base = nn.Parameter((torch.rand(self.in_dim, self.out_dim)*2-1) * 1/np.sqrt(self.in_dim)).requires_grad_(True)
            self.base_fun = nn.SiLU()
            self.base_mask = torch.ones(self.in_dim, self.out_dim).to(self.device)
            for dim, _ in self.monotonic_dims_dirs:
                self.base_mask[dim, :] = 0.0  # remove basis from monotonic dimensions
        else:
            self.scale_base = nn.Parameter(torch.zeros(self.in_dim, self.out_dim)).requires_grad_(False)
            self.base_fun = nn.Identity()

        self.to(self.device)
        return
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the KANLayer

        :param x: 2D torch.Tensor with shape (batch_size, in_dim)
        :type x: torch.Tensor
        
        :return: a tuple of (y, preacts, postacts, postspline)

            - y: outputs 
                2D torch.Tensor with shape (batch_size, out_dim)
            - preacts: fan out x into activations
                3D torch.Tensor with shape (batch_size, out_dim, in_dim)
            - postacts: 
                3D torch.Tensor with shape (batch_size, out_dim, in_dim)
            - postspline: 
                3D torch.Tensor with shape (batch_size, out_dim, in_dim)
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        
        Example
        -------
        >>> from src.KANLayer import KANLayer, torch
        >>> kan = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device='cpu')
        >>> x = torch.rand(10, 3)  # batch_size=10, in_dim=3
        >>> y, preacts, postacts, postspline = kan(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([10, 2]), torch.Size([10, 2, 3]), torch.Size([10, 2, 3]), torch.Size([10, 2, 3]))

        >>> kan = KANLayer(in_dim=3, out_dim=2, num=5, k=3, monotonic_dims_dirs=[(0, 1), (2, -1)])
        >>> x = torch.rand(10, 3)  # batch_size=10, in_dim=3
        >>> y, preacts, postacts, postspline = kan(x)
        """
        bs = x.shape[0]
        
        preacts = x[:, None, :].clone().expand(bs, self.out_dim, self.in_dim)
        
        # ===  monotonic coef ===
        coef = self.coef.clone()

        if len(self.monotonic_dims_dirs) > 0:
            coef = coef.permute(1, 0, 2)  # (out_dim, in_dim, n_coef)

            new_coef_list = []
            for j in range(coef.shape[0]):  # out_dim
                coef_j = coef[j]  # (in_dim, n_coef)
                for dim, direction in self.monotonic_dims_dirs:
                    delta = torch.nn.functional.softplus(coef_j[dim])
                    delta = torch.cumsum(delta, dim=-1)
                    if direction == -1:
                        delta = -delta
                    coef_j = coef_j.clone()
                    coef_j[dim] = delta
                new_coef_list.append(coef_j)

            coef = torch.stack(new_coef_list, dim=0)  # (out_dim, in_dim, n_coef)
            coef = coef.permute(1, 0, 2)  # back to (in_dim, out_dim, n_coef)
        
        
        y = coef2curve(x, self.grid, coef=coef, k=self.k)  # bs, in_dim, out_dim
        
        postspline = y.clone().permute(0, 2, 1)  # bs, out_dim, in_dim
        
        scale_sp = torch.nn.functional.softplus(self.scale_sp_raw)
        y = scale_sp[None, :, :] * y
        
        if self.include_basis:
            # Add the linear basis to the output
            base = self.base_mask * self.scale_base[None, :, :] * self.base_fun(x)[:, :, None]
            y += base  # bs, in_dim, out_dim
        
        y = self.mask[None, :, :] * y
        
        postacts = y.clone().permute(0, 2, 1)  # bs, out_dim, in_dim
        
        y = torch.sum(y, dim=1)  # bs, out_dim
        
        ###### Debugging Information ######
        # with torch.no_grad():
        #     print("=== DEBUG: KANLayer Forward ===")
        #     # print("x sample:", x[0])
        #     print("preacts sample:", preacts[0])
        #     print("base sample:", base[0])
        #     print("postspline sample:", postspline[0])
        #     print("scale_base:", self.scale_base)
        #     print("scale_sp (softplused):", scale_sp)
        #     # print("mask:", self.mask)
        #     print("y output sample:", y[0])
        #     print("coef of x2 (after softplus+cumsum):", coef[2])
        
        return y, preacts, postacts, postspline