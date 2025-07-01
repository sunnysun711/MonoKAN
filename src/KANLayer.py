import torch
import torch.nn as nn
import numpy as np
from src.spline import extend_grid, curve2coef, coef2curve, B_batch
from src.utils import sparse_mask

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer with configurable monotonicity constraints.
    
    This layer implements univariate functions on edges using B-spline parameterization
    with optional monotonicity constraints for discrete choice modeling applications.

    Parameters
    ----------
    in_dim : int, default=3
        Input dimension.
    out_dim : int, default=2
        Output dimension.
    num : int, default=5
        Number of grid intervals (G).
    k : int, default=3
        Piecewise polynomial order of splines.
    include_basis : bool, default=True
        Whether to include linear basis in each spline. Recommended for faster convergence.
    sparse_init : bool, default=False
        Whether to initialize coefficients with sparse mask to reduce parameters.
    grid_range : list of float, default=[-1., 1.]
        Range of grid points for spline initialization.
    monotonic_dims_dirs : list of tuple, optional
        Monotonic input dimensions and their directions. Each tuple contains
        (dimension_index, direction) where direction is 1 (increasing) or -1 (decreasing).
        Example: [(0, 1), (2, -1)] means dimension 0 is increasing, dimension 2 is decreasing.
    mono_cs_type : str, default='strict'
        Type of monotonicity constraint. Options: 'strict', 'soft'.
        - 'strict': Hard monotonicity constraints using cumulative softplus
        - 'soft': Soft constraints with cumulative ELU activation, instead of softplus.
    elu_alpha : float, default=0.01
        The alpha parameter for ELU activation. Only used when mono_cs_type is 'soft'.
    device : str or torch.device, default='cpu'
        Device to run the model on.
    
    Attributes
    ----------
    grid : torch.nn.Parameter
        Extended grid points for B-spline evaluation, shape (in_dim, G+2k+1).
    coef : torch.nn.Parameter
        B-spline coefficients, shape (in_dim, out_dim, G+k).
    mask : torch.nn.Parameter
        Sparse mask for coefficient selection, shape (in_dim, out_dim).
    scale_sp_raw : torch.nn.Parameter
        Raw scaling parameters for spline outputs.
    scale_base : torch.nn.Parameter
        Scaling parameters for linear basis functions.
    base_mask : torch.Tensor
        Mask to remove basis functions from monotonic dimensions.
    monotonic_dims_dirs : list of tuple
        List of tuples containing monotonic dimension indices and directions.
    monotonic_dims : list of int
        List of monotonic dimension indices extracted from monotonic_dims_dirs.
    mono_cs_type : str
        Type of monotonicity constraint being applied.
    elu_alpha : float
        Alpha parameter for ELU activation function in soft constraints.
    base_fun : torch.nn.Module
        Base activation function (SiLU or Identity).
    include_basis : bool
        Whether linear basis functions are included.
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
                 mono_cs_type: str = 'strict',
                 elu_alpha: float = 0.01,
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
        :param mono_cs_type: str, default='strict'
            Type of monotonicity constraint. Options: 'strict', 'soft', 'segment'.
            - 'strict': Hard monotonicity constraints using cumulative softplus
            - 'soft': Soft constraints with tolerance for violations
            - 'segment': Allow violations in specified proportion of segments
        :type mono_cs_type: str, optional
        :param elu_alpha: the alpha parameter for ELU activation, defaults to 0.01
            Only used when mono_cs_type is 'soft'.
        :type elu_alpha: float, optional

        :param device: the device to run the model, defaults to 'cpu'
        :type device: str | torch.device, optional
        
        :return: self
        :rtype: KANLayer
        
        Example::

            >>> from src.KANLayer import KANLayer, torch
            
            # Basic initialization
            >>> kan_layer = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device='cuda')
            >>> kan_layer.in_dim, kan_layer.out_dim, kan_layer.num, kan_layer.k, kan_layer.device
            (3, 2, 5, 3, device(type='cuda'))

            # With monotonic constraints
            >>> kan_mono = KANLayer(in_dim=4, out_dim=3, num=8, k=3, 
            ...                     monotonic_dims_dirs=[(0, 1), (2, -1), (3, 1)],
            ...                     mono_cs_type='soft')
            >>> kan_mono.monotonic_dims_dirs
            [(0, 1), (2, -1), (3, 1)]
            >>> kan_mono.mono_cs_type
            'soft'

            # With sparse initialization
            >>> kan_sparse = KANLayer(in_dim=5, out_dim=2, sparse_init=True, 
            ...                       grid_range=[-2., 2.], include_basis=False)
            >>> kan_sparse.mask.sum() < kan_sparse.mask.numel()  # Should have some zeros
            tensor(True)

            # Different constraint types
            >>> kan_strict = KANLayer(monotonic_dims_dirs=[(0, 1)], mono_cs_type='strict')
            >>> kan_soft = KANLayer(monotonic_dims_dirs=[(1, -1)], mono_cs_type='soft')
            >>> [kan_strict.mono_cs_type, kan_soft.mono_cs_type]
            ['strict', 'soft']
        """
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.monotonic_dims_dirs = monotonic_dims_dirs or []  # default to empty list if not provided
        self.monotonic_dims = [d[0] for d in self.monotonic_dims_dirs]
        
        # Validate monotonicity constraint type
        valid_mono_types = {'strict', 'soft'}  # TODO currently only support two types
        if mono_cs_type not in valid_mono_types:
            raise ValueError(f"mono_cs_type must be one of {valid_mono_types}, got {mono_cs_type}")
        self.mono_cs_type: str = mono_cs_type
        self.elu_alpha: float = elu_alpha
        
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

        # cache variables for quick calculation of regularization
        self._cache_spline_outputs = None
        self._cache_constrained_coef = None
        self._cache_input_batch = None
        
        self.to(self.device)
        return
    
    def clear_cache(self):
        """Clear cached intermediate results to free memory."""
        self._cache_spline_outputs = None
        self._cache_constrained_coef = None
        self._cache_input_batch = None
    
    def get_cached_outputs(self) -> dict:
        """Get cached intermediate outputs for debugging/analysis."""
        return {
            'spline_outputs': self._cache_spline_outputs,
            'constrained_coef': self._cache_constrained_coef,
            'input_batch': self._cache_input_batch
        }
    
    def _apply_cumulative_monotonic(self, coef: torch.Tensor) -> torch.Tensor:
        """
        Apply monotonicity constraints using cumulative activation transformation.
        
        This method enforces monotonicity constraints by applying an activation function
        to ensure non-negative (or non-positive) differences between consecutive B-spline
        coefficients, followed by cumulative summation.
        
        This method uses ELU activation with alpha=self.elu_alpha for soft constraints 
        when self.mono_cs_type='soft', and uses softplus for strict constraints when 
        self.mono_cs_type='strict'.
        
        Parameters
        ----------
        coef : torch.Tensor
            B-spline coefficients with shape (in_dim, out_dim, n_coef).
        

        Returns
        -------
        torch.Tensor
            Monotonicity-constrained coefficients with same shape as input.
            
        Notes
        -----
        Mathematical formulation:
        - For softplus: delta_i = softplus(coef_i), constrained_coef = cumsum(delta)
        - For ELU: delta_i = elu(coef_i, alpha), constrained_coef = cumsum(delta)
        - For decreasing: constrained_coef = -cumsum(delta)
        
        Examples::
            
            >>> import torch
            >>> from src.KANLayer import KANLayer
            
            # Test with strict monotonic constraints (increasing)
            >>> kan_strict = KANLayer(in_dim=2, out_dim=1, num=4, k=2,
            ...                       monotonic_dims_dirs=[(0, 1)], mono_cs_type='strict')
            >>> coef_input = torch.randn(2, 1, 6)
            >>> coef_constrained = kan_strict._apply_cumulative_monotonic(coef_input)
            >>> coef_constrained.shape
            torch.Size([2, 1, 6])
            
            # Verify increasing constraint: differences should be non-negative
            >>> diffs = torch.diff(coef_constrained[0, 0, :])
            >>> (diffs >= 0).all().item()
            True
            
            # Test with soft monotonic constraints (decreasing)
            >>> kan_soft = KANLayer(in_dim=3, out_dim=2, num=5, k=3,
            ...                     monotonic_dims_dirs=[(1, -1)], mono_cs_type='soft', elu_alpha=1e-10)  # very small number to enforce almost monotonicity
            >>> coef_multi = torch.randn(3, 2, 8)
            >>> coef_soft_constrained = kan_soft._apply_cumulative_monotonic(coef_multi)
            >>> coef_soft_constrained.shape
            torch.Size([3, 2, 8])
            
            # Verify decreasing constraint for dimension 1, should be True (still possible to be False)
            >>> diffs_decreasing = torch.diff(coef_soft_constrained[1, 0, :])
            >>> (diffs_decreasing <= 0).all().item()
            True
            
            # Test softer example:
            >>> kan_soft = KANLayer(in_dim=3, out_dim=2, num=5, k=3,
            ...                     monotonic_dims_dirs=[(1, -1)], mono_cs_type='soft', elu_alpha=0.01)
            >>> coef_multi = torch.randn(3, 2, 8)
            >>> coef_soft_constrained = kan_soft._apply_cumulative_monotonic(coef_multi)
            >>> diffs_decreasing = torch.diff(coef_soft_constrained[1, 0, :])
            >>> (diffs_decreasing <= 0).all().item()  # Very likely would be False
            False
            
            # Test with multiple constraints
            >>> kan_multi = KANLayer(in_dim=4, out_dim=2, num=6, k=3,
            ...                      monotonic_dims_dirs=[(0, 1), (2, -1), (3, 1)],
            ...                      mono_cs_type='strict')
            >>> coef_multi_input = torch.randn(4, 2, 9)
            >>> coef_multi_constrained = kan_multi._apply_cumulative_monotonic(coef_multi_input)
            
            # Check increasing constraints for dims 0 and 3
            >>> (torch.diff(coef_multi_constrained[0, 0, :]) >= 0).all().item()
            True
            >>> (torch.diff(coef_multi_constrained[3, 0, :]) >= 0).all().item()
            True
            
            # Check decreasing constraint for dim 2
            >>> (torch.diff(coef_multi_constrained[2, 0, :]) <= 0).all().item()
            True
            
            # Test with no constraints (should return unchanged)
            >>> kan_no_constraints = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[])
            >>> coef_unchanged = torch.randn(2, 1, 5)
            >>> coef_result = kan_no_constraints._apply_cumulative_monotonic(coef_unchanged)
            >>> torch.allclose(coef_unchanged, coef_result)
            True
        """
        if len(self.monotonic_dims_dirs) == 0:
            return coef
        
        coef = coef.permute(1, 0, 2)  # (out_dim, in_dim, n_coef)
        new_coef_list = []
        
        for j in range(coef.shape[0]):  # iterate over output dimensions
            coef_j = coef[j]  # (in_dim, n_coef)
            for dim, direction in self.monotonic_dims_dirs:
                # Apply activation function based on type
                if self.mono_cs_type == 'strict':
                    delta = torch.nn.functional.softplus(coef_j[dim])
                elif self.mono_cs_type == 'soft':
                    delta = torch.nn.functional.elu(coef_j[dim], alpha=self.elu_alpha)
                else:
                    raise ValueError(f"Unsupported mono_cs_type: {self.mono_cs_type}. "
                                f"Supported types: 'strict', 'soft'")
                
                # Cumulative sum to enforce monotonicity
                delta = torch.cumsum(delta, dim=-1)
                if direction == -1:
                    delta = -delta  # reverse for decreasing monotonicity
                
                coef_j = coef_j.clone()
                coef_j[dim] = delta
            new_coef_list.append(coef_j)
            
        coef = torch.stack(new_coef_list, dim=0).permute(1, 0, 2)
        return coef
    
    def _reverse_cumulative_monotonic(self, coef_constrained: torch.Tensor) -> torch.Tensor:
        """
        Reverse the cumulative monotonic transformation to recover effective coefficients.

        :param coef_constrained: Monotonicity-constrained coefficients with shape (in_dim, out_dim, n_coef).
        :type coef_constrained: torch.Tensor

        :returns: Recovered effective coefficients with same shape as input.
        :rtype: torch.Tensor

        .. note::
            The reverse transformation follows:

            For strict mode (softplus):

            1. constrained = direction * cumsum(softplus(raw))
            2. diff(constrained) = direction * softplus(raw[1:])
            3. raw[1:] = softplus_inverse(direction * diff(constrained))

            For soft mode (ELU):

            1. constrained = direction * cumsum(elu(raw, alpha))
            2. diff(constrained) = direction * elu(raw[1:], alpha)
            3. raw[1:] = elu_inverse(direction * diff(constrained), alpha)

        Example::
    
            >>> kan = KANLayer(in_dim=2, out_dim=1, num=4, k=2,
            ...                monotonic_dims_dirs=[(0, 1)], mono_cs_type='strict')
            >>> original_coef = torch.randn(2, 1, 6)
            >>> constrained_coef = kan._apply_cumulative_monotonic(original_coef)
            >>> recovered_coef = kan._reverse_cumulative_monotonic(constrained_coef)
            >>> recovered_coef.shape
            torch.Size([2, 1, 6])

            >>> kan_soft = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[(0, 1)],
            ...                     mono_cs_type='soft')
            >>> constrained_soft = kan_soft._apply_cumulative_monotonic(original_coef)
            >>> recovered_soft = kan_soft._reverse_cumulative_monotonic(constrained_soft)
            >>> recovered_soft.shape
            torch.Size([2, 1, 6])
        """
        if len(self.monotonic_dims_dirs) == 0:
            return coef_constrained
        
        coef_effective = coef_constrained.clone()
        coef_constrained = coef_constrained.permute(1, 0, 2)  # (out_dim, in_dim, n_coef)
        
        def stable_softplus_inverse(x: torch.Tensor) -> torch.Tensor:
            """Numerically stable inverse of softplus function."""
            x = torch.clamp(x, min=1e-8)
            mask = x > 20
            result = torch.zeros_like(x)
            result[mask] = x[mask]
            result[~mask] = torch.log(torch.expm1(x[~mask]))
            return result
        
        def stable_elu_inverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
            """Numerically stable inverse of ELU function."""
            mask = x >= 0
            result = torch.zeros_like(x)
            result[mask] = x[mask]  # For x >= 0, elu_inv(x) = x
            result[~mask] = torch.log(x[~mask] / alpha + 1)  # For x < 0
            return result
        
        new_coef_list = []
        for j in range(coef_constrained.shape[0]):  # output dimensions
            coef_j = coef_constrained[j].clone()  # (in_dim, n_coef)
            
            for dim, direction in self.monotonic_dims_dirs:
                constrained_seq = coef_constrained[j, dim, :]
                
                # Apply direction correction
                if direction == -1:
                    constrained_seq = -constrained_seq
                
                # Compute differences (reverse of cumsum)
                if len(constrained_seq) > 1:
                    differences = torch.diff(constrained_seq)
                    
                    # Apply inverse activation based on constraint type
                    if self.mono_cs_type == 'strict':
                        raw_diffs = stable_softplus_inverse(differences)
                    elif self.mono_cs_type == 'soft':
                        raw_diffs = stable_elu_inverse(differences, self.elu_alpha)
                    else:
                        raise ValueError(f"Unsupported mono_cs_type: {self.mono_cs_type}")
                    
                    # Reconstruct effective coefficients
                    effective_coef = torch.zeros_like(constrained_seq)
                    effective_coef[0] = constrained_seq[0]  # Keep first as anchor
                    effective_coef[1:] = raw_diffs
                    
                    coef_j[dim] = effective_coef
            
            new_coef_list.append(coef_j)
        
        coef_effective = torch.stack(new_coef_list, dim=0).permute(1, 0, 2)
        return coef_effective
    
    
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the KANLayer with configurable monotonicity constraints.

        :param torch.Tensor x: 2D torch.Tensor with shape (batch_size, in_dim)
        
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
        
        Example::
        
            # Basic forward pass
            >>> kan = KANLayer(in_dim=3, out_dim=2, num=5, k=3, device='cpu')
            >>> x = torch.randn(10, 3)  # batch_size=10, in_dim=3
            >>> y, preacts, postacts, postspline = kan(x)
            >>> y.shape, preacts.shape, postacts.shape, postspline.shape
            (torch.Size([10, 2]), torch.Size([10, 2, 3]), torch.Size([10, 2, 3]), torch.Size([10, 2, 3]))

            # Forward pass with strict monotonic constraints
            >>> kan_strict = KANLayer(in_dim=4, out_dim=3, monotonic_dims_dirs=[(0, 1), (2, -1)], 
            ...                       mono_cs_type='strict')
            >>> x_strict = torch.randn(5, 4)
            >>> y_strict, *_ = kan_strict(x_strict)
            >>> y_strict.shape
            torch.Size([5, 3])

            # Forward pass with soft constraints and custom parameters
            >>> kan_soft = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[(0, 1)], 
            ...                     mono_cs_type='soft')
            >>> x_soft = torch.randn(8, 2)
            >>> y_soft, *_ = kan_soft(x_soft)
            >>> y_soft.shape
            torch.Size([8, 1])

            # Test gradient flow
            >>> kan_grad = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[(0, 1)])
            >>> x_grad = torch.randn(3, 2, requires_grad=True)
            >>> y_grad, *_ = kan_grad(x_grad)
            >>> loss = y_grad.sum()
            >>> loss.backward()
            >>> x_grad.grad is not None
            True

            # Single sample forward pass
            >>> x_single = torch.randn(1, 3)
            >>> y_single, *_ = kan(x_single)
            >>> y_single.shape
            torch.Size([1, 2])
        """
        bs = x.shape[0]
        
        preacts = x[:, None, :].clone().expand(bs, self.out_dim, self.in_dim)
        
        # ===  monotonic coef ===
        # Apply monotonicity constraints based on selected type
        # Clone coefficients to avoid in-place modifications during gradient computation
        coef = self.coef.clone()
        if self.mono_cs_type in ['strict', 'soft']:
            coef = self._apply_cumulative_monotonic(coef)
        # No constraint application for invalid types (already validated in __init__)
        
        # Cache intermediate results for regularization
        self._cache_constrained_coef = coef.detach()
        self._cache_input_batch = x.detach()

        y = coef2curve(x, self.grid, coef=coef, k=self.k)  # bs, in_dim, out_dim
        
        # Cache spline outputs for regularization (before scaling and masking)
        self._cache_spline_outputs = y.detach()
        
        postspline = y.clone().permute(0, 2, 1)  # bs, out_dim, in_dim
        
        scale_sp = torch.nn.functional.softplus(self.scale_sp_raw)  # in_dim, out_dim
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
    
    
    def regularization_loss(
        self, 
        lambda_l1: float = 1.0, 
        lambda_coef: float = 0.0,
        lambda_entropy: float = 2.0, 
        lambda_smoothness: float = 0.0
    ) -> torch.Tensor:
        """            
        Compute the regularization loss using cached intermediate results from the forward pass.
        .. note::
            You must call :meth:`forward` before invoking this method to ensure the cache is populated.
        
        :param float lambda_l1: Weight for L1 regularization on activations. Default is 1.0.
        :param float lambda_coef: Weight for L1 regularization on coefficients. Default is 0.0.
        :param float lambda_entropy: Weight for entropy regularization (bidirectional: row + column). Default is 2.0.
        :param float lambda_smoothness: Weight for coefficient smoothness regularization. Default is 0.0.
        
        :returns torch.Tensor: Scalar regularization loss.
        
        :raises RuntimeError: If :meth:`forward` has not been called to populate the cache.
        
        Example::
        
            >>> kan = KANLayer(in_dim=3, out_dim=2, num=5, k=3)
            >>> x = torch.randn(10, 3)
            >>> y, *_ = kan(x)
            >>> reg_loss = kan.regularization_loss()
            >>> isinstance(reg_loss, torch.Tensor)
            True
            >>> reg_loss.dim()
            0

            # With custom regularization weights
            >>> reg_loss_custom = kan.regularization_loss(lambda_l1=0.5, lambda_coef=0.2, lambda_entropy=1.0, lambda_smoothness=0.1)
            >>> isinstance(reg_loss_custom, torch.Tensor)
            True

            # With monotonic constraints
            >>> kan_mono = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[(0, 1)], mono_cs_type='strict')
            >>> x_mono = torch.randn(5, 2)
            >>> y_mono, *_ = kan_mono(x_mono)
            >>> reg_loss_mono = kan_mono.regularization_loss()
            >>> reg_loss_mono.item() >= 0
            True

            # Raises error if forward not called
            >>> kan_unrun = KANLayer(in_dim=2, out_dim=1)
            >>> try:
            ...     _ = kan_unrun.regularization_loss()
            ... except RuntimeError as e:
            ...     'Must call forward()' in str(e)
            True
        """
        if self._cache_spline_outputs is None:
            raise RuntimeError("Must call forward() before regularization_loss() to populate cache")
        
        spline_outputs = self._cache_spline_outputs  # (batch, in_dim, out_dim)
        
        # L1 regularization on activations and coefficients
        activation_l1 = torch.abs(spline_outputs).mean()
        coef_l1 = torch.abs(self.coef).mean()
        
        # Entropy regularization for diverse activation patterns
        activation_strength = torch.abs(spline_outputs).mean(dim=0).permute(1, 0)  # (out_dim, in_dim)
        # Row-wise entropy (each output dimension's distribution across input dimensions)
        p_row = activation_strength / (activation_strength.sum(dim=1, keepdim=True) + 1e-8)
        entropy_row = -(p_row * torch.log(p_row + 1e-8)).sum(dim=1).mean()
        # Column-wise entropy (each input dimension's distribution across output dimensions)
        p_col = activation_strength / (activation_strength.sum(dim=0, keepdim=True) + 1e-8)
        entropy_col = -(p_col * torch.log(p_col + 1e-8)).sum(dim=0).mean()
        
        # Total entropy (bidirectional)
        entropy_total = entropy_row + entropy_col
        
        # Coefficient smoothness regularization
        smoothness_loss = 0.0
        if lambda_smoothness > 0:
            coef_diff = torch.diff(self.coef, dim=-1)
            smoothness_loss = torch.abs(coef_diff).mean()
        
        return (lambda_l1 * activation_l1 + 
                lambda_coef * coef_l1 + 
                lambda_entropy * entropy_total + 
                lambda_smoothness * smoothness_loss)

    def update_grid(self, x: torch.Tensor, margin: float = 0.01):
        """
        NOTE: NOT TESTED YET.
        """
        assert x.dim() == 2 and x.size(1) == self.in_dim
        batch = x.size(0)
        
        # 计算当前输出
        with torch.no_grad():
            splines = B_batch(x, self.grid, k=self.k)
            
            # 获取当前系数
            coef = self.coef.clone()
            if self.monotonic_dims_dirs:
                coef = self._apply_cumulative_monotonic(coef)
            
            # 计算未约简的样条输出
            unreduced_output = torch.einsum('bik,jok->bijo', splines, coef)
        
        # 按通道排序收集数据分布
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.num + 1, dtype=torch.int64, device=x.device)
        ]
        
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.num
        grid_uniform = (
            torch.arange(self.num + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step + x_sorted[0] - margin
        )
        
        # 混合自适应和均匀网格
        grid_eps = 0.02
        grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive
        
        # 扩展网格
        grid = torch.concatenate([
            grid[:1] - uniform_step * torch.arange(self.k, 0, -1, device=x.device).unsqueeze(1),
            grid,
            grid[-1:] + uniform_step * torch.arange(1, self.k + 1, device=x.device).unsqueeze(1),
        ], dim=0)
        
        # 更新网格和系数
        self.grid.copy_(grid.T)
        # 重新拟合系数
        new_coef = curve2coef(x, unreduced_output.mean(dim=3).permute(0, 2, 1), self.grid, self.k)
        self.coef.data.copy_(new_coef)
    
    def get_eval_coefficients(self) -> dict[str, torch.Tensor]:
        """
        Get coefficients for evaluation/analysis purposes.
        
        This method returns coefficients in evaluation mode (no gradient computation)
        with options to get either constrained coefficients (used by the model) or
        effective coefficients (recovered for symbolic analysis).
            
        Returns
        -------
        dict
            Dictionary containing coefficient information:
            - 'raw_coef': Original learnable coefficients
            - 'constrained_coef': Coefficients after applying monotonic constraints  
            - 'effective_coef': Recovered effective coefficients (only for 'strict' and 'soft' constraints, None otherwise)
            - 'constraint_info': Information about applied constraints
            - 'scaling_info': Scaling parameters information
            
        Examples::

            >>> import torch
            >>> kan = KANLayer(in_dim=3, out_dim=2, num=5, k=3,
            ...                monotonic_dims_dirs=[(0, 1), (2, -1)], mono_cs_type='strict')

            # Basic coefficient extraction
            >>> coef_info = kan.get_eval_coefficients()
            >>> set(coef_info.keys()) >= {'raw_coef', 'constrained_coef', 'constraint_info', 'scaling_info'}
            True
            >>> coef_info['raw_coef'].shape
            torch.Size([3, 2, 8])
            >>> coef_info['constrained_coef'].shape  
            torch.Size([3, 2, 8])

            # Extract effective coefficients for strict constraints
            >>> coef_with_effective = kan.get_eval_coefficients()
            >>> 'effective_coef' in coef_with_effective
            True
            >>> coef_with_effective['effective_coef'].shape
            torch.Size([3, 2, 8])

            # Check constraint information
            >>> constraint_info = coef_info['constraint_info']
            >>> constraint_info['type']
            'strict'
            >>> constraint_info['has_constraints']
            True
            >>> len(constraint_info['monotonic_dims'])
            2

            # Check scaling information
            >>> scaling_info = coef_info['scaling_info']
            >>> scaling_info['scale_sp'].shape
            torch.Size([3, 2])
            >>> scaling_info['mask'].shape
            torch.Size([3, 2])

            # Check dimension analysis
            >>> dim_analysis = coef_info['dimension_analysis']
            >>> dim_analysis['dim_0']['is_monotonic']
            True
            >>> dim_analysis['dim_0']['direction_str']
            'increasing'
            >>> dim_analysis['dim_1']['is_monotonic']
            False
            >>> dim_analysis['dim_2']['direction_str']
            'decreasing'

            # Test with soft constraints and custom parameters
            >>> kan_soft = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[(0, 1)], 
            ...                     mono_cs_type='soft')
            >>> soft_coef = kan_soft.get_eval_coefficients()
            >>> soft_coef['constraint_info']['type']
            'soft'

            # Test without constraints
            >>> kan_no_constraints = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[])
            >>> no_constraint_coef = kan_no_constraints.get_eval_coefficients()
            >>> no_constraint_coef['constraint_info']['has_constraints']
            False
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            result = {}
            
            # Original raw coefficients
            result['raw_coef'] = self.coef.clone()
            
            # Apply constraints based on type
            if self.mono_cs_type in ['strict', 'soft']:
                constrained_coef = self._apply_cumulative_monotonic(self.coef.clone())
            else:  # 'none' or no constraints
                constrained_coef = self.coef.clone()
            
            result['constrained_coef'] = constrained_coef
            
            # Recover effective coefficients if requested
            if self.mono_cs_type in ['strict', 'soft']:
                effective_coef = self._reverse_cumulative_monotonic(constrained_coef)
                result['effective_coef'] = effective_coef
            else:
                result['effective_coef'] = None
            
            # Constraint information
            result['constraint_info'] = {
                'type': self.mono_cs_type,
                'monotonic_dims': self.monotonic_dims_dirs,
                'has_constraints': len(self.monotonic_dims_dirs) > 0
            }
            
            # Scaling information
            result['scaling_info'] = {
                'scale_sp': torch.nn.functional.softplus(self.scale_sp_raw).clone(),
                'scale_base': self.scale_base.clone() if self.include_basis else None,
                'base_mask': self.base_mask.clone() if self.include_basis else None,
                'mask': self.mask.clone()
            }
            
            # Per-dimension analysis
            result['dimension_analysis'] = {}
            for in_dim in range(self.in_dim):
                is_monotonic = in_dim in self.monotonic_dims
                direction = None
                if is_monotonic:
                    direction = next(d[1] for d in self.monotonic_dims_dirs if d[0] == in_dim)
                
                result['dimension_analysis'][f'dim_{in_dim}'] = {
                    'is_monotonic': is_monotonic,
                    'direction': direction,
                    'direction_str': 'increasing' if direction == 1 else 'decreasing' if direction == -1 else 'none'
                }
        
        return result
    
    def analyze_monotonicity_compliance(self) -> dict[str, float]:
        """
        Analyze how well the current coefficients comply with monotonicity constraints.
            
        Returns
        -------
        dict
            Compliance metrics for each monotonic dimension.
            
        Examples::
            
            >>> import torch
            >>> kan = KANLayer(in_dim=3, out_dim=2, num=6, k=3,
            ...                monotonic_dims_dirs=[(0, 1), (2, -1)], mono_cs_type='strict')

            # Basic compliance analysis
            >>> compliance = kan.analyze_monotonicity_compliance()
            >>> len(compliance)  # Should have entries for each (monotonic_dim, output_dim) pair
            4
            >>> 'dim_0_to_out_0' in compliance
            True
            >>> 'dim_2_to_out_1' in compliance  
            True

            # Check compliance metrics structure
            >>> metric = compliance['dim_0_to_out_0']
            >>> set(metric.keys()) >= {'compliance_ratio', 'violation_count', 'total_segments', 'violation_severity', 'direction'}
            True
            >>> metric['direction']
            'increasing'
            >>> 0.0 <= metric['compliance_ratio'] <= 1.0
            True

            # Test with perfect compliance (strict constraints should give perfect compliance)
            >>> perfect_compliance = all(m['compliance_ratio'] == 1.0 for m in compliance.values())
            >>> perfect_compliance  # Strict constraints should achieve perfect compliance
            True

            # Test decreasing constraint compliance
            >>> decreasing_metric = compliance['dim_2_to_out_0']
            >>> decreasing_metric['direction']
            'decreasing'
            >>> decreasing_metric['violation_count']
            0.0

            # Test with soft constraints (should allow some violations)
            >>> kan_soft = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[(0, 1)], 
            ...                     mono_cs_type='soft', elu_alpha=0.1)
            >>> # Train briefly to get realistic coefficients
            >>> x_train = torch.randn(100, 2)
            >>> y_train = torch.randn(100, 1)
            >>> for _ in range(10):
            ...     y_pred, _, _, _ = kan_soft(x_train)
            ...     loss = torch.nn.functional.mse_loss(y_pred, y_train)
            ...     if hasattr(loss, 'backward'):
            ...         break
            >>> soft_compliance = kan_soft.analyze_monotonicity_compliance()
            >>> 'dim_0_to_out_0' in soft_compliance
            True

            # Test with no constraints
            >>> kan_no_mono = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=[])
            >>> no_mono_compliance = kan_no_mono.analyze_monotonicity_compliance()
            >>> len(no_mono_compliance)
            0

            # Test compliance with multiple output dimensions
            >>> kan_multi = KANLayer(in_dim=4, out_dim=3, monotonic_dims_dirs=[(0, 1), (1, -1), (3, 1)])
            >>> multi_compliance = kan_multi.analyze_monotonicity_compliance()
            >>> len(multi_compliance)  # 3 monotonic dims × 3 output dims = 9 entries
            9
        """
        self.eval()
        
        with torch.no_grad():
            compliance_metrics = {}
            
            coef_info = self.get_eval_coefficients()
            constrained_coef = coef_info['constrained_coef']
            
            for dim, direction in self.monotonic_dims_dirs:
                for out_dim in range(self.out_dim):
                    key = f'dim_{dim}_to_out_{out_dim}'
                    coef_seq = constrained_coef[dim, out_dim, :]
                    
                    # Calculate differences
                    diffs = torch.diff(coef_seq)
                    
                    # Count violations
                    if direction == 1:  # should be increasing
                        violations = (diffs < 0).float()
                    else:  # should be decreasing
                        violations = (diffs > 0).float()
                    
                    # Compliance metrics
                    total_segments = len(diffs)
                    violation_count = violations.sum().item()
                    compliance_ratio = 1.0 - (violation_count / total_segments)
                    
                    # Severity of violations
                    violation_severity = 0.0
                    if violation_count > 0:
                        if direction == 1:
                            violation_magnitudes = torch.abs(diffs[diffs < 0])
                        else:
                            violation_magnitudes = torch.abs(diffs[diffs > 0])
                        violation_severity = violation_magnitudes.mean().item()
                    
                    compliance_metrics[key] = {
                        'compliance_ratio': compliance_ratio,
                        'violation_count': violation_count,
                        'total_segments': total_segments,
                        'violation_severity': violation_severity,
                        'direction': 'increasing' if direction == 1 else 'decreasing'
                    }
            
            return compliance_metrics