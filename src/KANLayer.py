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
        Type of monotonicity constraint. Options: 'strict', 'soft', 'segment'.
        - 'strict': Hard monotonicity constraints using cumulative softplus
        - 'soft': Soft constraints with tolerance for violations
        - 'segment': Allow violations in specified proportion of segments
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
        :param device: the device to run the model, defaults to 'cpu'
        :type device: str | torch.device, optional
        
        :return: self
        :rtype: KANLayer
        
        Example
        -------
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
        >>> kan_segment = KANLayer(monotonic_dims_dirs=[(1, -1)], mono_cs_type='segment')
        >>> [kan_strict.mono_cs_type, kan_segment.mono_cs_type]
        ['strict', 'segment']
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
        valid_mono_types = {'strict', 'soft', 'segment'}
        if mono_cs_type not in valid_mono_types:
            raise ValueError(f"mono_cs_type must be one of {valid_mono_types}, got {mono_cs_type}")
        self.mono_cs_type = mono_cs_type
        
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
    
    
    def _apply_strict_monotonic(self, coef:torch.Tensor) -> torch.Tensor:
        """Apply strict monotonicity constraints using cumulative softplus transformation.
        
        This method enforces hard monotonicity constraints by ensuring that the differences
        between consecutive B-spline coefficients are non-negative (for increasing) or
        non-positive (for decreasing) through cumulative summation of softplus-transformed
        differences.
        
        :param coef: B spline coefficients, shape (in_dim, out_dim, G+k)
        :type coef: torch.Tensor
        
        :return: the coefficients with strict monotonicity, shape (batch_size, in_dim, out_dim)
        :rtype: torch.Tensor
        
        Examples
        --------
        >>> from src.KANLayer import torch, KANLayer
        >>> kan = KANLayer(in_dim=2, out_dim=1, num=4, k=2, 
        ...                monotonic_dims_dirs=[(0, 1), (1, -1)])
        >>> coef = torch.randn(2, 1, 6)  # (in_dim=2, out_dim=1, n_coef=6)
        >>> constrained_coef = kan._apply_strict_monotonic(coef)
        >>> constrained_coef.shape
        torch.Size([2, 1, 6])

        # Check monotonicity for increasing dimension (dim=0)
        >>> diffs_dim0 = torch.diff(constrained_coef[0, 0, :])
        >>> (diffs_dim0 >= 0).all().item()  # Should be monotonic increasing
        True

        # Check monotonicity for decreasing dimension (dim=1) 
        >>> diffs_dim1 = torch.diff(constrained_coef[1, 0, :])
        >>> (diffs_dim1 <= 0).all().item()  # Should be monotonic decreasing
        True

        # No constraints case
        >>> kan_no_mono = KANLayer(in_dim=2, out_dim=1, monotonic_dims_dirs=None)
        >>> original_coef = torch.randn(2, 1, 6)
        >>> result_coef = kan_no_mono._apply_strict_monotonic(original_coef)
        >>> torch.allclose(original_coef, result_coef)
        True
        """
        if len(self.monotonic_dims_dirs) == 0:
            return coef
        
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
        coef = torch.stack(new_coef_list, dim=0).permute(1, 0, 2)
        return coef
    
    def _reverse_strict_monotonic(self, coef_constrained: torch.Tensor) -> torch.Tensor:
        """
        Reverse the strict monotonic transformation to recover effective coefficients.
        
        This method reverses the cumsum(softplus()) transformation applied in 
        _apply_strict_monotonic to recover the original B-spline coefficients that 
        represent the actual function shape before constraint enforcement.
        
        Parameters
        ----------
        coef_constrained : torch.Tensor
            Monotonicity-constrained coefficients with shape (in_dim, out_dim, n_coef).
            
        Returns
        -------
        torch.Tensor
            Recovered effective coefficients with same shape as input.
            
        Notes
        -----
        The reverse transformation follows:
        1. constrained = direction * cumsum(softplus(raw))
        2. diff(constrained) = direction * softplus(raw[1:])
        3. raw[1:] = softplus_inverse(direction * diff(constrained))
        
        Examples
        --------
        >>> from src.KANLayer import torch, KANLayer
        >>> kan = KANLayer(in_dim=2, out_dim=1, num=4, k=2,
        ...                monotonic_dims_dirs=[(0, 1)])
        
        # Forward and reverse transformation
        >>> original_coef = torch.randn(2, 1, 6)
        >>> constrained_coef = kan._apply_strict_monotonic(original_coef)
        >>> recovered_coef = kan._reverse_strict_monotonic(constrained_coef)
        >>> recovered_coef.shape
        torch.Size([2, 1, 6])

        # Non-monotonic dimensions should be unchanged in reverse
        >>> torch.allclose(original_coef[1], recovered_coef[1], atol=1e-6)
        True

        # Test with multiple output dimensions
        >>> kan_multi = KANLayer(in_dim=3, out_dim=2, monotonic_dims_dirs=[(0, 1), (2, -1)])
        >>> coef_multi = torch.randn(3, 2, 8)
        >>> constrained_multi = kan_multi._apply_strict_monotonic(coef_multi)
        >>> recovered_multi = kan_multi._reverse_strict_monotonic(constrained_multi)
        >>> recovered_multi.shape
        torch.Size([3, 2, 8])

        # Verify numerical stability with extreme values
        >>> extreme_coef = torch.tensor([[[100., -100., 50., -50., 25.]]])  # (1, 1, 5)
        >>> kan_extreme = KANLayer(in_dim=1, out_dim=1, monotonic_dims_dirs=[(0, 1)])
        >>> stable_result = kan_extreme._reverse_strict_monotonic(
        ...     kan_extreme._apply_strict_monotonic(extreme_coef))
        >>> torch.isfinite(stable_result).all().item()
        True
        """
        if len(self.monotonic_dims_dirs) == 0:
            return coef_constrained
        
        coef_effective = coef_constrained.clone()
        coef_constrained = coef_constrained.permute(1, 0, 2)  # (out_dim, in_dim, n_coef)
        
        def stable_softplus_inverse(x: torch.Tensor) -> torch.Tensor:
            """Numerically stable inverse of softplus function."""
            x = torch.clamp(x, min=1e-8)
            # For large x, softplus_inv(x) ≈ x
            # For small x, use log(exp(x) - 1)
            mask = x > 20
            result = torch.zeros_like(x)
            result[mask] = x[mask]
            result[~mask] = torch.log(torch.expm1(x[~mask]))
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
                    
                    # Apply inverse softplus to recover raw coefficients
                    raw_diffs = stable_softplus_inverse(differences)
                    
                    # Reconstruct effective coefficients
                    # First coefficient could be the initial value before cumsum
                    effective_coef = torch.zeros_like(constrained_seq)
                    effective_coef[0] = constrained_seq[0]  # Keep first as anchor
                    effective_coef[1:] = raw_diffs
                    
                    coef_j[dim] = effective_coef
            
            new_coef_list.append(coef_j)
        
        coef_effective = torch.stack(new_coef_list, dim=0).permute(1, 0, 2)
        return coef_effective
    
    def _apply_segmented_monotonic(self, coef: torch.Tensor, violation_segments: float = 0.3) -> torch.Tensor:
        """
        
        NOTE: not tested, not available now.
        
        Apply segmented monotonicity constraints allowing violations in specified proportion.
        
        This method allows monotonicity violations in a controlled proportion of segments
        while enforcing constraints on the remaining segments. It prioritizes preserving
        the most significant violations to maintain model flexibility.
        
        :param coef: B-spline coefficients with shape (in_dim, out_dim, n_coef).
        :type coef: torch.Tensor
        :param violation_segments: Proportion of segments allowed to violate monotonicity (0.0 to 1.0).
        :type violation_segments: float, optional

        :return: Segmented monotonicity-constrained coefficients with same shape as input.
        :rtype: torch.Tensor
        """
        if len(self.monotonic_dims_dirs) == 0:
            return coef
        
        coef = coef.permute(1, 0, 2)
        new_coef_list = []
        
        for j in range(coef.shape[0]):
            coef_j = coef[j].clone()
            for dim, direction in self.monotonic_dims_dirs:
                original_coef = coef_j[dim]
                n_coef = len(original_coef)
                
                # Calculate current violations
                diffs = torch.diff(original_coef)
                if direction == 1:
                    violations = diffs < 0
                else:
                    violations = diffs > 0
                
                violation_ratio = violations.float().mean()
                
                # Adjust if violation ratio exceeds allowed threshold
                if violation_ratio > violation_segments:
                    # Preserve most severe violations, adjust others
                    violation_scores = torch.abs(diffs) * violations.float()
                    _, sorted_indices = torch.sort(violation_scores, descending=True)
                    
                    # Calculate number of violations to preserve
                    keep_violations = int(violation_segments * (n_coef - 1))
                    keep_mask = torch.zeros_like(violations)
                    if keep_violations > 0:
                        keep_mask[sorted_indices[:keep_violations]] = True
                    
                    # Fix violations not being preserved
                    fix_mask = violations & (~keep_mask)
                    
                    if fix_mask.any():
                        adjusted_coef = original_coef.clone()
                        for i in range(1, n_coef):
                            if fix_mask[i-1]:
                                if direction == 1:
                                    adjusted_coef[i] = adjusted_coef[i-1] + 1e-6
                                else:
                                    adjusted_coef[i] = adjusted_coef[i-1] - 1e-6
                        coef_j[dim] = adjusted_coef
            
            new_coef_list.append(coef_j)
        
        return torch.stack(new_coef_list, dim=0).permute(1, 0, 2)
    
    def _apply_soft_monotonic(self, coef: torch.Tensor, elu_alpha: float = 0.1) -> torch.Tensor:
        """
        Apply soft monotonicity constraints using ELU activation instead of softplus.
        
        This method uses ELU activation which allows negative values down to -alpha,
        providing a softer constraint compared to strict softplus-based constraints.
        
        Parameters
        ----------
        coef : torch.Tensor
            B-spline coefficients with shape (in_dim, out_dim, n_coef).
        elu_alpha : float, default=0.1
            The alpha parameter for ELU activation, controlling the minimum negative value.
            Larger alpha allows more negative values (softer constraint).

        Returns
        -------
        torch.Tensor
            Soft monotonicity-constrained coefficients with same shape as input.
            
        Examples
        --------
        >>> import torch
        >>> kan = KANLayer(in_dim=2, out_dim=1, num=4, k=2,
        ...                monotonic_dims_dirs=[(0, 1), (1, -1)], mono_cs_type='soft')

        # Test with default ELU alpha
        >>> coef = torch.randn(2, 1, 6)
        >>> soft_coef = kan._apply_soft_monotonic(coef)
        >>> soft_coef.shape
        torch.Size([2, 1, 6])

        # Test with custom ELU alpha (softer constraint)
        >>> softer_coef = kan._apply_soft_monotonic(coef, elu_alpha=2.0)
        >>> softer_coef.shape
        torch.Size([2, 1, 6])

        # Check that soft constraints allow some violations
        >>> diffs_soft = torch.diff(soft_coef[0, 0, :])  # Should be mostly increasing
        >>> violations_soft = (diffs_soft < -0.1).sum().item()
        >>> diffs_strict = torch.diff(kan._apply_strict_monotonic(coef)[0, 0, :])
        >>> violations_strict = (diffs_strict < 0).sum().item()
        >>> violations_soft >= violations_strict  # Soft should allow more violations
        True

        # Test decreasing monotonicity
        >>> diffs_decreasing = torch.diff(soft_coef[1, 0, :])
        >>> mostly_decreasing = (diffs_decreasing <= 0.1).float().mean() > 0.7
        >>> mostly_decreasing.item()
        True

        # No constraints case
        >>> kan_no_mono = KANLayer(monotonic_dims_dirs=[])
        >>> unchanged_coef = kan_no_mono._apply_soft_monotonic(coef)
        >>> torch.allclose(coef, unchanged_coef)
        True
        """
        if len(self.monotonic_dims_dirs) == 0:
            return coef
        
        coef = coef.permute(1, 0, 2)  # (out_dim, in_dim, n_coef)
        new_coef_list = []
        
        for j in range(coef.shape[0]):  # iterate over output dimensions
            coef_j = coef[j]  # (in_dim, n_coef)
            for dim, direction in self.monotonic_dims_dirs:
                # Apply ELU to ensure differences are mostly non-negative but allow some negative values
                delta = torch.nn.functional.elu(coef_j[dim], alpha=elu_alpha)
                # Cumulative sum to enforce monotonicity with soft constraint
                delta = torch.cumsum(delta, dim=-1)
                if direction == -1:
                    delta = -delta  # reverse for decreasing monotonicity
                coef_j = coef_j.clone()
                coef_j[dim] = delta
            new_coef_list.append(coef_j)
            
        coef = torch.stack(new_coef_list, dim=0).permute(1, 0, 2)
        return coef
    
    def forward(self, x: torch.Tensor, **constraint_kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the KANLayer with configurable monotonicity constraints.

        :param x: 2D torch.Tensor with shape (batch_size, in_dim)
        :type x: torch.Tensor
        :param constraint_kwargs: Additional parameters for constraint methods:
            - tolerance : float, for 'soft' constraints (default=0.1)
            - violation_segments : float, for 'segment' constraints (default=0.3)
        :type constraint_kwargs: dict, optional
        
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
        >>> y_soft, *_ = kan_soft(x_soft, elu_alpha=0.5)
        >>> y_soft.shape
        torch.Size([8, 1])

        # Forward pass with segmented constraints
        >>> kan_seg = KANLayer(in_dim=3, out_dim=2, monotonic_dims_dirs=[(1, -1)], 
        ...                    mono_cs_type='segment')
        >>> x_seg = torch.randn(6, 3)
        >>> y_seg, *_ = kan_seg(x_seg, violation_segments=0.2)
        >>> y_seg.shape
        torch.Size([6, 2])

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
        
        if self.mono_cs_type == 'strict':
            coef = self._apply_strict_monotonic(coef)
        elif self.mono_cs_type == 'soft':
            elu_alpha = constraint_kwargs.get('elu_alpha', 0.1)
            coef = self._apply_soft_monotonic(coef, elu_alpha=elu_alpha)
        elif self.mono_cs_type == 'segment':
            violation_segments = constraint_kwargs.get('violation_segments', 0.3)
            coef = self._apply_segmented_monotonic(coef, violation_segments=violation_segments)
        # No constraint application for invalid types (already validated in __init__)

        y = coef2curve(x, self.grid, coef=coef, k=self.k)  # bs, in_dim, out_dim
        
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
    
    
    def regularization_loss(self, x: torch.Tensor, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> torch.Tensor:
        """
        NOTE: NOT TESTED YET.
        """
        with torch.no_grad():
            b_splines = B_batch(x, self.grid, k=self.k)
            
            # 应用约束后的系数
            coef = self.coef.clone()
            if self.mono_cs_type == 'strict':
                coef = self._apply_strict_monotonic(coef)
            elif self.mono_cs_type == 'soft':
                coef = self._apply_soft_monotonic(coef)
            
            # 计算激活输出
            spline_outputs = torch.einsum('bik,jok->bjo', b_splines, coef)
        
        # 1. 激活值L1正则化
        activation_l1 = torch.abs(spline_outputs).mean()
        
        # 2. 权重L1正则化
        weight_l1 = torch.abs(self.coef).mean()
        
        # 3. 熵正则化
        activation_strength = torch.abs(spline_outputs).mean(dim=0)  # (out_dim, in_dim)
        activation_strength = activation_strength / (activation_strength.sum(dim=1, keepdim=True) + 1e-8)
        entropy_reg = -(activation_strength * torch.log(activation_strength + 1e-8)).sum()
        
        return regularize_activation * (activation_l1 + weight_l1) - regularize_entropy * entropy_reg

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
            if self.mono_cs_type == 'strict':
                coef = self._apply_strict_monotonic(coef)
            elif self.mono_cs_type == 'soft':
                coef = self._apply_soft_monotonic(coef)
            
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
    
    def get_eval_coefficients(self, return_effective: bool = False, **constraint_kwargs) -> dict[str, torch.Tensor]:
        """
        Get coefficients for evaluation/analysis purposes.
        
        This method returns coefficients in evaluation mode (no gradient computation)
        with options to get either constrained coefficients (used by the model) or
        effective coefficients (recovered for symbolic analysis).
        
        Parameters
        ----------
        return_effective : bool, default=False
            If True, returns effective coefficients recovered from constraints.
            If False, returns constrained coefficients as used by the model.
        **constraint_kwargs : dict
            Additional parameters for constraint methods (tolerance, violation_segments).
            
        Returns
        -------
        dict
            Dictionary containing coefficient information:
            - 'raw_coef': Original learnable coefficients
            - 'constrained_coef': Coefficients after applying monotonic constraints  
            - 'effective_coef': Recovered effective coefficients (if return_effective=True)
            - 'constraint_info': Information about applied constraints
            - 'scaling_info': Scaling parameters information
            
        Examples
        --------
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
        >>> coef_with_effective = kan.get_eval_coefficients(return_effective=True)
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
        >>> soft_coef = kan_soft.get_eval_coefficients(return_effective=True, elu_alpha=0.3)
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
            if self.mono_cs_type == 'strict':
                constrained_coef = self._apply_strict_monotonic(self.coef.clone())
            elif self.mono_cs_type == 'soft':
                elu_alpha = constraint_kwargs.get('elu_alpha', 0.1)
                constrained_coef = self._apply_soft_monotonic(self.coef.clone(), elu_alpha=elu_alpha)
            elif self.mono_cs_type == 'segment':
                violation_segments = constraint_kwargs.get('violation_segments', 0.3)
                constrained_coef = self._apply_segmented_monotonic(self.coef.clone(), violation_segments=violation_segments)
            else:  # 'none' or no constraints
                constrained_coef = self.coef.clone()
            
            result['constrained_coef'] = constrained_coef
            
            # Recover effective coefficients if requested
            if return_effective and self.mono_cs_type == 'strict':
                effective_coef = self._reverse_strict_monotonic(constrained_coef)
                result['effective_coef'] = effective_coef
            elif return_effective and self.mono_cs_type != 'strict':
                # For non-strict constraints, effective ≈ constrained
                result['effective_coef'] = constrained_coef
            
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
    
    def analyze_monotonicity_compliance(self, **constraint_kwargs) -> dict[str, float]:
        """
        Analyze how well the current coefficients comply with monotonicity constraints.
        
        Parameters
        ----------
        **constraint_kwargs : dict
            Additional parameters for constraint methods.
            
        Returns
        -------
        dict
            Compliance metrics for each monotonic dimension.
            
        Examples
        --------
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
        ...                     mono_cs_type='soft')
        >>> # Train briefly to get realistic coefficients
        >>> x_train = torch.randn(100, 2)
        >>> y_train = torch.randn(100, 1)
        >>> for _ in range(10):
        ...     y_pred, _, _, _ = kan_soft(x_train, elu_alpha=1.0)
        ...     loss = torch.nn.functional.mse_loss(y_pred, y_train)
        ...     if hasattr(loss, 'backward'):
        ...         break
        >>> soft_compliance = kan_soft.analyze_monotonicity_compliance(elu_alpha=1.0)
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
            
            coef_info = self.get_eval_coefficients(**constraint_kwargs)
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