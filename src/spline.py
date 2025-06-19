import torch
from torch import Tensor


# ========== Shared Helper Functions ==========


def _unsqueeze_inputs(x: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
    """Prepare tensors for batch operations by adding dimensions.

    :param x: Input tensor of shape (num_splines, num_samples)
    :type x: torch.Tensor
    :param grid: Grid tensor of shape (num_splines, num_grid_points)
    :type grid: torch.Tensor
    :return: Tuple of reshaped tensors

        - `x_unsqueezed`: `(num_splines, 1, num_samples)`
        - `grid_unsqueezed`: `(num_splines, num_grid_points, 1)`

    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    return x.unsqueeze(dim=1), grid.unsqueeze(dim=2)


def _compute_B_km1(x: Tensor, grid: Tensor, k: int) -> Tensor:
    """Compute B-spline basis of order k-1.

    :param x: Input tensor of shape (num_splines, num_samples)
    :type x: torch.Tensor
    :param grid: Grid tensor of shape (num_splines, num_grid_points)
    :type grid: torch.Tensor
    :param k: Order of spline
    :type k: int
    :return: B-spline basis of order k-1
    :rtype: torch.Tensor
    :raises AssertionError: If result is not a Tensor
    """
    B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
    assert isinstance(B_km1, Tensor), "B_km1 must be a Tensor"
    return B_km1


def _safe_nan_zeroing(value: Tensor) -> Tensor:
    mask = torch.isnan(value)
    if mask.any():
        value = torch.where(mask, torch.zeros_like(value), value)
    return value


# ========== Grid Extension Functions ==========


def extend_grid(grid: Tensor, k_extend: int = 0, repeat: bool = False) -> Tensor:
    """Extend grid points for B-spline basis.

    :param grid: Grid tensor of shape (num_splines, num_grid_points)
    :type grid: torch.Tensor
    :param k_extend: Number of points to extend on both ends, defaults to 0
    :type k_extend: int, optional
    :param repeat: If True, repeat the first and last grid points, defaults to False
    :type repeat: bool, optional
    :return: Extended grid tensor of shape (num_splines, num_grid_points + 2 * k_extend)
    :rtype: torch.Tensor
    """
    if repeat:
        for _ in range(k_extend):
            grid = torch.cat([grid[:, [0]], grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]]], dim=1)
    else:
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for _ in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


# ========== B-spline Basis ==========


def B_batch(
    x: Tensor,
    grid: Tensor,
    k: int = 0,
    extend: bool = True,
    return_extended: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Evaludate x on B-spline bases

    :param x: inputs, shape (number of splines, number of samples)
    :type x: 2D torch.tensor

    :param grid: grids, shape (number of splines, number of grid points)
    :type grid: 2D torch.tensor

    :param k: the piecewise polynomial order of splines.
    :type k: int

    :param extend: If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
    :type extend: bool, optional

    :param return_extended: If True, return (value, grid). If False, return value. Default: False
    :type return_extended: bool, optional

    :return: shape (number of splines, number of B-spline bases (coeffcients), number of samples).
        The number of B-spline bases = number of grid points + k - 1.
    :rtype: 3D torch.tensor

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    """
    assert (
        x.device == grid.device
    ), f"x device: {x.device} != grid device: {grid.device}"
    if extend:
        grid = extend_grid(grid, k_extend=k)
    x, grid = _unsqueeze_inputs(x, grid)

    if k == 0:
        value = (x >= grid[:, :-1]) & (x < grid[:, 1:])
    else:
        B_km1 = _compute_B_km1(x, grid, k)
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:-k]
        ) * B_km1[
            :, 1:
        ]

    return (value, grid) if return_extended else value


def B_batch_mod(
    x: Tensor, grid: Tensor, k: int = 0, extend: bool = True, return_extended=False
):
    assert (
        x.device == grid.device
    ), f"x device: {x.device} != grid device: {grid.device}"
    if extend:
        grid = extend_grid(grid, k_extend=k, repeat=True)
    x, grid = _unsqueeze_inputs(x, grid)

    if k == 0:
        value = (x >= grid[:, :-1]) & (x < grid[:, 1:])
    else:
        B_km1 = _compute_B_km1(x, grid, k)
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:-k]
        ) * B_km1[
            :, 1:
        ]

    value = _safe_nan_zeroing(value)
    return (value, grid) if return_extended else value


# ========== Derivatives ==========


def der_B_batch(
    x: Tensor,
    grid: Tensor,
    k: int = 0,
    extend: bool = True,
    return_extended: bool = False,
):
    if extend:
        grid = extend_grid(grid, k_extend=k)
    x, grid = _unsqueeze_inputs(x, grid)

    if k == 0:
        return 0
    B_km1 = _compute_B_km1(x, grid, k)
    value = (
        k / (grid[:, k:-1] - grid[:, : -(k + 1)]) * B_km1[:, :-1]
        - k / (grid[:, k + 1 :] - grid[:, 1:-k]) * B_km1[:, 1:]
    )
    return (value, grid) if return_extended else value


def der2_B_batch(
    x: Tensor,
    grid: Tensor,
    k: int = 0,
    extend: bool = True,
    return_extended: bool = False,
):
    """Compute the second derivative of B-spline basis functions with respect to x.

    :param x: Input tensor of shape (number of splines, number of samples)
    :type x: torch.Tensor
    :param grid: Knot vector of shape (number of splines, number of grid points)
    :type grid: torch.Tensor
    :param k: Order of the B-spline, defaults to 0
    :type k: int, optional
    :param extend: Whether to extend the grid, defaults to True
    :type extend: bool, optional
    :param return_extended: Whether to return the extended grid, defaults to False
    :type return_extended: bool, optional
    :return: Second derivative of B-spline basis
        Shape: (number of splines, number of basis, number of samples)
    :rtype: torch.Tensor
    """
    if extend:
        grid = extend_grid(grid, k_extend=k)
    x, grid = _unsqueeze_inputs(x, grid)

    if k <= 1:
        return 0

    B_km2 = _compute_B_km1(x, grid, k - 1)  # Actually calls k-2 recursion

    denom1 = (grid[:, k:-1] - grid[:, : -(k + 1)]) * (
        grid[:, k:-1] - grid[:, : -(k + 1)]
    )
    denom2 = (grid[:, k + 1 :] - grid[:, 1:-k]) * (grid[:, k + 1 :] - grid[:, 1:-k])

    value = k * (k - 1) / denom1 * B_km2[:, :-1] - k * (k - 1) / denom2 * B_km2[:, 1:]

    return (value, grid) if return_extended else value


def der_B_spline(
    x: Tensor,
    grid: Tensor,
    k: int = 0,
    extend: bool = True,
    return_extended: bool = False,
):

    if extend:
        grid = extend_grid(grid, k_extend=k)
    x, grid = _unsqueeze_inputs(x, grid)

    if k == 0:
        return 0
    B_km1 = _compute_B_km1(x, grid, k)
    value = k / (grid[:, k + 1 :] - grid[:, 1:-k]) * B_km1[:, 1:]
    return (value, grid) if return_extended else value


# ========== Curve and Coefficient Conversion ==========


def coef2curve(
    x_eval: Tensor,
    grid: Tensor,
    coef: Tensor,
    k: int,
    extend: bool = True,
    repeated: bool = False,
) -> Tensor:
    """Convert B-spline coefficients to B-spline curves.

    Evaluates x on B-spline curves by summing up B_batch results over B-spline basis.

    :param x_eval: Input evaluation points
    :type x_eval: torch.Tensor
        Shape: (number_of_splines, number_of_samples)
    :param grid: Grid points for spline evaluation
    :type grid: torch.Tensor
        Shape: (number_of_splines, number_of_grid_points)
    :param coef: B-spline coefficients
    :type coef: torch.Tensor
        Shape: (number_of_splines, number_of_coefficients)
        Note: number_of_coefficients = number_of_grid_intervals + k
    :param k: Order of the B-spline
    :type k: int
    :param extend: Whether to extend grid points, defaults to True
    :type extend: bool, optional
    :param repeated: Whether to use repeated boundary points, defaults to False
    :type repeated: bool, optional
    :return: Evaluated B-spline curves
    :rtype: torch.Tensor
        Shape: (number_of_splines, number_of_samples)

    Example:
    --------

    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    """
    coef = coef.to(x_eval.dtype)
    assert (
        x_eval.device == grid.device == coef.device
    ), f"x_eval device: {x_eval.device} != grid device: {grid.device} != coef device: {coef.device}"
    basis_fn = B_batch_mod if repeated else B_batch
    B = basis_fn(x_eval, grid, k=k, extend=extend)
    return torch.einsum("ij,ijk->ik", coef, B)


def der_coef2curve(x_eval: Tensor, grid: Tensor, coef: Tensor, k: int):
    coef = coef.to(x_eval.dtype)
    return torch.einsum("ij,ijk->ik", coef, der_B_batch(x_eval, grid, k))


def der_coef2curve_coef(x_eval: Tensor, grid: Tensor, coef: Tensor, k: int):
    coef = coef[:, 1:] - coef[:, :-1]
    grid = grid[:, :-1]
    return torch.einsum("ij,ijk->ik", coef, der_B_spline(x_eval, grid, k))


def curve2coef(x_eval: Tensor, y_eval: Tensor, grid: Tensor, k: int) -> Tensor:
    """Convert B-spline curves to coefficients using least squares.

    :param x_eval: Evaluation points
    :type x_eval: torch.Tensor
        Shape: (number_of_splines, number_of_samples)
    :param y_eval: Curve values at evaluation points
    :type y_eval: torch.Tensor
        Shape: (number_of_splines, number_of_samples)
    :param grid: Grid points for spline evaluation
    :type grid: torch.Tensor
        Shape: (number_of_splines, number_of_grid_points)
    :param k: Order of the B-spline
    :type k: int
    :return: Computed B-spline coefficients
    :rtype: torch.Tensor
        Shape: (number_of_splines, number_of_coefficients)
        Where number_of_coefficients = number_of_grid_points + k - 1

    Example:
    --------

    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> curve2coef(x_eval, y_eval, grids, k).shape
    torch.Size([5, 13])
    """
    device = x_eval.device
    b_batch = B_batch(x_eval, grid, k)
    assert isinstance(b_batch, Tensor), "b_batch must be a Tensor"
    mat = b_batch.permute(0, 2, 1)
    y = y_eval.unsqueeze(dim=2)
    solution = torch.linalg.lstsq(mat, y, driver="gelsy" if device == "cpu" else "gels")
    return solution.solution[:, :, 0].to(device)


# ========== Test ==========


def _test_all():
    torch.manual_seed(42)
    device = "cuda"

    # 1. Generate random data
    data_range = (-3, 3)
    n, m, g, k = 4, 50, 8, 2  # num_splines, num_samples, num_grid_interval, k
    x_data = torch.linspace(data_range[0], data_range[1], m).repeat(n, 1).to(device=device)
    
    y_true = 2 * torch.sin(x_data) + 0.5 * x_data**2 - 0.1 * x_data**3 + torch.sqrt(x_data.abs())
    y_true += 0.05 * torch.randn_like(y_true)  # Add noise
    
    # 2. Extend grid points
    grid = torch.einsum(
        "i,j->ij", torch.ones(n), torch.linspace(data_range[0], data_range[1], g + 1)
    ).to(device)
    extended_grid = extend_grid(grid, k_extend=k)
    
    # 3. Generate random B-spline coefficients
    coef = torch.randn((n, g + k)).to(device=device)  # random fits
    
    # testing
    print(">> Testing B_batch...")
    B = B_batch(x_data, grid, k)
    assert isinstance(B, Tensor), "B must be a Tensor"
    print("Shape of B-spline basis:", B.shape)

    print(">> Testing coef2curve...")
    y = coef2curve(x_data, grid, coef, k)
    print("Shape of curve output:", y.shape)

    print(">> Testing der_coef2curve...")
    dy_dx = der_coef2curve(x_data, grid, coef, k)
    print("Shape of dy/dx:", dy_dx.shape)

    print(">> Testing der_coef2curve_coef...")
    dy_dcoef = der_coef2curve_coef(x_data, grid, coef, k)
    print("Shape of dy/dcoef:", dy_dcoef.shape)

    print(">> Testing curve2coef...")
    coef_hat = curve2coef(x_data, y, grid, k)
    print("Shape of recovered coef:", coef_hat.shape)

    recon = coef2curve(x_data, grid, coef_hat, k)
    err = torch.norm(y - recon)
    print(f">> Reconstruction error: {err:.4e}")
    
    # 4. Fit B-spline coefficients with extended grid points
    fitted_coef = curve2coef(x_data, y_true, grid, k=k)
    fitted_curve = coef2curve(x_data, grid, fitted_coef, k=k)

    # 5. Visualization Results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))

    # Plot true curve
    plt.plot(x_data[0].cpu(), y_true[0].cpu(), 'b-', alpha=0.6, linewidth=2, label='True Curve')

    # Compute and plot individual Gaussian components
    basis_functions = B_batch(x_data, grid, k=k, extend=True)
    assert isinstance(basis_functions, Tensor), "basis_functions must be a Tensor"
    num_basis = basis_functions.shape[1]

    # Create color gradient for individual components
    cmap = plt.get_cmap('viridis', num_basis)

    # Plot each basis function scaled by its coefficient
    components = torch.zeros_like(fitted_curve)
    for i in range(num_basis):
        component = fitted_coef[0, i] * basis_functions[0, i]
        components[0] += component
        plt.plot(x_data[0].cpu(), component.cpu(), '--', 
                color=cmap(i/num_basis), alpha=0.6, 
                label=f'Basis {i+1}' if i < 3 else None)  # Only label first few for clarity

    # Plot the reconstructed curve
    plt.plot(x_data[0].cpu(), fitted_curve[0].cpu(), 'r-', linewidth=2, alpha=0.6,
            label='Reconstructed Curve (Sum of Basis Functions)')

    # Plot grid points
    plt.scatter(grid[0].cpu(), torch.zeros_like(grid[0].cpu()), 
                c='g', s=100, marker='|', label='Grid Points')

    plt.title(f"B-spline Fitting (k={k}, Grid Points={g+1})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _test_all()
