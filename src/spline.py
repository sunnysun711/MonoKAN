import torch


def extend_grid(grid, k_extend=0):
    # pad k to left and right
    # grid shape: (batch, grid)
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


def B_batch(
    x, grid, k=0, extend=True, return_extended=False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

    # x shape: (size, x); grid shape: (size, grid)
    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:(-k)]
        ) * B_km1[
            :, 1:
        ]
    if return_extended:
        return value, grid
    else:
        return value


def B_batch_mod(x, grid, k=0, extend=True, return_extended=False):
    """
    Modification to repeat the first k values of the grid to the left and
    the last k values of the grid to the right
    """

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid_mod(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]], grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]]], dim=1)
        return grid

    if extend == True:
        grid = extend_grid_mod(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = (x - grid[:, : -(k + 1)]) / (
            grid[:, k:-1] - grid[:, : -(k + 1)]
        ) * B_km1[:, :-1] + (grid[:, k + 1 :] - x) / (
            grid[:, k + 1 :] - grid[:, 1:(-k)]
        ) * B_km1[
            :, 1:
        ]

    ## Convert to 0 if the value is nan
    for j in range(value.shape[1]):
        if torch.all(torch.isnan(value[:, j, :])):
            value[:, j, :] = torch.where(
                torch.isnan(value[:, j, :]),
                torch.zeros_like(value[:, j, :]),
                value[:, j, :],
            )

    if return_extended:
        return value, grid
    else:
        return value


def der_B_batch(x, grid, k=0, extend=True, return_extended=False):

    # x shape: (size, x); grid shape: (size, grid)

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    ## Compute the derivative of B-spline basis functions
    if k == 0:
        return 0
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = (
            k / (grid[:, k:-1] - grid[:, : -(k + 1)]) * B_km1[:, :-1]
            - k / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * B_km1[:, 1:]
        )
    if return_extended:
        return value, grid
    else:
        return value


def der_B_batch_mod(x, grid, k=0, extend=True, return_extended=False):

    # x shape: (size, x); grid shape: (size, grid)

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    ## Compute the derivative of B-spline basis functions
    if k == 0:
        return 0
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = (
            k / (grid[:, k:-1] - grid[:, : -(k + 1)]) * B_km1[:, :-1]
            - k / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * B_km1[:, 1:]
        )
    if return_extended:
        return value, grid
    else:
        return value, B_km1


def der_B_spline(x, grid, k=0, extend=True, return_extended=False):

    # x shape: (size, x); grid shape: (size, grid)

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    if k == 0:
        return 0
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False)
        value = k / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    if return_extended:
        return value, grid
    else:
        return value


def coef2curve(x_eval, grid, coef, k, extend=True, repeated=False):
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
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
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    ## In the torch.einsum, the k index is the batch sample index
    if not repeated:
        y_eval = torch.einsum(
            "ij,ijk->ik", coef, B_batch(x_eval, grid, k, extend=extend)
        )
    else:
        y_eval = torch.einsum(
            "ij,ijk->ik", coef, B_batch_mod(x_eval, grid, k, extend=extend)
        )
    return y_eval


def der_coef2curve(x_eval, grid, coef, k):

    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    ## In the torch.einsum, the k index is the batch sample index
    y_eval = torch.einsum("ij,ijk->ik", coef, der_B_batch(x_eval, grid, k))
    return y_eval


def der_coef2curve_coef(x_eval, grid, coef, k):

    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    ## Consider coef_new = coef[i+1]-coef[i]
    coef = coef[:, 1:] - coef[:, :-1]
    ## In the torch.einsum, the k index is the batch sample index
    ## Take out the last elment of the grid
    grid = grid[:, :-1]
    y_eval = torch.einsum("ij,ijk->ik", coef, der_B_spline(x_eval, grid, k))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    """
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    torch.Size([5, 13])
    """
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k).permute(0, 2, 1)
    # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    coef = torch.linalg.lstsq(
        mat.to(device),
        y_eval.unsqueeze(dim=2).to(device),
        driver="gelsy" if device == "cpu" else "gels",
    ).solution[:, :, 0]
    return coef.to(device)


def _test_all():
    torch.manual_seed(42)

    n, m, g, k = 4, 50, 8, 3
    x_eval = torch.linspace(-1.2, 1.2, m).repeat(n, 1)
    grid = torch.linspace(-1, 1, g + 1).repeat(n, 1)
    coef = torch.randn((n, g + k))

    print(">> Testing B_batch...")
    B = B_batch(x_eval, grid, k)
    print("Shape of B-spline basis:", B.shape)

    print(">> Testing coef2curve...")
    y = coef2curve(x_eval, grid, coef, k)
    print("Shape of curve output:", y.shape)

    print(">> Testing der_coef2curve...")
    dy_dx = der_coef2curve(x_eval, grid, coef, k)
    print("Shape of dy/dx:", dy_dx.shape)

    print(">> Testing der_coef2curve_coef...")
    dy_dcoef = der_coef2curve_coef(x_eval, grid, coef, k)
    print("Shape of dy/dcoef:", dy_dcoef.shape)

    print(">> Testing curve2coef...")
    coef_hat = curve2coef(x_eval, y, grid, k)
    print("Shape of recovered coef:", coef_hat.shape)

    recon = coef2curve(x_eval, grid, coef_hat, k)
    err = torch.norm(y - recon)
    print(f">> Reconstruction error: {err:.4e}")

if __name__ =="__main__":
    _test_all()

