import torch


def B_batch(x: torch.Tensor, grid: torch.Tensor, k:int=0) -> torch.Tensor:
    """
    Evaludate x on B-spline bases
    
    :param torch.Tensor x: inputs, shape (number of samples, number of splines)
    :param torch.Tensor grid: grids, shape (number of splines, number of grid points)
    :param int k: the piecewise polynomial order of splines, defaults to 0
    
    :return torch.Tensor: spline values, shape (batch, in_dim, G-k).
        G: the number of grid intervals, k: spline order.

    Example::
    
        >>> from src.spline import B_batch, torch
        >>> m=100; n=5; k=3; G=20; out_dim=2
        >>> x = torch.rand(m, n)
        >>> grid = torch.linspace(-1,1,steps=G+1).repeat(n, 1)  # n, G+1
        >>> B_batch(x, grid, k=3).shape  # m, n, G-k
        torch.Size([100, 5, 17])
    """
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value



def coef2curve(x_eval:torch.Tensor, extended_grid:torch.Tensor, coef:torch.Tensor, k:int=0) -> torch.Tensor:
    """
    Converting B-spline coefficients to B-spline curves. 
    Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    :param torch.Tensor x_eval: 2D torch.tensor
        shape (batch, in_dim)
    :param torch.Tensor extended_grid: 2D torch.tensor
        shape (in_dim, G+2k+1). G: num grid intervals; k: spline order.
    :param torch.Tensor coef: 3D torch.tensor
        shape (in_dim, out_dim, G+k)
    :param int k: the piecewise polynomial order of splines, defaults to 0
    
    :return: y_eval, shape (batch, in_dim, out_dim)
    :rtype: torch.Tensor
    
    Example::
    
        >>> from src.spline import torch, coef2curve, extend_grid
        >>> m=100; n=5; k=3; G=10; out_dim=2
        >>> x_eval = torch.rand(m, n)
        >>> grid = torch.linspace(-1, 1, steps=G+1).repeat(n, 1)
        >>> extended_grid = extend_grid(grid, k_extend=k)  # (n, G+2k+1)
        >>> coef = torch.rand(n, out_dim, G+k)
        >>> y_eval = coef2curve(x_eval, extended_grid, coef, k=k)
        >>> y_eval.shape  # batch, in_dim, out_dim
        torch.Size([100, 5, 2])
    """
    
    b_splines = B_batch(x_eval, extended_grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval


def curve2coef(x_eval:torch.Tensor, y_eval:torch.Tensor, extended_grid:torch.Tensor, k:int=0) -> torch.Tensor:
    """
    Converting B-spline curves to B-spline coefficients using least squares.
    
    :param torch.Tensor x_eval: 2D torch.tensor
        shape (batch, in_dim)
    :param torch.Tensor y_eval: 3D torch.tensor
        shape (batch, in_dim, out_dim)
    :param torch.Tensor extended_grid: 2D torch.tensor
        shape (in_dim, G+2k+1)
    :param int k: the piecewise polynomial order of splines, defaults to 0
    
    :raise RuntimeError: if least squares fails
    
    :return torch.Tensor: coef, shape (in_dim, out_dim, G+k)
    
    Example::
    
        >>> from src.spline import torch, curve2coef, extend_grid
        >>> m=100; n=5; k=3; G=10; out_dim=2
        >>> x_eval = torch.rand(m, n)
        >>> y_eval = torch.rand(m, n, out_dim)
        >>> grid = torch.linspace(-1, 1, steps=G+1).repeat(n, 1)
        >>> extended_grid = extend_grid(grid, k_extend=k)  # (n, G+2k+1)
        >>> coef = curve2coef(x_eval, y_eval, extended_grid, k=k)
        >>> coef.shape  # in_dim, out_dim, G+k
        torch.Size([5, 2, 13])
    """
    #print('haha', x_eval.shape, y_eval.shape, grid.shape)
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = extended_grid.shape[1] - k - 1
    mat = B_batch(x_eval, extended_grid, k)
    mat = mat.permute(1,0,2)[:,None,:,:].expand(in_dim, out_dim, batch, n_coef)
    #print('mat', mat.shape)
    y_eval = y_eval.permute(1,2,0).unsqueeze(dim=3)
    #print('y_eval', y_eval.shape)
    device = mat.device
    
    #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,:,0]
    coef = None
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
    except:
        print('lstsq failed')
        raise RuntimeError('Least squares failed, please check the input data.')
    
    # manual psuedo-inverse
    """lamb=1e-8
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0,1,3,2), y_eval)
    n1, n2, n = XtX.shape[0], XtX.shape[1], XtX.shape[2]
    identity = torch.eye(n,n)[None, None, :, :].expand(n1, n2, n, n).to(device)
    A = XtX + lamb * identity
    B = Xty
    coef = (A.pinverse() @ B)[:,:,:,0]"""
    
    return coef


def extend_grid(grid:torch.Tensor, k_extend:int=0) -> torch.Tensor:
    """
    Extend grid by k points on both ends
    
    :param torch.Tensor grid: 2D torch.tensor, shape (in_dim, G+1), where G is the number of grid intervals
    :param int k_extend: number of points to extend on both ends, defaults to 0
    
    :return torch.Tensor: extended grid, shape (in_dim, G+2k+1)
    
    Example::
    
        >>> from src.spline import extend_grid, torch
        >>> G=10; k=3; in_dim=2;  # number of grid intervals, spline order, input dimension (number of splines)
        >>> grid = torch.linspace(-1, 1, steps=G+1).repeat(in_dim, 1)
        >>> extended_grid = extend_grid(grid, k_extend=k)
        >>> extended_grid.shape  # in_dim, G+2k+1
        torch.Size([2, 17])
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid