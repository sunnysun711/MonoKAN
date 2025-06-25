import numpy as np
import torch
import sympy
from sklearn.linear_model import LinearRegression


# ============== Symbolic library for KANLayer ==============
# Copied from KAN.utils.py

# singularity protection functions
f_inv = lambda x, y_th: (
    (x_th := 1 / y_th),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x) * (torch.abs(x) >= x_th),
)
f_inv2 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 2)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**2) * (torch.abs(x) >= x_th),
)
f_inv3 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 3)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**3) * (torch.abs(x) >= x_th),
)
f_inv4 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 4)),
    y_th * (torch.abs(x) < x_th) + torch.nan_to_num(1 / x**4) * (torch.abs(x) >= x_th),
)
f_inv5 = lambda x, y_th: (
    (x_th := 1 / y_th ** (1 / 5)),
    y_th / x_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / x**5) * (torch.abs(x) >= x_th),
)
f_sqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    x_th / y_th * x * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.sqrt(torch.abs(x)) * torch.sign(x))
    * (torch.abs(x) >= x_th),
)
f_power1d5 = lambda x, y_th: torch.abs(x) ** 1.5
f_invsqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(1 / torch.sqrt(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_log = lambda x, y_th: (
    (x_th := torch.e ** (-y_th)),
    -y_th * (torch.abs(x) < x_th)
    + torch.nan_to_num(torch.log(torch.abs(x))) * (torch.abs(x) >= x_th),
)
f_tan = lambda x, y_th: (
    (clip := x % torch.pi),
    (delta := torch.pi / 2 - torch.arctan(y_th)),
    -y_th / delta * (clip - torch.pi / 2) * (torch.abs(clip - torch.pi / 2) < delta)
    + torch.nan_to_num(torch.tan(clip)) * (torch.abs(clip - torch.pi / 2) >= delta),
)
f_arctanh = lambda x, y_th: (
    (delta := 1 - torch.tanh(y_th) + 1e-4),
    y_th * torch.sign(x) * (torch.abs(x) > 1 - delta)
    + torch.nan_to_num(torch.arctanh(x)) * (torch.abs(x) <= 1 - delta),
)
f_arcsin = lambda x, y_th: (
    (),
    torch.pi / 2 * torch.sign(x) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arcsin(x)) * (torch.abs(x) <= 1),
)
f_arccos = lambda x, y_th: (
    (),
    torch.pi / 2 * (1 - torch.sign(x)) * (torch.abs(x) > 1)
    + torch.nan_to_num(torch.arccos(x)) * (torch.abs(x) <= 1),
)
f_exp = lambda x, y_th: (
    (x_th := torch.log(y_th)),
    y_th * (x > x_th) + torch.exp(x) * (x <= x_th),
)

SYMBOLIC_LIB = {
    "x": (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
    "x^2": (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
    "x^3": (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
    "x^4": (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
    "x^5": (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
    "1/x": (lambda x: 1 / x, lambda x: 1 / x, 2, f_inv),
    "1/x^2": (lambda x: 1 / x**2, lambda x: 1 / x**2, 2, f_inv2),
    "1/x^3": (lambda x: 1 / x**3, lambda x: 1 / x**3, 3, f_inv3),
    "1/x^4": (lambda x: 1 / x**4, lambda x: 1 / x**4, 4, f_inv4),
    "1/x^5": (lambda x: 1 / x**5, lambda x: 1 / x**5, 5, f_inv5),
    "sqrt": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    "x^0.5": (lambda x: torch.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    "x^1.5": (
        lambda x: torch.sqrt(x) ** 3,
        lambda x: sympy.sqrt(x) ** 3,
        4,
        f_power1d5,
    ),
    "1/sqrt(x)": (
        lambda x: 1 / torch.sqrt(x),
        lambda x: sympy.Pow(x, -0.5),
        2,
        f_invsqrt,
    ),
    "1/x^0.5": (
        lambda x: 1 / torch.sqrt(x),
        lambda x: sympy.Pow(x, -0.5),
        2,
        f_invsqrt,
    ),
    "exp": (lambda x: torch.exp(x), lambda x: sympy.exp(x), 2, f_exp),
    "log": (lambda x: torch.log(x), lambda x: sympy.log(x), 2, f_log),
    "abs": (
        lambda x: torch.abs(x),
        lambda x: sympy.Abs(x),
        3,
        lambda x, y_th: ((), torch.abs(x)),
    ),
    "sin": (
        lambda x: torch.sin(x),
        lambda x: sympy.sin(x),
        2,
        lambda x, y_th: ((), torch.sin(x)),
    ),
    "cos": (
        lambda x: torch.cos(x),
        lambda x: sympy.cos(x),
        2,
        lambda x, y_th: ((), torch.cos(x)),
    ),
    "tan": (lambda x: torch.tan(x), lambda x: sympy.tan(x), 3, f_tan),
    "tanh": (
        lambda x: torch.tanh(x),
        lambda x: sympy.tanh(x),
        3,
        lambda x, y_th: ((), torch.tanh(x)),
    ),
    "sgn": (
        lambda x: torch.sign(x),
        lambda x: sympy.sign(x),
        3,
        lambda x, y_th: ((), torch.sign(x)),
    ),
    "arcsin": (lambda x: torch.arcsin(x), lambda x: sympy.asin(x), 4, f_arcsin),
    "arccos": (lambda x: torch.arccos(x), lambda x: sympy.acos(x), 4, f_arccos),
    "arctan": (
        lambda x: torch.arctan(x),
        lambda x: sympy.atan(x),
        4,
        lambda x, y_th: ((), torch.arctan(x)),
    ),
    "arctanh": (lambda x: torch.arctanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
    "0": (lambda x: x * 0, lambda x: x * 0, 0, lambda x, y_th: ((), x * 0)),
    "gaussian": (
        lambda x: torch.exp(-(x**2)),
        lambda x: sympy.exp(-(x**2)),
        3,
        lambda x, y_th: ((), torch.exp(-(x**2))),
    ),
    #'cosh': (lambda x: torch.cosh(x), lambda x: sympy.cosh(x), 5),
    #'sigmoid': (lambda x: torch.sigmoid(x), sympy.Function('sigmoid'), 4),
    #'relu': (lambda x: torch.relu(x), relu),
}


def fit_params(
    x,
    y,
    fun,
    a_range=(-10, 10),
    b_range=(-10, 10),
    grid_number=101,
    iteration=3,
    verbose=True,
    device="cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit parameters a, b, c, d such that the function

        y ≈ c * fun(a * x + b) + d

    is best approximated in the least-squares sense. This is achieved by
    grid search over (a, b), followed by linear regression to estimate (c, d).

    This function minimizes the squared error:

        |y - (c * fun(a * x + b) + d)|^2

    :param x: 1D array
    :type x: torch.Tensor
    :param y: 1D array
    :type y: torch.Tensor
    :param fun: symbolic function to fit
    :type fun: sympy.Function
    :param a_range: tuple, range for parameter a
    :type a_range: tuple, optional
    :param b_range: tuple, range for parameter b
    :type b_range: tuple, optional
    :param grid_number: int, number of steps along a and b
    :type grid_number: int, optional
    :param iteration: int, number of zooming in
    :type iteration: int, optional
    :param verbose: bool, print extra information if True
    :type verbose: bool, optional
    :param device: str, device to run the model
    :type device: str, optional
    
    :return: (a_best, b_best, c_best, d_best), r2_best
        Tuple[torch.Tensor, torch.Tensor]:
            - params (torch.Tensor): Tensor of shape (4,) containing best-fit parameters [a, b, c, d].
            - r2 (torch.Tensor): Scalar tensor representing the coefficient of determination R².
    :rtype: tuple[torch.Tensor, torch.Tensor]

    Example
    -------
    >>> num = 100
    >>> x = torch.linspace(-1,1,steps=num)
    >>> noises = torch.normal(0,1,(num,)) * 0.02
    >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
    >>> fit_params(x, y, torch.sin)
    r2 is 0.9999727010726929
    (tensor([2.9982, 1.9996, 5.0053, 0.7011]), tensor(1.0000))
    """
    # fit a, b, c, d such that y=c*fun(a*x+b)+d; both x and y are 1D array.
    # sweep a and b, choose the best fitted model
    assert iteration > 0, "Iteration must be greater than 0."
    for _ in range(iteration):
        a_ = torch.linspace(a_range[0], a_range[1], steps=grid_number, device=device)
        b_ = torch.linspace(b_range[0], b_range[1], steps=grid_number, device=device)
        a_grid, b_grid = torch.meshgrid(a_, b_, indexing="ij")
        post_fun = fun(a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :])
        x_mean = torch.mean(post_fun, dim=[0], keepdim=True)
        y_mean = torch.mean(y, dim=[0], keepdim=True)
        numerator = (
            torch.sum((post_fun - x_mean) * (y - y_mean)[:, None, None], dim=0) ** 2
        )
        denominator = torch.sum((post_fun - x_mean) ** 2, dim=0) * torch.sum(
            (y - y_mean)[:, None, None] ** 2, dim=0
        )
        r2 = numerator / (denominator + 1e-4)
        r2 = torch.nan_to_num(r2)

        best_id = torch.argmax(r2)
        a_id, b_id = (
            torch.div(best_id, grid_number, rounding_mode="floor"),
            best_id % grid_number,
        )

        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose == True:
                print("Best value at boundary.")
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]

        else:
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]

    a_best = a_[a_id]  # type: ignore
    b_best = b_[b_id]  # type: ignore
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]  # type: ignore

    if verbose == True:
        print(f"r2 is {r2_best}")
        if r2_best < 0.9:
            print(
                f"r2 is not very high, please double check if you are choosing the correct symbolic function."
            )

    post_fun = torch.nan_to_num(post_fun)
    reg = LinearRegression().fit(
        post_fun[:, None].detach().cpu().numpy(), y.detach().cpu().numpy()
    )
    c_best = torch.from_numpy(reg.coef_)[0].to(device)
    d_best = torch.from_numpy(np.array(reg.intercept_)).to(device)
    return torch.stack([a_best, b_best, c_best, d_best]), r2_best


def sparse_mask(in_dim:int, out_dim:int) -> torch.Tensor:
    """
    Get sparse mask
    
    :param in_dim: input dimension
    :type in_dim: int
    :param out_dim: output dimension
    :type out_dim: int
    
    :return: sparse mask, shape (in_dim, out_dim)
    :rtype: torch.Tensor
    
    Example
    -------
    >>> from src.utils import sparse_mask
    >>> sparse_mask(5, 3)
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.]])
    >>> sparse_mask(4, 3)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    
    """
    in_coord = (torch.arange(in_dim) + 0.5) / in_dim
    out_coord = (torch.arange(out_dim) + 0.5) / out_dim

    in_to_out = torch.argmin(torch.abs(out_coord[None, :] - in_coord[:, None]), dim=1)
    out_to_in = torch.argmin(torch.abs(in_coord[None, :] - out_coord[:, None]), dim=1)

    mask = torch.zeros(in_dim, out_dim)
    mask[torch.arange(in_dim), in_to_out] = 1.0
    mask[out_to_in, torch.arange(out_dim)] = 1.0

    return mask
