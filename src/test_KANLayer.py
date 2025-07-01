
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Assuming KANLayer is imported from your module
# from src.KANLayer_temp import KANLayer
from src.KANLayer import KANLayer

def true_func(x):
    """Ground truth function with complex patterns."""
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    # y0: x0 ↑, x2 ↓ (with high-frequency oscillations)
    y0 = 1.5 * x0 + torch.sin(2 * torch.pi * x1) - x2**3 + 0.5 * torch.sin(4 * x2)
    # y1: x0 ↑, x2 ↑ (with high-frequency oscillations, opposite direction to y0)
    y1 = 2.5 * x0 + torch.exp(x2) + 0.5 * torch.cos(4 * x2)
    res = torch.stack([y0, y1], dim=1)
    res += torch.randn_like(res) * 0.05 - 0.025
    return res

def true_func_derivatives(x):
    """Calculate true derivatives of the ground truth function."""
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    
    # dy0/dx derivatives
    dy0_dx0 = torch.ones_like(x0) * 1.5  # constant
    dy0_dx1 = 2 * torch.pi * torch.cos(2 * torch.pi * x1)  # oscillating
    dy0_dx2 = -3 * x2**2 + 0.5 * 4 * torch.cos(4 * x2)  # decreasing trend + oscillation
    
    # dy1/dx derivatives  
    dy1_dx0 = torch.ones_like(x0) * 2.5  # constant
    dy1_dx1 = torch.zeros_like(x1)  # no x1 dependence in y1
    dy1_dx2 = torch.exp(x2) + 0.5 * (-4) * torch.sin(4 * x2)  # increasing trend + oscillation
    
    return {
        'dy0': [dy0_dx0, dy0_dx1, dy0_dx2],
        'dy1': [dy1_dx0, dy1_dx1, dy1_dx2]
    }

def train_model(constraint_type: str, monotonic_dims_dirs: list[tuple[int, int]] | None, 
                constraint_kwargs: dict | None = None, max_epochs: int = 2000, 
                patience: int = 20, verbose: bool = True) -> tuple[nn.Module, list[float]]:
    """
    Train a KANLayer model with specified constraint type.
    
    Parameters
    ----------
    constraint_type : str
        Type of monotonic constraint ('strict', 'soft', 'unconstrained').
    monotonic_dims_dirs : list
        Monotonic dimensions and directions.
    constraint_kwargs : dict, optional
        Additional constraint parameters.
    max_epochs : int, default=2000
        Maximum training epochs.
    patience : int, default=20
        Early stopping patience.
    verbose : bool, default=True
        Whether to print training progress.
        
    Returns
    -------
    tuple
        Trained model and loss history.
    """
    if constraint_kwargs is None:
        constraint_kwargs = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Generate training data
    N = 10000
    x_train = torch.rand(N, 3) * 2 - 1  # uniform(-1, 1)
    y_train = true_func(x_train)
    
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    
    G = 10
    
    # Initialize model
    if constraint_type == "unconstrained":
        model = KANLayer(
            in_dim=3, out_dim=2, num=G, k=3,
            monotonic_dims_dirs=None,
            include_basis=True,
            device=device
        )
    else:
        model = KANLayer(
            in_dim=3, out_dim=2, num=G, k=3,
            monotonic_dims_dirs=monotonic_dims_dirs,
            mono_cs_type=constraint_type,
            include_basis=True,
            device=device,
            **constraint_kwargs  # Pass constraint_kwargs here
        )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    for epoch in range(max_epochs):
        model.train()
        y_pred, *_ = model(x_train)
        loss = loss_fn(y_pred, y_train) + model.regularization_loss(lambda_l1=1e-2, lambda_entropy=2e-2, lambda_smoothness=0)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 200 == 0 or epoch == max_epochs - 1):
            print(f"  Epoch {epoch:4d}: Loss = {loss.item():.6f} (best {best_loss:.6f})")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}. No improvement in {patience} steps.")
            break
    
    return model, loss_history

def evaluate_monotonicity(model: KANLayer) -> dict:
    """Evaluate monotonicity compliance of trained model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Test monotonicity on a range of values
    x_test = torch.linspace(-1, 1, 100).to(device)
    results = {}
    
    for dim_idx, (dim, direction) in enumerate(model.monotonic_dims_dirs):
        for out_dim in range(model.out_dim):
            # Create test input varying only the target dimension
            x_probe = torch.zeros(100, 3).to(device)
            x_probe[:, dim] = x_test
            
            with torch.no_grad():
                y_pred, *_ = model(x_probe)
                y_values = y_pred[:, out_dim]
            
            # Calculate differences
            diffs = torch.diff(y_values).cpu()
            
            # Check violations
            if direction == 1:  # should be increasing
                violations = (diffs < -1e-6).float()
            else:  # should be decreasing
                violations = (diffs > 1e-6).float()
            
            violation_count = violations.sum().item()
            total_points = len(diffs)
            compliance_ratio = 1.0 - (violation_count / total_points)
            
            key = f'x{dim}_to_y{out_dim}'
            results[key] = {
                'compliance_ratio': compliance_ratio,
                'violation_count': violation_count,
                'total_points': total_points,
                'direction': 'increasing' if direction == 1 else 'decreasing'
            }
    
    return results

def plot_comparison_results(models: dict, constraint_kwargs_dict: dict):
    """Plot comparison of different constraint types."""
    
    # Generate test data
    device = next(iter(models.values())).parameters().__next__().device  # GET DEVICE FROM MODEL
    x_plot = torch.linspace(-1, 1, 1000).to(device)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Monotonic Constraints Comparison', fontsize=16)
    
    # Colors for different constraint types
    colors = {'strict': 'red', 'soft': 'blue', 'unconstrained': 'green'}
    
    for dim_idx, dim_name in enumerate(['x0', 'x1', 'x2']):
        # Prepare probe input for this dimension
        x_probe = torch.zeros(1000, 3).to(device)
        x_probe[:, dim_idx] = x_plot
        
        # Get true derivatives
        true_derivs = true_func_derivatives(x_probe)
        
        for constraint_type, model in models.items():
            with torch.no_grad():
                # Get model predictions
                y_pred, *_ = model(x_probe)
                
                # Get ground truth
                y_real = true_func(x_probe)
                
                x_plot_cpu = x_plot.cpu()
                y_pred_cpu = y_pred.cpu()
                y_real_cpu = y_real.cpu()
                
                # Plot y0 vs xi
                ax = axes[0, dim_idx]
                ax.plot(x_plot_cpu, y_pred_cpu[:, 0], color=colors[constraint_type], 
                       label=f'{constraint_type} (y0)', linewidth=2)
                if constraint_type == 'strict':  # Only plot ground truth once
                    ax.plot(x_plot_cpu, y_real_cpu[:, 0], '--', color='black', 
                           label='ground truth (y0)', alpha=0.7)
                ax.set_title(f'y0 vs {dim_name}')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Plot y1 vs xi
                ax = axes[1, dim_idx]
                ax.plot(x_plot_cpu, y_pred_cpu[:, 1], color=colors[constraint_type], 
                       label=f'{constraint_type} (y1)', linewidth=2)
                if constraint_type == 'strict':  # Only plot ground truth once
                    ax.plot(x_plot_cpu, y_real_cpu[:, 1], '--', color='black', 
                           label='ground truth (y1)', alpha=0.7)
                ax.set_title(f'y1 vs {dim_name}')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Plot differences (monotonicity check)
                ax = axes[2, dim_idx]
                # Calculate proper derivatives: dy/dx = diff(y) / diff(x)
                dx = torch.diff(x_plot_cpu)  # x step size
                diffs_y0 = torch.diff(y_pred_cpu[:, 0]) / dx
                diffs_y1 = torch.diff(y_pred_cpu[:, 1]) / dx
                
                ax.plot(x_plot_cpu[1:], diffs_y0, color=colors[constraint_type], 
                       alpha=0.7, label=f'{constraint_type} (dy0/d{dim_name})')
                ax.plot(x_plot_cpu[1:], diffs_y1, color=colors[constraint_type], 
                       linestyle=':', alpha=0.7, label=f'{constraint_type} (dy1/d{dim_name})')
                
                # Plot true derivatives (only once)
                if constraint_type == 'strict':
                    true_dy0 = true_derivs['dy0'][dim_idx].cpu()
                    true_dy1 = true_derivs['dy1'][dim_idx].cpu()
                    ax.plot(x_plot_cpu, true_dy0, '--', color='black', 
                           alpha=0.5, label=f'true dy0/d{dim_name}')
                    ax.plot(x_plot_cpu, true_dy1, ':', color='black', 
                           alpha=0.5, label=f'true dy1/d{dim_name}')
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_title(f'Derivatives d/d{dim_name}')
                ax.grid(True, alpha=0.3)
                ax.legend()
    
    plt.tight_layout()
    plt.show()

def run_monotonic_constraints_comparison():
    """Run complete comparison of monotonic constraint types."""
    
    print("=== Monotonic Constraints Comparison Test ===\n")
    
    # Define constraint configurations
    monotonic_dims_dirs = [(0, 1), (2, -1)]  # x0 ↑, x2 ↓
    
    constraint_configs = {
        'strict': {},
        'soft': {'elu_alpha': 0.002},
        'unconstrained': {}
    }
    
    # Train models with different constraint types
    models = {}
    loss_histories = {}
    
    for constraint_type, constraint_kwargs in constraint_configs.items():
        print(f"Training with {constraint_type} constraints...")
        model, loss_history = train_model(
            constraint_type=constraint_type,
            monotonic_dims_dirs=monotonic_dims_dirs,
            constraint_kwargs=constraint_kwargs,
            verbose=True,
            max_epochs=2000,
        )
        models[constraint_type] = model
        loss_histories[constraint_type] = loss_history
        print(f"Final loss: {loss_history[-1]:.6f}\n")
    
    # Evaluate monotonicity compliance
    print("=== Monotonicity Compliance Analysis ===")
    compliance_results = {}
    
    for constraint_type, model in models.items():
        compliance = evaluate_monotonicity(model)
        compliance_results[constraint_type] = compliance
        
        print(f"\n{constraint_type.upper()} Constraints:")
        for key, metrics in compliance.items():
            print(f"  {key}: {metrics['compliance_ratio']:.3f} compliance "
                  f"({metrics['violation_count']}/{metrics['total_points']} violations)")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    for constraint_type, loss_history in loss_histories.items():
        plt.plot(loss_history, label=f'{constraint_type} constraints', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    # Plot detailed comparison
    plot_comparison_results(models, constraint_configs)
    
    # Analyze coefficient differences
    print("\n=== Coefficient Analysis ===")
    for constraint_type, model in models.items():
        coef_info = model.get_eval_coefficients()  # Remove return_effective and constraint_kwargs
        
        print(f"\n{constraint_type.upper()} Constraints:")
        print(f"  Raw coef range: [{coef_info['raw_coef'].min():.3f}, {coef_info['raw_coef'].max():.3f}]")
        print(f"  Constrained coef range: [{coef_info['constrained_coef'].min():.3f}, {coef_info['constrained_coef'].max():.3f}]")
        if coef_info['effective_coef'] is not None:  # Check if not None instead of key existence
            print(f"  Effective coef range: [{coef_info['effective_coef'].min():.3f}, {coef_info['effective_coef'].max():.3f}]")
    
    return models, compliance_results, loss_histories

# Example usage for notebook
if __name__ == "__main__":
    # Run the comparison
    # torch.manual_seed(42)
    models, compliance_results, loss_histories = run_monotonic_constraints_comparison()
    
    print("\n=== Summary ===")
    print("The test compares three model types:")
    print("- STRICT: Hard monotonic constraints using cumsum(softplus())")
    print("- SOFT: Soft constraints allowing minor violations with ELU")
    print("- UNCONSTRAINED: No monotonic constraints (baseline)")
    print("\nExpected behavior:")
    print("- x0 should show increasing trend for constrained models")
    print("- x1 should show natural oscillations (no constraints for all)")
    print("- x2 should show decreasing trend for constrained models")
    print("\nStrict constraints should show best compliance but potentially worse fit.")
    print("Soft constraints should balance compliance and flexibility.")
    print("Unconstrained should show best fit but no monotonic compliance.")