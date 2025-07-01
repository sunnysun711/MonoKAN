import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from kan import KAN


def create_dcm_swissmetro_dataset(
    train_num=1000,
    test_num=100,
    ranges=[-2, 2],  # Feature range
    noise_std=0.1,  # Noise intensity
    normalize_input=False,
    normalize_label=False,
    device="cpu",
    seed=0,
):
    """
    Generate SwissMetro DCM three-alternative synthetic dataset with observed choices

    Args:
    -----
        train_num : int
            Number of training samples
        test_num : int
            Number of testing samples
        ranges : list, (2,) or (n_var, 2)
            Feature range (min,max for each feature), supports uniform/per-feature
        noise_std : float
            Utility disturbance Gaussian noise standard deviation
        normalize_input : bool
            Whether to normalize input
        normalize_label : bool
            Whether to normalize output (not applicable for choice data)
        device : str
            Device
        seed : int
            Random seed

    Returns:
    -------
        dataset : dict
            'train_input':  (train_num, 9)
            'test_input':   (test_num, 9)
            'train_label':  (train_num, 3) - one-hot encoded choices
            'test_label':   (test_num, 3) - one-hot encoded choices
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    n_var = 9

    # Uniform range or per-feature range
    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)

    def sample_input(num):
        x = torch.zeros(num, n_var)
        for i in range(n_var):
            x[:, i] = torch.rand(num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
        return x

    train_input = sample_input(train_num)
    test_input = sample_input(test_num)

    # Alias: x[0~2] = train, x[3~6] = SM, x[7~8] = car
    def utility_func(x):
        # x: [batch, 9]
        # train
        u_train = (
            -2.0 * x[:, 0]  # train_tt
            - 1.5 * x[:, 1]  # train_co
            - 0.5 * x[:, 2]  # train_he
            + 1.0 * x[:, 0] * x[:, 2]  # interaction: train_tt * train_he
        )
        # SM
        u_sm = (
            -2.2 * x[:, 3]  # SM_tt
            - 1.4 * x[:, 4]  # SM_co
            - 0.8 * x[:, 5]  # SM_he
            + 1.2 * x[:, 3] * x[:, 5]  # interaction: SM_tt * SM_he
            + 0.6 * x[:, 6]  # SM_seats
        )
        # car
        u_car = (
            -1.8 * x[:, 7]  # car_TT
            - 2.1 * x[:, 8]  # car_CO
            + 0.7 * x[:, 7] * x[:, 8]  # interaction: car_TT * car_CO
        )
        # Add Gaussian noise
        batch = x.shape[0]
        noise = noise_std * torch.randn(batch, 3)
        return torch.stack([u_train, u_sm, u_car], dim=1) + noise

    def generate_choices(x):
        """Convert utilities to observed choices using multinomial logit"""
        utilities = utility_func(x)  # [batch, 3]
        
        # Calculate choice probabilities using softmax
        probabilities = torch.softmax(utilities, dim=1)  # [batch, 3]
        
        # Sample choices from multinomial distribution
        choice_indices = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # [batch]
        
        # Convert to one-hot encoding
        choices_onehot = torch.zeros(x.shape[0], 3)
        choices_onehot.scatter_(1, choice_indices.unsqueeze(1), 1)  # [batch, 3]
        
        return choices_onehot

    train_label = generate_choices(train_input)
    test_label = generate_choices(test_input)

    def normalize(data, mean, std):
        return (data - mean) / std

    if normalize_input:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
    
    # Note: normalize_label is not applicable for discrete choice data
    if normalize_label:
        print("Warning: normalize_label=True ignored for discrete choice data")

    dataset = dict(
        train_input=train_input.to(device),
        test_input=test_input.to(device),
        train_label=train_label.to(device),
        test_label=test_label.to(device),
    )
    return dataset


# use pykan package directly
def dcm_loss_fn(y_pred, y_true):
    """
    Custom loss function for discrete choice model
    
    Args:
        y_pred: [batch, 3] - raw utilities from KAN
        y_true: [batch, 3] - one-hot encoded choices
    
    Returns:
        loss: scalar - cross entropy loss
    """
    # Convert utilities to probabilities using softmax
    probabilities = F.softmax(y_pred, dim=1)  # [batch, 3]
    
    # Calculate cross entropy loss (equivalent to negative log-likelihood)
    # Method 1: Using built-in cross entropy with one-hot labels
    loss = F.mse_loss(probabilities, y_true)
    
    # Alternative Method 2: Manual cross entropy calculation
    # loss = -torch.sum(y_true * torch.log(probabilities + 1e-8)) / y_true.shape[0]
    
    return loss

def test_multikan():
    
    width = [
        [9, 0],  # 9 input features
        [6, 3],  # 6 sum nodes, 3 mult nodes (for 3 pairwise interactions)
        [3, 0],  # 3 outputs (utility for each alternative)
    ]
    mult_arity = [
        [],  # input layer: no mult node
        [2, 2, 2],  # each mult node does two-way interaction
        [],  # output layer
    ]
    kan = KAN(width, mult_arity=2, device="cuda", sparse_init=False, auto_save=False)

    dataset = create_dcm_swissmetro_dataset(
        train_num=1000, test_num=100, device="cuda"
    )
    print(dataset["train_input"].shape, dataset["train_label"].shape)
    
    # Test forward pass
    utilities = kan(dataset["train_input"])  # Raw utilities
    probabilities = F.softmax(utilities, dim=1)  # Convert to probabilities
    print("Utilities shape:", utilities.shape)
    print("Probabilities shape:", probabilities.shape)
    print("Sample probabilities:", probabilities[:5])
    print("Sample labels:", dataset["train_label"][:5])
    
    # Fit the model with custom loss function
    kan.fit(
        dataset,
        opt="LBFGS", 
        steps=50,
        lamb=0.01,
        loss_fn=dcm_loss_fn,  # Use custom loss function
    )
    
    # Evaluate model performance
    with torch.no_grad():
        train_utilities = kan(dataset["train_input"])
        train_probs = F.softmax(train_utilities, dim=1)
        
        test_utilities = kan(dataset["test_input"]) 
        test_probs = F.softmax(test_utilities, dim=1)
        
        # Calculate accuracy
        train_pred = torch.argmax(train_probs, dim=1)
        train_true = torch.argmax(dataset["train_label"], dim=1)
        train_accuracy = (train_pred == train_true).float().mean()
        
        test_pred = torch.argmax(test_probs, dim=1)
        test_true = torch.argmax(dataset["test_label"], dim=1)
        test_accuracy = (test_pred == test_true).float().mean()
        
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    kan.plot(
        in_vars=[
            "train_tt",
            "train_co", 
            "train_he",
            "SM_tt",
            "SM_co",
            "SM_he",
            "SM_seats",
            "car_TT",
            "car_CO",
        ],
        out_vars=["train", "SM", "car"],
        title="SwissMetro DCM KAN (trained)",
        varscale=1,
        scale=0.7,
        # beta=30,
        sample=False,
    )
    plt.show()
    
    
    
test_multikan()