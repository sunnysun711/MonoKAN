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
        # noise = noise_std * torch.randn(batch, 3)
        noise = torch.zeros(batch, 3)
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
    kan = KAN(width, mult_arity=2, grid=10, device="cuda", sparse_init=False, auto_save=False)

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
    
    # kan.plot(
    #     in_vars=[
    #         "train_tt",
    #         "train_co", 
    #         "train_he",
    #         "SM_tt",
    #         "SM_co",
    #         "SM_he",
    #         "SM_seats",
    #         "car_TT",
    #         "car_CO",
    #     ],
    #     out_vars=["train", "SM", "car"],
    #     title="SwissMetro DCM KAN (trained)",
    #     varscale=1,
    #     scale=0.7,
    #     # beta=30,
    #     sample=False,
    # )
    # plt.show()
    
def test_dnn():
    
    class SimpleDNN(torch.nn.Module):
        def __init__(self, input_dim, output_dim, hidden_layers=[64, 32], device="cpu"):
            super(SimpleDNN, self).__init__()
            self.device = device
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(0.1))
                prev_dim = hidden_dim
            layers.append(torch.nn.Linear(prev_dim, output_dim))
            self.network = torch.nn.Sequential(*layers)
            self.to(device)
        
        def forward(self, x):
            return self.network(x)
    
    hidden_layers = [128, 64, 32]  # Adjust to match KAN parameter count
    
    model = SimpleDNN(
        input_dim=9, 
        output_dim=3, 
        hidden_layers=hidden_layers,
        device="cuda"
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DNN Total parameters: {total_params}")
    
    dataset = create_dcm_swissmetro_dataset(
        train_num=1000, test_num=100, device="cuda"
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = dcm_loss_fn
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        utilities = model(dataset["train_input"])
        loss = criterion(utilities, dataset["train_label"])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        train_utilities = model(dataset["train_input"])
        train_probs = F.softmax(train_utilities, dim=1)
        
        test_utilities = model(dataset["test_input"])
        test_probs = F.softmax(test_utilities, dim=1)
        
        # Calculate accuracy
        train_pred = torch.argmax(train_probs, dim=1)
        train_true = torch.argmax(dataset["train_label"], dim=1)
        train_accuracy = (train_pred == train_true).float().mean()
        
        test_pred = torch.argmax(test_probs, dim=1)
        test_true = torch.argmax(dataset["test_label"], dim=1)
        test_accuracy = (test_pred == test_true).float().mean()
        
        print(f"DNN Train Accuracy: {train_accuracy:.4f}")
        print(f"DNN Test Accuracy: {test_accuracy:.4f}")


def test_dnn_dcm():
    class DCM_DNN(torch.nn.Module):
        def __init__(self, alternative_configs, shared_layers=[32, 16], device="cpu"):
            super(DCM_DNN, self).__init__()
            self.device = device
            self.alternative_configs = alternative_configs
            self.n_alternatives = len(alternative_configs)
            
            self.alt_networks = torch.nn.ModuleList()
            
            for input_dim, alt_name in alternative_configs:
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in shared_layers:
                    layers.append(torch.nn.Linear(prev_dim, hidden_dim))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout(0.1))
                    prev_dim = hidden_dim
                
                # Output single utility for this alternative
                layers.append(torch.nn.Linear(prev_dim, 1))
                
                alt_network = torch.nn.Sequential(*layers)
                self.alt_networks.append(alt_network)
            
            self.to(device)
        
        def forward(self, x):
            utilities = []
            # train: x[:, 0:3], SM: x[:, 3:7], car: x[:, 7:9]
            alt_inputs = [
                x[:, 0:3],   # train features
                x[:, 3:7],   # SM features  
                x[:, 7:9]    # car features
            ]
            for i, alt_input in enumerate(alt_inputs):
                utility = self.alt_networks[i](alt_input)  # [batch, 1]
                utilities.append(utility)
            return torch.cat(utilities, dim=1)  # [batch, 3]
    # Alternative-specific configurations
    alternative_configs = [
        (3, "train"),  # train_tt, train_co, train_he
        (4, "SM"),     # SM_tt, SM_co, SM_he, SM_seats
        (2, "car")     # car_TT, car_CO
    ]
    
    shared_layers = [32, 16]  # Smaller networks since inputs are separated
    
    model = DCM_DNN(
        alternative_configs=alternative_configs,
        shared_layers=shared_layers,
        device="cuda"
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DCM-DNN Total parameters: {total_params}")
    
    dataset = create_dcm_swissmetro_dataset(
        train_num=1000, test_num=100, device="cuda"
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = dcm_loss_fn
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        
        utilities = model(dataset["train_input"])
        loss = criterion(utilities, dataset["train_label"])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        train_utilities = model(dataset["train_input"])
        train_probs = F.softmax(train_utilities, dim=1)
        
        test_utilities = model(dataset["test_input"])
        test_probs = F.softmax(test_utilities, dim=1)
        
        # Calculate accuracy
        train_pred = torch.argmax(train_probs, dim=1)
        train_true = torch.argmax(dataset["train_label"], dim=1)
        train_accuracy = (train_pred == train_true).float().mean()
        
        test_pred = torch.argmax(test_probs, dim=1)
        test_true = torch.argmax(dataset["test_label"], dim=1)
        test_accuracy = (test_pred == test_true).float().mean()
        
        print(f"DCM-DNN Train Accuracy: {train_accuracy:.4f}")
        print(f"DCM-DNN Test Accuracy: {test_accuracy:.4f}")
    pass

if __name__ == "__main__":
    print("=== Testing KAN ===")
    test_multikan()
    
    print("\n=== Testing Standard DNN ===")
    test_dnn()
    
    print("\n=== Testing DCM-specific DNN ===")
    test_dnn_dcm()