"""federated-defense: A Flower / PyTorch app."""

from flwr.common import Context, NDArrays, Scalar, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAdam
from federated_defense.task import Net, get_weights, get_net

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from federated_defense.task import Net, set_weights

def get_evaluate_fn(model):
    """Return a server-side evaluation function for PyTorch."""

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=64)

    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_weights(model, parameters) 
        model.eval()

        loss, correct, total = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return loss / len(val_loader), {"accuracy": accuracy}

    return evaluate


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(get_net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAdam(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_fn=get_evaluate_fn(get_net()),
        eta=1e-2,
        eta_l=0.01, 
        tau=1e-3,
        )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
