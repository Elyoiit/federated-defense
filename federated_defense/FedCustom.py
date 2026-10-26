from flwr.common import Context, NDArrays, Parameters, Scalar, ndarrays_to_parameters, FitIns
from flwr.server import ClientManager, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch import List, Tuple
from torchvision import transforms
from federated_defense.task import apply_pixel_pattern, get_weights, set_weights, get_net
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from logging import INFO, DEBUG
from flwr.common import log, logger
from flwr.server.client_proxy import ClientProxy


class FedCustom(FedAvg):
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {}
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        log(INFO, "Clients: %s have been selected for training", clients)

        return [(client, fit_ins) for client in clients] 