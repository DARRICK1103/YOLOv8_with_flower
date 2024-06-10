import os
import torch
from collections import OrderedDict
import numpy as np
from ultralytics import YOLO
import flwr as fl
from typing import List, Tuple, Dict, Union, Optional
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

HOME = os.getcwd()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

YOLO.VERBOSE = False

# Load data
def load_data(client_id: str):
    data_path = f"client{client_id}.yaml"
    return data_path

# Load model
def load_model():
    net = YOLO(f'best.pt')
    return net

def train(net, data, epochs: int):
    net.train(data=data, epochs=epochs, imgsz=640, batch=4, plots=False, device=0)

# Test function
def test(net):
    metrics = net.val(data='global.yaml')
    map50 = metrics.box.map50
    return map50

def set_parameters(net: YOLO, parameters: List[np.ndarray]) -> None:
    for i, param in enumerate(net.parameters()):
        param_ = torch.from_numpy(parameters[i]).to(param.device)
        param.data.copy_(param_)



def get_parameters(net: YOLO) -> List[np.ndarray]:
    params = [param.data.cpu().numpy() for param in net.parameters()]
    return params

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, data_path, client_id, imgsz=640):
        self.net = net
        self.data_path = data_path
        self.client_id = client_id
        self.imgsz = imgsz

    def get_parameters(self, config):
        print("I am get_parameters() and getting parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print("I passed get_parameters() and starting training")
        set_parameters(self.net, parameters)
        train(self.net, data=self.data_path, epochs=30)
        return get_parameters(self.net), 2096, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        map50 = test(self.net)
        return map50, 2096, {"map50": map50}
    
def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    net = load_model()
    data_path = load_data(int(cid))
    client_id = int(cid)
    return FlowerClient(net, data_path, client_id)

# def save_aggregated_model(parameters, round_num):
#     model = load_model()
#     state_dict = OrderedDict()
#     print("Length of parameters:", len(parameters))
#     for i, param in enumerate(model.parameters()):
#         state_dict[param.name] = parameters[i]
#     model.load_state_dict(state_dict)
#     torch.save(model, f'model/FL_aggregated_round_{round_num}.pt')




# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(self, rnd, results, failures):
#         aggregated_weights = super().aggregate_fit(rnd, results, failures)
#         if aggregated_weights is not None:
#             if isinstance(aggregated_weights, torch.Tensor):  # If aggregated_weights is PyTorch Tensor
#                 aggregated_weights = aggregated_weights.cpu().detach().numpy()  # Convert to NumPy array
#             print(f"Saving aggregated model for round {rnd}")
#             save_aggregated_model(aggregated_weights, rnd)
#         return aggregated_weights

# class SaveModelStrategy(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: List[Tuple[ClientProxy, fl.common.FitRes]],
#         failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
#     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
#         """Aggregate model weights using weighted average and store checkpoint"""
#         net = load_model()
#         # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
#         aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

#         if aggregated_parameters is not None:
#             print(f"Saving round {server_round} aggregated_parameters...")

#             # Convert `Parameters` to `List[np.ndarray]`
#             aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

#             # Convert `List[np.ndarray]` to PyTorch`state_dict`
#             params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
#             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#             net.load_state_dict(state_dict, strict=True)

#             # Save the model
#             torch.save(net.state_dict(), f"model/model_round_{server_round}.pt")

#         return aggregated_parameters, aggregated_metrics

def save_aggregated_model(parameters: List[np.ndarray], round_num: int):
    model = load_model()
    set_parameters(model, parameters)
    model.save(f'model/FL_aggregated_round_{round_num}.pt')


# def save_aggregated_model(parameters, round_num):
#     model = load_model()
#     # Convert Parameters to a list of NumPy arrays
#     parameters_list = fl.common.parameters_to_ndarrays(parameters)
#     state_dict = OrderedDict()
#     for i, param in enumerate(model.parameters()):
#         state_dict[param.name] = parameters_list[i]
#     model.load_state_dict(state_dict)
#     torch.save(model, f'model/FL_aggregated_round_{round_num}.pt')



class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
             # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            print(f"Saving aggregated model for round {rnd}")
            save_aggregated_model(aggregated_ndarrays, rnd)
        return aggregated_parameters, metrics_aggregated

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
)

client_resources = {"num_cpus": 1, "num_gpus": 0.0}
if DEVICE.type == "cuda":
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}

NUM_CLIENTS = 2

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
    client_resources=client_resources,
)
