##
import ray
import numpy as np
import torch
from dataclasses import dataclass
import time


# Dataclass for Model Configuration
@dataclass
class ModelConfig:
    a: float = 1.0
    b: float = 0.0


# Simple PyTorch Model
class SimpleModel(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super(SimpleModel, self).__init__()
        self.a = config.a
        self.b = config.b

    def forward(self, x):
        return self.a * x + self.b


# Base Actor Class
class BaseActor:
    def __init__(self):
        self.downstream_actors = []

    def set_downstream(self, *downstream_actors):
        self.downstream_actors.extend(downstream_actors)

    def process(self, data):
        result = self.forward(data)
        if result is not None:
            for downstream_actor in self.downstream_actors:
                downstream_actor.process.remote(result)

    def forward(self, data):
        raise NotImplementedError("Subclasses should implement this method")


# Example subclass for data generation
@ray.remote
class DataGenerator(BaseActor):
    def forward(self, _):
        time.sleep(0.01)  # Simulating data generation delay
        return np.random.rand()

    def start_producing(self):
        while True:
            self.process(None)

# Example subclass for data accumulation
@ray.remote
class DataAccumulator(BaseActor):
    def __init__(self):
        super().__init__()
        self.buffer = []

    def forward(self, data):
        self.buffer.append(data)
        if len(self.buffer) >= 100:  # Accumulate 100 data points
            batch = np.array(self.buffer)
            self.buffer = []
            return batch


# Example subclass for data processing with PyTorch model
@ray.remote
class DataProcessor(BaseActor):
    def __init__(self, model_cfg):
        super().__init__()
        self.model = SimpleModel(model_cfg)

    def forward(self, data):
        tensor_data = torch.tensor(data, dtype=torch.float32)
        return self.model(tensor_data).detach().numpy()


# Example subclass for printing data
@ray.remote
class DataPrinter(BaseActor):
    def forward(self, data):
        print("Data avg:", data.mean())


# Main Execution Logic
def main():
    ray.init()

    # Instantiate actors
    generator = DataGenerator.remote()
    accumulator = DataAccumulator.remote()
    processor = DataProcessor.remote(ModelConfig(a=1.0, b=0.0))
    printer = DataPrinter.remote()

    # Set up the DAG
    ray.get(generator.set_downstream.remote(accumulator))
    ray.get(accumulator.set_downstream.remote(processor))
    ray.get(processor.set_downstream.remote(printer))

    # Start continuous data generation
    generator.start_producing.remote()


if __name__ == "__main__":
    main()

##
