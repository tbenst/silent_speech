import typer
import numpy as np
import torch

app = typer.Typer()

def complex_function(x: float):
    very_large_array = torch.from_numpy(np.random.rand(1000000).reshape(1000, 1000))
    result = x / 0  # This will cause a division by zero error
    return result

@app.command()
def main(value: float = typer.Option(0.0, help="A value to be processed")):
    return complex_function(value)

if __name__ == "__main__":
    app()