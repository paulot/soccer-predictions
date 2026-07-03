import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import click
from ml_model.train import train_models
from ml_model.backtest import run_ml_backtest

@click.group(help="Unified CLI for MCMC Soccer Prediction Pipeline")
def cli() -> None:
    pass

@cli.command(help="Train transition and outcome models")
@click.option("--model", type=click.Choice(["logistic_regression", "random_forest", "xgboost", "neural_network"]), default="random_forest", help="Classifier architecture to train")
@click.option("--mode", type=click.Choice(["iteration", "production"]), default="iteration", help="Iteration vs Production mode")
def train(model: str, mode: str) -> None:
    train_models(model, mode)
    print("DEBUG: train_models finished", flush=True)

@cli.command(help="Evaluate models using MCMC LOOCV")
@click.option("--model", type=click.Choice(["heuristic", "logistic_regression", "random_forest", "xgboost", "neural_network"]), default="random_forest", help="Model to evaluate")
@click.option("--mode", type=click.Choice(["iteration", "production"]), default="iteration", help="Iteration vs Production mode")
@click.option("--sims", type=int, default=500, help="Number of MCMC simulations per match")
def backtest(model: str, mode: str, sims: int) -> None:
    run_ml_backtest(model, mode, sims)

if __name__ == "__main__":
    cli()
