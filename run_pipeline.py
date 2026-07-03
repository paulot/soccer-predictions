import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
import sys
from ml_model.train import train_models
from ml_model.backtest import run_ml_backtest

def main():
    parser = argparse.ArgumentParser(description="Unified CLI for MCMC Soccer Prediction Pipeline")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: train
    train_parser = subparsers.add_parser("train", help="Train transition and outcome models")
    train_parser.add_argument("--model", type=str, default="random_forest",
                              choices=["logistic_regression", "random_forest", "xgboost", "neural_network"],
                              help="Classifier architecture to train")
    train_parser.add_argument("--mode", type=str, default="iteration",
                              choices=["iteration", "production"],
                              help="Iteration (fast, 50 matches) vs Production (slow, all matches)")
                              
    # Subcommand: backtest
    backtest_parser = subparsers.add_parser("backtest", help="Evaluate models using MCMC LOOCV")
    backtest_parser.add_argument("--model", type=str, default="random_forest",
                                 choices=["heuristic", "logistic_regression", "random_forest", "xgboost", "neural_network"],
                                 help="Model to evaluate")
    backtest_parser.add_argument("--mode", type=str, default="iteration",
                                 choices=["iteration", "production"],
                                 help="Iteration (fast, 2 matches, 100 sims) vs Production (slow, 5 matches, 500 sims)")
    backtest_parser.add_argument("--sims", type=int, default=500,
                                 help="Number of MCMC simulations per match (maxed in iteration mode)")
                                 
    args = parser.parse_args()
    print("DEBUG: parsed args =", args)
    
    if args.command == "train":
        train_models(args.model, args.mode)
        print("DEBUG: train_models finished", flush=True)
    elif args.command == "backtest":
        run_ml_backtest(args.model, args.mode, args.sims)

if __name__ == "__main__":
    main()
