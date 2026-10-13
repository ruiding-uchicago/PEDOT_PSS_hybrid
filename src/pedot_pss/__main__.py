"""Command-line JSON input/output for the conductivity black box."""
import argparse
import json
from pathlib import Path
import sys
from . import predict, predict_many, setup_encoder, setup_model


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "setup-model":
        parser = argparse.ArgumentParser(description="Download the pinned complete surrogate model from its GitHub release")
        parser.add_argument("setup-model")
        parser.add_argument("--destination", type=str, default=None, help="Model file path; default is the installed package models/model.joblib")
        args = parser.parse_args()
        print(json.dumps({"model": str(setup_model(args.destination))}))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "setup-encoder":
        parser = argparse.ArgumentParser(description="Download original frozen MatBERT assets from the author S3 server")
        parser.add_argument("setup-encoder")
        parser.add_argument("--destination", type=str, default=None,
                            help="Asset directory; default is the installed package assets/matbert directory")
        args = parser.parse_args()
        print(json.dumps({"matbert_assets": str(setup_encoder(args.destination))}))
        return
    parser = argparse.ArgumentParser(description="Predict PEDOT:PSS conductivity from experimental JSON")
    parser.add_argument("input", type=Path, help="One experiment dictionary or a list of experiments")
    parser.add_argument("--model", type=Path, default=None, help="Optional complete trained model file")
    parser.add_argument("--output", type=Path, default=None, help="Write predictions as JSON")
    args = parser.parse_args()
    experiments = json.loads(args.input.read_text(encoding="utf-8"))
    result = (predict(experiments, model_path=args.model) if isinstance(experiments, dict)
              else predict_many(experiments, model_path=args.model))
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
