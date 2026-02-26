"""
Train 10 models for each of the 3 priors (gaussian, MoG, flow) and evaluate
test-set ELBO for each run. Results are saved to elbo_results.json.

Usage:
    python train_eval_10x3.py [--device cuda] [--latent-dim 10] [--epochs 10] [--runs 10]
"""

import sys
import os
import json
import argparse

import torch
import torch.nn as nn
import torch.utils.data

# ---------------------------------------------------------------------------
# Allow imports from week1_simon/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "week1_simon"))

from vae_bernoulli import (
    GaussianBase,
    MaskedCouplingLayer,
    Flow,
    GaussianPrior,
    MixtureOfGaussiansPrior,
    GaussianEncoder,
    BernoulliDecoder,
    MultivariateGaussianDecoder,
    VAE,
    train,
    evaluate_test_elbo,
)

from torchvision import datasets, transforms


def build_model(prior_name: str, M: int, device: str):
    """Construct a fresh VAE with the requested prior."""
    # --- Prior ---
    if prior_name == "MoG":
        K = 10
        prior = MixtureOfGaussiansPrior(M, K)

    elif prior_name == "flow":
        base = GaussianBase(M)
        num_transformations = 16
        num_hidden = 64
        mask = torch.zeros((M,))
        mask[::2] = 1

        transformations = []
        for _ in range(num_transformations):
            mask = 1 - mask  # flip
            scale_net = nn.Sequential(
                nn.Linear(M, num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, M),
            )
            translation_net = nn.Sequential(
                nn.Linear(M, num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, num_hidden), nn.ReLU(),
                nn.Linear(num_hidden, M),
            )
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        prior = Flow(base, transformations)

    else:  # gaussian
        prior = GaussianPrior(M)

    # --- Encoder / Decoder ---
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )

    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)
    model = VAE(prior, decoder, encoder).to(device)
    return model


def get_data_loaders(batch_size: int, data_root: str = "data/"):
    threshold = 0.5
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (threshold < x).float().squeeze()),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_root, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_root, train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False,
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Train 10x3 VAE models and evaluate test ELBO")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--latent-dim", type=int, default=10, metavar="N")
    parser.add_argument("--epochs", type=int, default=10, metavar="N")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N")
    parser.add_argument("--runs", type=int, default=10, metavar="N",
                        help="number of independent runs per prior (default: 10)")
    parser.add_argument("--output", type=str, default="elbo_results.json",
                        help="path to save results (default: elbo_results.json)")
    parser.add_argument("--data-root", type=str, default="data/",
                        help="root directory for MNIST data (default: data/)")
    args = parser.parse_args()

    print("# Settings")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU.")
        device = "cpu"

    train_loader, test_loader = get_data_loaders(args.batch_size, args.data_root)

    priors = ["gaussian", "MoG", "flow"]
    results = {}  # {prior: [elbo_run0, elbo_run1, ...]}

    for prior_name in priors:
        print(f"\n{'='*60}")
        print(f"Prior: {prior_name}")
        print(f"{'='*60}")
        run_elbos = []

        for run in range(args.runs):
            print(f"\n  -- Run {run + 1}/{args.runs} --")

            model = build_model(prior_name, args.latent_dim, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train(model, optimizer, train_loader, args.epochs, device)

            elbo = evaluate_test_elbo(model, test_loader, torch.device(device))
            print(f"  Test ELBO: {elbo:.6f}")
            run_elbos.append(elbo)

            # Free GPU memory between runs
            del model, optimizer
            if device == "cuda":
                torch.cuda.empty_cache()

        import numpy as np
        mean_elbo = float(np.mean(run_elbos))
        std_elbo = float(np.std(run_elbos))
        print(f"\n  [{prior_name}] mean={mean_elbo:.4f}  std={std_elbo:.4f}")

        results[prior_name] = {
            "elbos": run_elbos,
            "mean": mean_elbo,
            "std": std_elbo,
        }

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Prior':<12}  {'Mean ELBO':>12}  {'Std ELBO':>12}")
    print("-" * 40)
    for prior_name, res in results.items():
        print(f"{prior_name:<12}  {res['mean']:>12.4f}  {res['std']:>12.4f}")


if __name__ == "__main__":
    main()
