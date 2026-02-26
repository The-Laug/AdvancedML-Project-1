### Exercise 2 Documentation

**Overview**

This document explains the changes I implemented to add evaluation and visualization functionality to the `vae_bernoulli.py` VAE script. The new features are:

- ELBO evaluation on the binarized MNIST test set.
- Aggregate-posterior sampling for all test datapoints.
- 2D visualization of aggregate-posterior samples colored by true labels.
- PCA projection applied when latent dimension `M > 2`.

**Files changed / added**

- `vae_bernoulli.py` — main changes: imports, a new CLI mode `eval`, and three helper functions used by evaluation and plotting.
- `pyproject.toml` — added `scikit-learn` dependency for PCA.
- `README.md` — added example command for evaluation/plotting.

Paths in repository:

- File: [vae_bernoulli.py](vae_bernoulli.py)
- File: [pyproject.toml](pyproject.toml)
- File: [README.md](README.md)
- Documentation: [exercises/exercise_2.md](exercises/exercise_2.md)

**What I implemented in `vae_bernoulli.py`**

1) New CLI mode `eval` (added to `argparse` choices). Running the script with `eval` loads a saved model and performs the evaluation workflow.

2) New imports used for evaluation and plotting:

- `import numpy as np`
- `import matplotlib.pyplot as plt`
- `from sklearn.decomposition import PCA`

3) Helper functions added inside the `eval` branch:

- `evaluate_test_elbo(model, data_loader, device)`
  - Iterates the test `data_loader` and computes per-example ELBO terms: log p(x|z) (via `decoder(z).log_prob(x)`) minus KL `td.kl_divergence(q, prior)`.
  - Sums the per-example ELBOs and divides by the number of examples to return the average ELBO (nats per example).
  - Uses `torch.no_grad()` and `model.eval()` to avoid gradient computation and dropout/BatchNorm side effects.

- `aggregate_posterior_samples(model, data_loader, device)`
  - Encodes every test example: `q = model.encoder(x)` and calls `q.rsample()` to get a latent sample per example.
  - Collects and concatenates all latent samples into a NumPy array `zs` shape `(N_test, M)` and the corresponding labels array `labels` shape `(N_test,)`.
  - Note: this collects all samples in memory; for very large datasets or M, consider streaming or incremental PCA.

- `plot_aggregate(zs, labels, out_path, M)`
  - If `M > 2`, runs `PCA(n_components=2)` on `zs` and projects to two principal components. Otherwise, uses the first two latent dims directly.
  - Creates a scatter plot colored by `labels` (colormap `tab10`) and saves to `out_path` (PNG).

4) The `eval` workflow

- Loads the model state via `torch.load(args.model, map_location=torch.device(args.device))`.
- Calls `evaluate_test_elbo(...)` and prints the average test ELBO.
- Calls `aggregate_posterior_samples(...)`, runs PCA if needed, and saves a 2D scatter plot where each point corresponds to a test example and is colored by its true label.

**Why PCA for M>2**

For visualization we need 2D coordinates. When latent dimensionality `M` is greater than two, the code uses PCA to project the `M`-dimensional aggregate posterior samples to the first two principal components. This is done using scikit-learn's `PCA(n_components=2)`.

**Usage**

Train and save your model (if you have not already):

```bash
poetry run python vae_bernoulli.py train --model model.pt
```

Evaluate and produce plot:

```bash
poetry run python vae_bernoulli.py eval --model model.pt --samples aggregate_posterior.png
```

- `--model` points to the model file to load/save.
- `--samples` is reused as the output filename for the saved aggregate-posterior plot.

**Outputs produced**

- Printed average test ELBO (nats per example) on STDOUT.
- A PNG file (default `aggregate_posterior.png`) containing the colored scatter plot of aggregate-posterior samples.

**Assumptions and implementation notes**

- The script assumes MNIST is loaded binarized exactly as in the training code (threshold at 0.5).
- `q.rsample()` is used to draw reparameterized latent samples (so gradients flow during training; for evaluation the use is for sampling only).
- `decoder(z).log_prob(x)` returns per-example log-probabilities because the decoder is wrapped in `td.Independent(..., reinterpreted_batch_ndims=2)`; that sums log-prob across pixel event dims.
- `td.kl_divergence(q, prior)` returns a per-example KL (since encoder `q` is `Independent(..., reinterpreted_batch_ndims=1)`).
- The ELBO computed here is Monte Carlo estimate using a single `rsample()` per datapoint (same as used in training code).

**Caveats & possible improvements**

- Memory: collecting all `zs` in memory can be large if the test set or `M` is large. Alternatives:
  - Subsample the test set for plotting.
  - Use incremental PCA (`sklearn.decomposition.IncrementalPCA`) to avoid retaining all samples.
  - Save intermediate `zs` to disk and run PCA separately.
- Evaluation metric: current ELBO is averaged across examples; for reporting you may want per-datapoint breakdowns or NLL in bits/dim.
- Reproducibility: for deterministic PCA and plotting, set random seeds when needed.

**Dependencies**

- Project dependencies updated in `pyproject.toml`: `scikit-learn` (for PCA), `matplotlib`, `numpy` (already commonly available). After editing `pyproject.toml`, run:

```bash
poetry lock
poetry install
```

**Next steps you might want me to do**

- Run `poetry lock` and attempt `poetry install` and report any dependency conflicts.
- Replace full in-memory aggregation with incremental PCA for lower memory footprint.
- Save a small grid of decoded samples for a set of grid points in latent space (useful when `M=2`).

If you'd like, I can now run `poetry lock` and attempt `poetry install` here and report any errors. Otherwise you can run the commands above locally and I can help interpret any install errors.

***

Generated by the implementation changes requested in Exercise 1.5.
