# Plan: Exercises 1.6 — 1.8

This document outlines a practical implementation plan, tasks, checks and commands to implement Exercises 1.6–1.8 for the `vae_bernoulli.py` project.

Summary of deliverables
- Exercise 1.6: Replace the Gaussian prior with a Mixture-of-Gaussians (MoG) prior (prefer `MixtureSameFamily`). Re-formulate ELBO and evaluate on test set; produce aggregate-posterior plots and compare clustering vs. Gaussian prior.
- Exercise 1.7: Replace Bernoulli decoder with a multivariate Gaussian decoder (fixed and learned variance modes). Evaluate sample quality and compare p(x|z) mean visualizations.
- Exercise 1.8 (optional): Implement CNN encoder/decoder and compare sample quality and ELBO.

High-level approach
- Make small, incremental changes and keep training/evaluation scripts compatible with existing CLI (`train`, `sample`, `eval`).
- Reuse data loading and plotting code added earlier (aggregate posterior sampling + PCA projection).
- Add unit/functional checks to verify shapes and ELBO computation before full training.

Dependencies
- PyTorch (existing)
- torchvision (existing)
- scikit-learn (PCA; already added)
- (optional) `continuous-bernoulli` package if you try continuous Bernoulli, otherwise implement formula yourself.

Files to change / create
- `vae_bernoulli.py` — main changes for 1.6, 1.7, 1.8; add flags and helper functions.
- `exercises/planning_exercises_1_6_to_1_8.md` — this document.
- `exercises/exercise_2.md` — already created; update if necessary.
- `exercises/test_scripts/` — small scripts to run quick checks (optional).

Implementation plan (step-by-step)

Exercise 1.6 — MoG prior
1. Design API for MoG prior class (e.g. `MixturePrior`):
   - Option A: implement `MixtureSameFamily` wrapper that returns a `torch.distributions.Distribution` on `forward()`.
   - Option B: implement custom mixture class (only if needed).
   - Parameters to expose: number of components K, component means shape (K, M), component stds shape (K, M), (optional) mixture logits (K).
2. Implement `MixturePrior` class in `vae_bernoulli.py` or separate module. Example using PyTorch:
   - cat = Categorical(logits=mixture_logits)
   - comp = Independent(Normal(loc=means, scale=scales), 1)  # event=M
   - mix = MixtureSameFamily(cat, comp)
3. Modify ELBO formulation if needed:
   - ELBO per-example: E_{q(z|x)}[log p(x|z)] - E_{q(z|x)}[log q(z|x) - log p(z)]
   - `td.kl_divergence(q, prior)` can work if `prior` is a Distribution object implementing `log_prob` for samples; using `kl_divergence` directly may not be supported for MixtureSameFamily—compute KL manually as E_q[log q(z|x) - log p(z)] using `q.log_prob(z)` and `prior.log_prob(z)`.
4. Implement and test small shape checks:
   - Create a dummy batch, compute q = encoder(x), z = q.rsample(), check `q.log_prob(z)` shape is (B,), `prior.log_prob(z)` is (B,) and per-example KL = q.log_prob(z)-prior.log_prob(z).
5. Train or fine-tune model with MoG prior; evaluate test ELBO using the evaluation function implemented earlier.
6. Visualize aggregate posterior samples and compare clustering vs Gaussian prior plot.

Notes and caveats for 1.6
- Mixture priors often give multi-modal latent priors that can improve clustering but may need careful initialization (e.g. spread component means).
- Computing analytic KL with MoG is non-trivial; use Monte-Carlo estimate KL = E_q[log q(z|x) - log p(z)] computed via samples z ~ q.

Exercise 1.7 — Continuous pixel outputs (Gaussian decoder)
1. Implement a new decoder class `GaussianDecoder`:
   - Output: network produces mean image μ(x) shape (B, 28, 28) and optionally a log-variance parameter per pixel (B, 28, 28) or a single scalar variance.
   - Wrap as `Independent(Normal(loc=mean, scale=scale), 2)` so `log_prob` returns a per-example scalar.
2. Add CLI flags for decoder mode: `--output dist` with options `bernoulli|gaussian-fixed|gaussian-learned`.
3. If fixed variance mode: set `scale = fixed_sigma` (e.g. 0.1) — not learned.
4. If learned variance mode: predict `log_var` from network, convert to `scale = exp(0.5*log_var)` or use `softplus` for stability.
5. Update training and evaluation code paths to use the chosen decoder and compute ELBO (use existing ELBO implementation which uses decoder.log_prob(z). If using Monte Carlo KL for MoG, account for that.)
6. Train models for both fixed and learned variance settings; evaluate test ELBO.
7. For qualitative check: sample z~p(z) (prior) and visualize `decoder.mean(z)` rather than sampling from output — compare mean images.

Notes and caveats for 1.7
- Pixel-scale and loss scale matter: compare metrics in nats or bits/dim.
- If variance is learned per pixel, ensure numerical stability (`clamp` or `softplus`) and reasonable initialization.

Exercise 1.8 — CNN encoder/decoder (optional)
1. Implement CNN encoder: conv layers → flattened → linear → output 2*M. Ensure output shape remains `(B, 2*M)`.
2. Implement CNN decoder: map latent z → FC → reshaped feature map → ConvTranspose layers → image shape (1, 28, 28) logits or mean.
3. Add CLI flag `--arch` with options `mlp|cnn` and `--output dist` to choose decoder distribution.
4. Train CNN variants, evaluate test ELBO and qualitatively inspect samples.

Testing and verification
- Unit checks:
  - Shapes: check q.log_prob(z), prior.log_prob(z), decoder.log_prob(x) shapes are (B,).
  - For MoG prior check manual Monte-Carlo KL computation equals `q.log_prob(z)-prior.log_prob(z)` averaged over z samples.
- Functional checks:
  - Run `poetry run python vae_bernoulli.py eval --model model.pt` on a small saved model or after a few epochs to ensure evaluation pipeline runs and produces plots.
- Compare runs:
  - Baseline: existing Gaussian prior + Bernoulli decoder.
  - MoG prior: compare test ELBO and clustering plots.
  - Gaussian decoder: compare sample quality and mean images p(x|z).
  - CNN architectures: compare improvements.

Estimated effort and order of work (suggested)
- Phase A (1.6 baseline + tests): 3–6 hours
  - Implement MixturePrior, change ELBO to monte-carlo KL, unit tests, small train run.
- Phase B (1.7 continuous outputs): 3–6 hours
  - Implement GaussianDecoder (fixed & learned), add CLI flags, train and compare.
- Phase C (1.8 optional CNN): 4–8 hours
  - Implement CNN encoder/decoder, rerun experiments.

Example quick commands
```bash
# train baseline (Bernoulli + Gaussian prior)
poetry run python vae_bernoulli.py train --model model_baseline.pt --latent-dim 32

# train with MoG prior (assumes new CLI flag --prior mog and optional --mog-components K)
poetry run python vae_bernoulli.py train --model model_mog.pt --latent-dim 32 --prior mog --mog-components 10

# evaluate and plot
poetry run python vae_bernoulli.py eval --model model_mog.pt --samples mog_aggregate.png

# train Gaussian-output decoder (learned variance)
poetry run python vae_bernoulli.py train --model model_gauss.pt --output gaussian-learned

# show mean images for z ~ p(z)
poetry run python vae_bernoulli.py sample --model model_gauss.pt --samples mean_samples.png --show-mean
```

Implementation tips and pitfalls
- For MoG prior KL: prefer MC estimate KL = E_q[log q - log p] computed with 1–10 samples for stability. Be careful to use same device/dtype.
- Initialize MoG component means (spread across latent space) to avoid collapse.
- When using learned variances predict `log_var`; use `scale = torch.exp(0.5 * log_var)` or `softplus` for positivity and stability.
- If `td.kl_divergence` does not support `MixtureSameFamily`, compute KL via `q.log_prob(z) - prior.log_prob(z)`.

Next actions I can take for you
- Implement Exercise 1.6 MoG prior in `vae_bernoulli.py` (code + tests).
- Implement Gaussian decoder variants and CLI flags for Exercise 1.7.
- Implement CNN encoder/decoder (Exercise 1.8) if desired.

If you want me to start, tell me which exercise to implement first (I recommend starting with 1.6), and I will open a focused plan and begin patching the code and adding tests.
