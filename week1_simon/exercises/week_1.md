### Exercise 1.4 In this first exercise, you should just inspect the code in vae_bernoulli.py.


Answer the following questions:

## 1. How is parameterization handled in the code

By rsample in the elbo function in the VAE class

## 2. Consider the implementation of the ELBO. What is the dimension of 
```self.decoder(z).log_prob(x) and of td.kl_divergence(q, self.prior.distribuion)?```

(shapes; B = batch size, M = latent dim):

x has shape (B, 28, 28).

encoder_net(x) → (B, 2*M); mean, std → (B, M) each.

q = self.encoder(x) is an Independent(Normal) with batch_shape (B,) and event_shape (M,).

z = q.rsample() → (B, M) (latent samples).

decoder(z) is an Independent(Bernoulli) with batch_shape (B,) and event_shape (28, 28).

self.decoder(z).log_prob(x) → (B,) — the log-probability of each image (pixels summed).

td.kl_divergence(q, self.prior()) → (B,) — KL per example (q depends on x even if the prior does not).

torch.mean(..., dim=0) then reduces (B,) to a scalar ELBO.

Note: the model uses Bernoulli per-pixel outputs (binary pixels), not “two classes” for the whole example.

## 3. The implementation of the prior, encoder and decoder classes all make use of td.Independent. What does this do?

`td.Independent` reinterprets one or more trailing batch dimensions of a base distribution as event dimensions. Effects:

- The reinterpreted dims become part of `event_shape`, so `log_prob` sums across them and returns a tensor shaped by the remaining `batch_shape` (typically one scalar per example).
- Sampling (`sample`/`rsample`) and KL computations treat those dims as a single multivariate event, producing correctly-shaped joint samples and KL values.
- In this code: the encoder reinterprets the latent dimension as an event (so `q.rsample()` yields `(B, M)` and `td.kl_divergence` gives a per-example KL), while the decoder treats the 28×28 pixel axes as the event (so `log_prob` returns one log-likelihood per image).

## 4. What is the purpose using the function torch.chunk in GaussianEncoder.forward?

`torch.chunk` splits the encoder network output of size `2*M` along the last axis into two tensors of size `M` each. The network therefore predicts both the Gaussian mean and a second parameter (here used as a log-scale). Splitting lets the code use the first chunk as `mean` and convert the second chunk into a positive `scale` via `torch.exp`, yielding the `Normal(loc=mean, scale=exp(std))` used by the encoder.