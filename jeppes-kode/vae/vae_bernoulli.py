# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import numpy as np
from torch.nn import functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from tqdm import tqdm
from flow import Flow, MaskedCouplingLayer


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MixtureOfGaussianPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a MoG prior distribution with learnable mean, variance, and weight.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of Gaussian components.
        """
        super(MixtureOfGaussianPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=True)
        self.std = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.weights = nn.Parameter(torch.ones(self.K) / K, requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.MixtureSameFamily(
            td.Categorical(self.weights),
            td.Independent(td.Normal(loc=self.mean, scale=F.softplus(self.std)), 1)
        )

class FlowPrior(nn.Module):
    def __init__(self, M, K=8, H=64):
        """
        Define a Flow-based prior with trainable scaling and translation nets.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int] 
           Number of flow transformations.
        H: [int] 
           Size of hidden layers in scale and translation networks.
        """
        super(FlowPrior, self).__init__()
        self.M = M
        # "chequerboard" mask
        self.mask = torch.tensor([i % 2 for i in range(M)], dtype=torch.float32)
        
        # define normalizing flow
        self.transformations = self.build_transformations(K, H)
        self.flow = Flow(GaussianPrior(M), self.transformations)

    def build_transformations(self, num_transformations, num_hidden):
        transformations = []
        
        for _ in range(num_transformations):
            self.mask = (1-self.mask) # flip mask
            scale_net = nn.Sequential(nn.Linear(self.M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, self.M), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(self.M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, self.M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, self.mask))
        
        return transformations

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return self.flow


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        if type(self.prior) != GaussianPrior:
            q = self.encoder(x)
            z = q.rsample()
            elbo = torch.mean(self.decoder(z).log_prob(x) - (q.log_prob(z) - self.prior().log_prob(z)), dim=0)
            return elbo
        
        else:
            q = self.encoder(x)
            z = q.rsample()
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
            return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'plot-latent', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='type of prior (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--latent-samples', type=int, default=4000, metavar='N', help='number of samples in latent space visualization (default: %(default)s)')
    parser.add_argument('--mixture-k', type=int, default=10, metavar='N', help='number of Gaussians to use in MoG prior (default: %(default)s)')
    parser.add_argument('--flow-transformations', type=int, default=10, metavar='N', help='number of transformations to use in flow-based prior (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
    print("")

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
        batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
        batch_size=args.batch_size, shuffle=True
    )

    # Define prior distribution
    M = args.latent_dim
    if args.prior == "gaussian":
        prior = GaussianPrior(M)
    elif args.prior == "mog":
        prior = MixtureOfGaussianPrior(M, args.mixture_k)
    elif args.prior == "flow":
        prior = FlowPrior(M, args.flow_transformations)
    else:
        print(f"Unknown prior '{args.prior}'")
        exit()

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # Generate samples
        with torch.no_grad():
            samples = (model.sample(4)).cpu() 
            save_image(samples.view(4, 1, 28, 28), args.samples)
    
    elif args.mode == 'plot-latent':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # Generate latent-space representations
        latent_vectors = []
        labels = []
        count = 0
        with torch.no_grad():
            for batch_imgs, batch_labels in mnist_test_loader:
                batch_imgs = batch_imgs.to(args.device)
                q = model.encoder(batch_imgs)
                z = q.rsample()

                labels.append(batch_labels.cpu())
                latent_vectors.append(z.cpu())
                
                count += batch_imgs.size(0)
                if count >= args.latent_samples:
                    break

        labels = torch.cat(labels).numpy()
        latent_vectors = torch.cat(latent_vectors).numpy()

        pca_posterior = PCA(n_components=2)
        latent_2d = pca_posterior.fit_transform(latent_vectors)
        posterior_explained_var = np.sum(pca_posterior.explained_variance_ratio_) * 100
        
        # Prepare prior samples
        with torch.no_grad():
            prior_samples = model.prior().sample(torch.Size([args.latent_samples]))
        pca_prior = PCA(n_components=2)
        prior_samples = prior_samples.cpu().numpy()
        prior_2d = pca_prior.fit_transform(prior_samples)
        prior_explained_var = np.sum(pca_prior.explained_variance_ratio_) * 100

        # Side-by-side plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: Aggregate posterior
        scatter = axes[0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=10)
        fig.colorbar(scatter, ax=axes[0], ticks=range(10))
        axes[0].set_xlabel('PCA 1')
        axes[0].set_ylabel('PCA 2')
        axes[0].set_title(f'Aggregate posterior (test set)\nPCA explained variance: {posterior_explained_var:.2f}%')
        colors = cm.get_cmap('tab10', 10)
        for digit in range(10):
            idx = labels == digit
            class_points = latent_2d[idx]
            mean = class_points.mean(axis=0)
            std = class_points.std(axis=0)
            ell = Ellipse(xy=mean, width=2*std[0], height=2*std[1], angle=0,
                          color=colors(digit), alpha=0.3, lw=2, zorder=10)
            #axes[0].add_patch(ell)
        axes[0].set_aspect('equal', adjustable='datalim')

        # Right: Prior samples
        axes[1].scatter(prior_2d[:, 0], prior_2d[:, 1], c='gray', s=10, alpha=0.7)
        axes[1].set_xlabel('PCA 1')
        axes[1].set_ylabel('PCA 2')
        axes[1].set_title(f'Prior\nPCA explained variance: {prior_explained_var:.2f}%')
        axes[1].set_aspect('equal', adjustable='datalim')

        plt.tight_layout()
        plt.show()
    
    elif args.mode == "eval":
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        
        total_elbo = 0
        total_imgs = 0
        with torch.no_grad():
            for batch_imgs, _ in mnist_test_loader:
                batch_imgs = batch_imgs.to(args.device)
                total_elbo += model.elbo(batch_imgs).item() * len(batch_imgs)
                total_imgs += len(batch_imgs)

        mean_elbo = total_elbo / total_imgs
        print(f"Mean ELBO for test set: {mean_elbo}")

    else:
        print(f"Unknown mode '{args.mode}'")
