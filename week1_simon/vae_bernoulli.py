# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

### FROM FLOW ###########################################################################################


class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Define a Gaussian base distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Return the base distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        """
        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        bz = self.mask * z
        log_scale = 0.5 * torch.tanh(self.scale_net(bz))
        z_prime = bz + (1-self.mask) * (z * torch.exp(log_scale) + self.translation_net(bz)) 
        log_det_J = torch.sum((1-self.mask) * log_scale, dim=1)
        return z_prime, log_det_J
    
    def inverse(self, z_prime):
        """
        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        bz_prime = self.mask * z_prime
        log_scale = 0.5 * torch.tanh(self.scale_net(bz_prime))
        z = bz_prime + (1-self.mask) * (z_prime - self.translation_net(bz_prime)) * torch.exp(-log_scale)
        log_det_J = torch.sum((1-self.mask) * (-log_scale), dim=1)

        return z, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))

#############################################################################################################

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

class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians prior distribution.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of mixture components.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.M = M
        self.K = K
        means_init = 0.05 * torch.randn(K, M)
        self.means = nn.Parameter(means_init)
        self.logvars = nn.Parameter(torch.randn(K, M))
        self.w_logits = nn.Parameter(torch.zeros(K))

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        probs = F.softmax(self.w_logits, dim=-1)
        mix = td.Categorical(probs=probs)
        scales = F.softplus(self.logvars) + 1e-4  
        comp = td.Independent(td.Normal(loc=self.means, scale=scales), 1)
        return td.MixtureSameFamily(mix, comp)



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
        self.std = nn.Parameter(torch.ones(28, 28)*0.1, requires_grad=False)
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class MultivariateGaussianDecoder(nn.Module):
    def __init__(self, decoder_net, learn_variance=True):
        """
        Define a Multivariate Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """

        
        super(MultivariateGaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        init_raw = torch.log(torch.exp(torch.tensor(0.1)) - 1.0) 
        self.learn_variance = learn_variance
        if learn_variance:
            raw_var = init_raw * torch.ones(28, 28)    
            self.std = nn.Parameter(raw_var, requires_grad=True)
        else:
            self.std = nn.Parameter(torch.ones(28, 28)*0.1, requires_grad=False)


    def forward(self, z):
        """
        Given a batch of latent variables, return a Multivariate Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)
        if self.learn_variance:
            std = F.softplus(self.std) + 1e-4
            return td.Independent(td.Normal(loc=mean, scale=std), 2)
        else: 
            return td.Independent(td.Normal(loc=mean, scale=self.std), 2)


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

    def elbo(self, x, n_samples=1):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        if type(self.prior)==MixtureOfGaussiansPrior:

            
            q = self.encoder(x)
            z = q.rsample()
            log_q = q.log_prob(z)
            log_p = self.prior().log_prob(z)
            KL = log_q - log_p
            elbo = torch.mean(self.decoder(z).log_prob(x) - KL, dim=0)
            return elbo
        
        elif type(self.prior)==Flow:
            
            q = self.encoder(x)
            z = q.rsample()
            log_q = q.log_prob(z)
            log_p = self.prior.log_prob(z)
            KL = log_q - log_p
            elbo = torch.mean(self.decoder(z).log_prob(x) - KL, dim=0)
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
        if type(self.prior)==Flow:
            z = self.prior.sample(torch.Size([n_samples]))
        else:
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


def evaluate_test_elbo(model, data_loader, device):
    model.eval()
    total_elbo = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            images, _ = batch
            images = images.to(device)
            q = model.encoder(images)
            z = q.rsample()
            log_px = model.decoder(z).log_prob(images)
            test_elbo = model.elbo(images)
            total_elbo += test_elbo.sum().item()
            total_samples += images.shape[0]
    return total_elbo / total_samples


def aggregate_posterior_samples(model, data_loader, device):
    model.eval()
    zs = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            images, labs = batch
            images = images.to(device)
            q = model.encoder(images)
            z = q.rsample()  # shape (B, M)
            zs.append(z.cpu().numpy())
            labels.append(labs.numpy())
    zs = np.concatenate(zs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return zs, labels


def prior_samples(model, data_loader, device):
    model.eval()
    zs = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            if args.prior == 'flow':
                z = model.prior.sample(torch.Size([64]))
            else:
                z = model.prior().sample(torch.Size([64]))    
            zs.append(z.cpu().numpy())

    zs = np.concatenate(zs, axis=0)
    return zs


def plot_aggregate(zs, labels, out_path, M):
    if M > 2:
        tsne = TSNE(n_components=2)
        Z2 = tsne.fit_transform(zs)
    else:
        Z2 = zs[:, :2]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap='tab10', s=6, alpha=0.8)
    plt.colorbar(scatter, ticks=range(10))
    plt.xlabel('T-SNE dim 1')
    plt.ylabel('T-SNE dim 2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_prior(z, out_path, M):
    if M > 2:
        tsne = TSNE(n_components=2)
        Z2 = tsne.fit_transform(z)
    else:
        Z2 = z[:, :2]

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], s=6, alpha=0.8)
    plt.xlabel('T-SNE dim 1')
    plt.ylabel('T-SNE dim 2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'MoG', 'flow'], help='prior distribution to use (default: %(default)s)')
    parser.add_argument('--binarized-mnist', type=bool, default=False, help='whether to use binarized MNIST (default: %(default)s)')
    parser.add_argument('--only_prior', type=bool, default=False, help='whether to only sample from the prior (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    if not args.binarized_mnist:
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    else:
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'MoG':
        K = 10
        prior = MixtureOfGaussiansPrior(M, K)

    # FLOW PRIOR
    elif args.prior == 'flow':

        base = GaussianBase(M)

        # Define transformations
        transformations =[]
        mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
        
        num_transformations = 16
        num_hidden = 64
        mask = torch.zeros((M,))
        mask[::2] = 1

        for i in range(num_transformations):
            mask = (1-mask) # Flip the mask
            scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
            translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        # Define flow model
        prior = Flow(base, transformations).to(args.device)

    else:
        prior = GaussianPrior(M)

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    if args.binarized_mnist:
        decoder = MultivariateGaussianDecoder(decoder_net)
    else:
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

        # Generate samples
        model.eval()
        with torch.no_grad():
            if args.only_prior:
                if args.prior == 'flow':
                    z = model.prior.sample(torch.Size([64]))
                else:
                    z = model.prior().sample(torch.Size([64]))
                save_image(model.decoder(z).mean.view(64, 1, 28, 28), args.samples)

            else:
                samples = (model.sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'eval':
        # load model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.to(device)

        # Use module-level evaluation and plotting helpers defined above.

        # Evaluate ELBO on test set
        test_elbo = evaluate_test_elbo(model, mnist_test_loader, torch.device(args.device))
        print(f"Test ELBO (nats per example): {test_elbo:0.6f}")

        # Aggregate posterior sampling and plotting
        zs, labs = aggregate_posterior_samples(model, mnist_test_loader, torch.device(args.device))
        z = prior_samples(model, mnist_test_loader, torch.device(args.device))
        plot_path_prior = args.samples.replace('.png', '_prior.png') if args.samples else 'prior_samples.png'
        plot_path = args.samples if args.samples else 'aggregate_posterior.png'
        plot_aggregate(zs, labs, plot_path, M)
        plot_prior(z, plot_path_prior, M)
        print(f"Saved aggregate posterior plot to {plot_path}")
        print(f"Saved prior plot to {plot_path_prior}")
