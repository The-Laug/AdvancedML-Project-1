# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet
from vae import VAE, GaussianEncoder, GaussianPrior

# ==== Classes for Beta-VAE ====
class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """Define a Gaussian decoder distribution based on a given decoder network."""
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        mean, log_std = torch.chunk(self.decoder_net(z), 2, dim=-1)
        mean = mean.view(-1, 28, 28)
        std = torch.exp(log_std).view(-1, 28, 28)
        return td.Independent(td.Normal(loc=mean, scale=std), 2)

class BetaVAE(VAE):
    """A simple extension of VAE to support a Beta value."""
    def __init__(self, prior, decoder, encoder, beta):           
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta

    def elbo(self, x):
        # reshape flattened input to (batch, 28, 28) if needed
        if x.dim() == 2 and x.shape[1] == 784:
            x = x.view(-1, 28, 28)
        
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - self.beta * (q.log_prob(z) - self.prior().log_prob(z)), dim=0)
        return elbo


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### JEPPE: ALGORITHM 1 ###
        eps = torch.randn_like(x)
        ts = torch.randint(1, self.T+1, (x.shape[0], 1), device=x.device)
        alpha_bar = self.alpha_cumprod[ts - 1]
        
        res = torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * eps
        y = self.network(res, (ts - 1) / (self.T - 1))  # normalized timestep

        neg_elbo = torch.norm((eps - y), dim=tuple(range(1, eps.dim())))
        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T, 0, -1):
            ### JEPPE: ALGORITHM 2 ###
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t) 

            frac = (1 - self.alpha[t-1]) / torch.sqrt((1 - self.alpha_cumprod[t-1]))
            ts = torch.full((x_t.shape[0],), t-1, device=x_t.device).view(-1, 1)  # normalized timestep

            x_t = 1 / torch.sqrt(self.alpha[t-1])  *  (x_t - frac * self.network(x_t, ts / (self.T - 1))) + torch.sqrt(self.beta[t-1]) * z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a model (DDPM or VAE).

    Parameters:
    model: [nn.Module]
       The model to train.
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
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x) if hasattr(model, 'loss') else model(x)  # dirty fix to support both ddpm and vae
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim+1, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, input_dim)
        )
        
    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


class LatentDDPM(nn.Module):
    def __init__(self, vae, ddpm):
        """Wrapper class that combines a frozen VAE encoder with a DDPM in latent space."""
        super(LatentDDPM, self).__init__()
        self.vae = vae
        self.ddpm = ddpm
        
    def loss(self, x):
        """Compute DDPM loss in latent space."""
        # encode images to latent space
        with torch.no_grad():
            # reshape if needed
            if x.dim() == 2 and x.shape[1] == 784:
                x = x.view(-1, 28, 28)
            
            # get latent representation (use mean)
            q_z = self.vae.encoder(x)
            z = q_z.rsample()
        
        return self.ddpm.loss(z)
    
    def sample(self, shape):
        """Sample from latent DDPM and decode to images."""
        # get latent dimension from DDPM network
        latent_dim = self.ddpm.network.network[0].in_features - 1  # -1 for time dimension
        
        # sample from DDPM in latent space
        z_samples = self.ddpm.sample((shape[0], latent_dim))
        
        # decode latent samples to images (use mean for sharp outputs)
        with torch.no_grad():
            x_dist = self.vae.decoder(z_samples)
            x_samples = x_dist.mean  # Use mean instead of sample() for sharper images
        
        return x_samples.view(shape[0], -1) 


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'fid', 'plot-ddpm-latent'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--type', type=str, default='ddpm', choices=['ddpm', 'bvae', 'latent-ddpm'], help='type of model to train (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--bvae-model', type=str, default='bvae-model.pt', help='file to load pre-trained Beta-VAE model from (required in latent-ddpm) (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=64, metavar='N', help='number of latent dimensions for Beta-VAE (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=1.0, metavar='V', help='beta used in Beta-VAE (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x + torch.rand(x.shape) / 255),
        transforms.Lambda(lambda x : (x -0.5) *2.0),
        transforms.Lambda(lambda x : x.flatten())
    ])

    train_dataset = datasets.MNIST('data/',
        train=True,
        download=True,
        transform=transform 
    )
    
    test_dataset = datasets.MNIST('data/',
        train=False,
        download=True,
        transform=transform 
    )
    
    # wrap datasets to return only images, not labels
    class UnlabeledDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.dataset[idx][0]
    
    train_loader = torch.utils.data.DataLoader(UnlabeledDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(UnlabeledDataset(test_dataset), batch_size=args.batch_size, shuffle=False)

    # get the dimension of the dataset
    D = next(iter(train_loader)).shape[1]

    M = args.latent_dim
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
        nn.Linear(512, 784*2),
    )

    decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    prior = GaussianPrior(M)

    # Set the number of steps in the diffusion process
    T = 1000

    # define model
    if args.type == "ddpm":
        network = Unet()
        model = DDPM(network, T=T).to(args.device)

    elif args.type == "bvae":
        model = BetaVAE(prior, decoder, encoder, args.beta).to(args.device)

    elif args.type == 'latent-ddpm':
        # load pre-trained BetaVAE
        bvae_model = BetaVAE(prior, decoder, encoder, args.beta)
        bvae_model.load_state_dict(torch.load(args.bvae_model, map_location=torch.device(args.device)))
        bvae_model.eval()
        bvae_model.to(args.device)

        # freeze VAE parameters
        for param in bvae_model.parameters():
            param.requires_grad = False
    
        # create DDPM for latent space
        network = FcNetwork(M, 512)
        ddpm = DDPM(network, T=T)
        
        # wrap in LatentDDPM
        model = LatentDDPM(bvae_model, ddpm).to(args.device)

    else:
        print(f"Unknown type '{args.type}'")
        exit()

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = model.sample((4, D)).cpu()

        # Transform the samples back to the original space
        samples = samples / 2 + 0.5

        # Reshape and save as image grid
        samples = samples.view(-1, 1, 28, 28)
        save_image(samples, args.samples, nrow=8)

    elif args.mode == 'fid':
        from fid import compute_fid
        from time import time

        fid_test_dataset = datasets.MNIST('data/',
            train=False,
            download=True,
            transform=transform 
        )

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        real_samples = []

        for i in range(1000):
            real_samples.append(fid_test_dataset[i][0])
        real_samples = torch.stack(real_samples).to(args.device)
        real_samples = real_samples.view(-1, 1, 28, 28)
        
        t0 = time()
        with torch.no_grad():
            samples = model.sample((1000, D)).to(args.device)
        t = time() - t0
        print(f"Time for 1000 samples: {t}")

        # Keep samples in [-1, 1] range to match real_samples
        samples = samples.view(-1, 1, 28, 28)

        print(f"FID: {compute_fid(real_samples, samples, args.device)}")

    elif args.mode == 'plot-ddpm-latent':
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        # Aggregate posterior: encode test images through the VAE encoder
        z_aggregate = []
        with torch.no_grad():
            for x in test_loader:
                x = x.to(args.device)
                if x.dim() == 2 and x.shape[1] == 784:
                    x = x.view(-1, 28, 28)
                q_z = model.vae.encoder(x)
                z_aggregate.append(q_z.mean)
        z_aggregate = torch.cat(z_aggregate).cpu().numpy()

        # Prior: sample from prior
        z_prior = torch.randn(len(z_aggregate), M).numpy()

        # Latent DDPM: sample z (before decoding)
        with torch.no_grad():
            z_ddpm = model.ddpm.sample((len(z_aggregate), M)).cpu().numpy()

        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        pca.fit(z_aggregate) 
        
        z_aggregate_2d = pca.transform(z_aggregate)
        z_prior_2d = pca.transform(z_prior)
        z_ddpm_2d = pca.transform(z_ddpm)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].scatter(z_prior_2d[:, 0], z_prior_2d[:, 1], alpha=0.3, s=1)
        #axes[0].set_title('Prior')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_xlim(-5, 5)
        axes[0].set_ylim(-5, 5)
        
        axes[1].scatter(z_ddpm_2d[:, 0], z_ddpm_2d[:, 1], alpha=0.3, s=1)
        #axes[1].set_title('Latent DDPM Distribution')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_xlim(-80, 20)
        axes[1].set_ylim(-10, 50)
        
        axes[2].scatter(z_aggregate_2d[:, 0], z_aggregate_2d[:, 1], alpha=0.3, s=1)
        #axes[2].set_title('Aggregate Posterior')
        axes[2].set_xlabel('PC1')
        axes[2].set_ylabel('PC2')
        axes[2].set_xlim(-80, 20)
        axes[2].set_ylim(-10, 50)

        plt.tight_layout()
        plt.savefig('latent_distributions.png', dpi=150)
        plt.show()

    else:
        print(f"Unknown mode '{args.mode}'")
        exit()