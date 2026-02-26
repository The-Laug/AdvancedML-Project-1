
### Introduction

in week1_simon/vae_bernoulli.py, we have some VAE for respectively a gaussian, mixture of gaussian and a flow prior. The goal is to send a job and script to the HPC that can evaluate our models for us. 


Take a look at the arguments:

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


We are interested in the following:
For each of the 3 priors, we want to train 10 models and evaluate them 10 times. The goal is to have a test-set ELBO mean and standard deviation for each of the three models. Hyperparameters:

--latent-dim = 10
--device = cuda
Let all other arguments be default
```
elbos ={}
for prior in ['gaussian', 'MoG', 'flow'], 
    elbos = []
    for i in range(0,10)
        train model
        eval model
        elbos.append(elbo)
    elbos[prior] = elbo

save elbos in json
```
Your task:

Either make a script or file that can do this using the functions/files you find in week1_simon/vae_bernoulli.py

You should also make a bash job that runs this on a single GPU on the cluster.

