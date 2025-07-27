# How DDPM Learns the Target Distribution
Generative models are defined by their ability to learn the distribution of the training data $`p(x)`$, and DDPM is one example of such a model. But how does DDPM learn this? Let me explain in simple language. We'll start with VAE, which is simpler, and then move on to DDPM later.

## 1. How VAE Learns
VAE generates data samples in two steps:
1. Sample $`z`$ from the prior distribution $`p(z)`$
2. Sample $`x`$  from the conditional distribution $`p(x|z)`$

These steps define the joint distribution: $`p(x,z)=p(z)p(x|z)`$. But ultimately, we don't just want $`p(x,z)`$. We want the model to learn the distribution over $`x`$ alone, that is: $`p(x)`$.

Generally speaking, to check whether our model has learned the target distribution $`p(x)`$, we can generate many samples $`x`$ and compare their distribution to $`p(x)`$.

According to [Kingma et al. (2013), "2.1 Problem scenario"](#3-1-auto-encoding-variational-bayes), we can obtain $`p(x)`$ by marginalizing out $`z`$ from the joint distribution:

$`p(x)=\int p(x,z)dz=\int p(z)p(x|z)dz`$

Although each generated sample $`x`$ initially comes from $`p(x,z)`$ as described above, what really matters is that, over many samples, the marginalization over $`z`$ (i.e., integrating $`z`$ out) ensures that the generated samples follow $`p(x)`$.

The integral $`\int ...dz`$ makes sense, because it is like summarizing or averaging over all possible latent variables $`z`$. This process ensures that, overall, our model produces data samples $`x`$ from the target distribution $`p(x)`$.

## 2. How DDPM Learns

## 3. References
#### 3-1. Auto-Encoding Variational Bayes
[Kingma et al. (2013)](https://arxiv.org/abs/1312.6114)

#### 3-2. Denoising Diffusion Probabilistic Models
[Ho et al. (2020)](https://arxiv.org/abs/2006.11239)
