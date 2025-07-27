# How VAE Learns the Target Distribution
Generative models are defined by their ability to learn the distribution of the training data $p(x)$, and VAE is one example of such a model. But how does a VAE learn this? Let's find out together. The main purpose of this article is to explore DDPM, but we'll start with VAE first, since they're simpler, and then move on to DDPM later.

You may ask: Why bother writing another blog about VAE when there are already tons of them out there? The reason is that most of those blogs or papers immediately dive into math-heavy explanations, flooding the reader with equations about ELBO, KL Divergence, Gaussians, and more - which can be difficult for newcomers to follow.

I truly respect those authors and thank them for sharing their knowledge with the world. However, I've never felt fully satisfied when reading such articles. I don't just want to comprehend the math, I want to see the whole picture. I don't just want to grasp what the authors think, I also want to figure out how they came up with their solutions. Understanding what others have done makes us followers, but understanding the general principles of solving a problem can someday make us leaders in the field.

Let's define the problem in clear and simple language, and then gradually work toward explaining how a VAE can learn $p(x)$, using intuitive reasoning and as little math as possible. While breaking the explanation into multiple steps, I will try to connect all the crucial terms, because the ability to connect everything together is key to achieving a general view. If you just want to see the short version, you can jump straight to [Section "2. Conclusion"](#2-conclusion), but if you are curious like me and want to gain a deeper understanding, come with me.

## 1. Understanding VAE Learning
### 1-1. Learning the Target Distribution
Suppose we have a *target distribution* $p(x)$, which is unknown. We will never have direct access to its form or equation, but we can sample from it. We also have a model $p_\theta(x)$, which could be a neural network or any other parameterized function with parameter $\theta$.

> [!TIP]
> The goal of learning $p(x)$ is to adjust the parameter $\theta$ during training so that $p_\theta(x_i)$ closely approximates $p(x_i)$ for all observed sample $x_i$.

Whenever someone says, "we are training a generative model", they actually mean, "we are training a model that can learn the target distribution", or more precisely, a model that can produce samples whose distribution is close to that of samples they observed from the target distribution.

### 1-2. Maximum Likelihood Estimation (MLE)
What can we do to enable a model to learn $p(x)$? Generally speaking, we achieve this through **Maximum Likelihood Estimation (MLE)**.

The likelihood function (also called *marginal likelihood* or *evidence*) is defined as the probability of observing a data point $x_i$ given parameter $\theta$. Training a model with MLE means finding the optimal $\hat{\theta}$ so that the model assigns the highest possible value to $p_{\hat{\theta}}(x_i)$ for each $x_i$.

$$
\hat{\theta}=\arg\max_{\theta}\prod_{i}p_\theta(x_i)
$$

In practice, it is encouraged to rewrite this formula using the log-likelihood function because it is easier to compute.

$$
\hat{\theta}=\arg\max_{\theta}\log\left(\prod_{i}p_\theta(x_i)\right)=\arg\max_{\theta}\sum_{i}\log p_\theta(x_i)
$$

But why do we want to maximize the log-likelihood? How does it help us achieve the goal of learning $p(x)$? We can explain this in simple language: we have observed several samples from $p(x)$, named $x_1$, $x_2$, $x_3$,... Maximizing the log-likelihood is like saying: "Here are samples we observed. There may be many other samples we haven't seen yet, or the target distribution of these samples might differ from what we observed. However, our best guess for a distribution close to $p(x)$ is to assume that what we observe has the highest probability". We don't worry about what hasn't happened yet; we only care about what we see. If something happens many times, it must have a high probability, so we simply maximize that.

Still not convinced yet? Let's take a simple example. Suppose we have a coin that follows a target distribution $p(x)$, where $x$ can take only two values: $H$ for heads and $T$ for tails. After tossing the coin 6 times, we observe the following results: $H-T-H-T-H-H$. So, our problem of learning $p(x)$ boils down to estimating a parameter $\theta$ using MLE:

$$
\hat{\theta}=\arg\max_{\theta}\left(\log p_\theta(H)+\log p_\theta(T)+\log p_\theta(H)+\log p_\theta(T)+\log p_\theta(H)+\log p_\theta(H)\right)
$$

We don't know the true $p(x)$, and we don't care about what we haven't seen. We only know two things: the probabilities of each event, i.e., $p_\theta(H)$ and $p_\theta(T)$, must sum to 1 (which is obvious since $p$ is a probability function), and the observations we have above. In this case, the coin seems unfair: we got 4 heads and 2 tails, so the model estimates the probabilities as $p_\theta(H)=\frac{4}{6}=\frac{2}{3}$ and $p_\theta(T)=\frac{2}{6}=\frac{1}{3}$. Is this the true answer? What if the target distribution is totally different: $p(H)=\frac{1}{10}$ and $p(T)=\frac{9}{10}$? Unfortunately, we will never know for sure. But we can improve the accuracy of our model by doing a (hopefully) very simple thing: collecting more data. This is the essence of [Law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers): the more samples we observe from the target distribution, the better chance we have of learning it.

> [!TIP]
> In a generative model, $p_\theta(x)$ is learned with MLE by maximizing $\log p_\theta(x_i)$ for every $x_i$.

But wait a minute. We just dropped $\prod_{i}$! Why can we do that? You may have seen in many papers about generative models, including this article, that MLE is often expressed not in its full form over all observed samples but only for a single example. It turns out this is totally fine: maximizing $\log p_\theta(x_i)$ for a single sample $x_i$ doesn't violate the full form, because we don't do it for just one sample, we do it for all samples and let them "compete" with each other. This means we still maximize the sum of the log-likelihoods over all samples, not just one. By doing this, the values that appear more frequently (i.e., the same $x_i$ repeated in different observations) will gradually dominate the distribution. For example, in the coin toss case above, we perform MLE over all $x_i$, but since $H$ happens many times, the model will prioritize $H$ over $T$, leading to $p_\theta(H)$ eventually becoming larger than $p_\theta(T)$.

### 1-3. Latent Variable
Ok, then how do we train $p_\theta(x)$ with MLE? By introducing a *latent variable* $z$. The detailed reasons for leveraging $z$ when training a VAE are beyond the scope of this article, but we can understand that this variable serves as a hidden, compressed representation of the data. Most importantly, it should be *tractable* so that we can control it.

$$
p_\theta(x)=\int p_\theta(x,z)dz=\int p_\theta(x\mid z)p(z)dz
$$

What we are doing here is *marginalizing out* the latent variable $z$, and this is also how we generate samples from $p_\theta(x)$. At first, this integral may look scary, but it isn't. Let's break down the role of each term:
- $p(z)$: This represents drawing a sample $z$ that follows the probability distribution of the latent variable $p(z)$, which is also called the *prior distribution*. It can be chosen as a simple distribution (such as a Gaussian) and does not need to be learned (hence, it does not include any parameters), so sampling it is straightforward.
- $p_\theta(x\mid z)$: After obtaining a sample $z$, we treat it as a fixed value and draw a sample $x$ from $p_\theta(x\mid z)$. This is a conditional distribution that gives the probability of each value of $x$ given a fixed input $z$, and needs to be learned during training.

> [!TIP]
> We learn $p_\theta(x)$ indirectly by learning a *likelihood distribution* $p_\theta(x\mid z)$, which is called the *decoder*.

But sampling $z$ from $p(z)$ and then $x$ from $p_\theta(x\mid z)$ only gives us the joint distribution $p_\theta(x\mid z)p(z)=p_\theta(x,z)$, whereas what we actually need is $p_\theta(x)$. So, what are we missing here? This is where the remaining term comes in: the integral $\int...dz$ tells us that $p_\theta(x)$ is calculated by summing (or *marginalizing*) $p_\theta(x,z)$ over all possible values of $z$. This makes sense because when we draw a sample $z$ from $p(z)$, it never comes alone, but implicitly represents an infinitesimal range $dz$ that $z$ belongs to.

*Marginalizing out* every $z$ is impossible, isn't it? Actually, it's not about approximating the integral directly at generation, but rather that the marginal likelihood involves the integral. Generating only a single sample or even a combination of a few samples of $x$, won't statistically represent the distribution well. For example, the samples may cluster in a region far from the true mean of $p(x)$ and may seem completely different from $p(x)$. But that's ok. The point is: we are dealing with distributions, so a single or a few samples tell us nothing about whether $p_\theta(x)$ has truly learned $p(x)$ or not. What really matters is that if we draw a sufficiently large number of samples, the distribution represented by $p_\theta(x)$ should closely approximate $p(x)$. This is the role of the integral $\int$, to emphasize that the more samples we generate from $p_\theta(x,z)$, the closer $p_\theta(x)$ becomes to $p(x)$.

### 1-4. Evidence lower bound (ELBO)
So far, we know that in order to learn $p(x)$, we need to apply MLE to the decoding process $p_\theta(x)=\int p_\theta(x\mid z)p(z)dz$. In other words, we want to find the optimal $\hat{\theta}$ that maximizes that integral. Our problem is that this integral is not easy to evaluate, or as the authors of VAE put it, intractable. We therefore need to find another way to express $p_\theta(x)$ in tractable terms, while also ensuring it includes $p_\theta(x\mid z)$ and $p(z)$ - the terms required when generating samples from $p_\theta(x)$ during decoding.

The solution the authors of VAE arrived at is to use **Evidence lower bound (ELBO)**.
<details><summary>Disclaimer</summary>

I'm trying to connect the use of ELBO with the terms required during decoding. In many VAE articles, authors simply show ELBO and the decoding process without explaining why both include $p_\theta(x\mid z)$ and $p(z)$. This can't be a coincidence. I assume there may be alternative ways to write ELBO or other methods, but the authors chose this particular form so it shares common terms with decoding, connecting them together.
</details>

```math
\log p_\theta(x)\ge\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)
```

Great! Our new expression includes both $p_\theta(x\mid z)$ and $p(z)$, as expected. But we also see a new term introduced: $q_\phi(z\mid x)$. What is happening here? Let's break things down and take a closer look. They are so elegant and all make sense:
- The LHS is our model expressed in terms of log-likelihood. As discussed above, we want to find a way to maximize this term for every sample $x_i$.
- The RHS is ELBO. A detailed derivation of ELBO is beyond this article's scope, but all you need to remember is: ELBO stands for "Evidence Lower Bound", meaning that by maximizing ELBO, we indirectly maximize the LHS and achieve our goal. ELBO consists of two different terms with distinct roles, but they are all connected to each other and share key terms with the decoding process:
    - *Reconstruction term* $`\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]`$

        This term measures the likelihood of the decoder $p_\theta(x\mid z)$, given a latent variable $z$ sampled from the new term $q_\phi(z\mid x)$, which we will refer to as the *encoder* (with a parameter $\phi$) from now on. Please keep in mind: during training, the decoder $p_\theta(x\mid z)$ takes input $z$ sampled from the encoder $q_\phi(z\mid x)$, not from the prior distribution $p(z)$. And the reason we introduce $q_\phi(z\mid x)$ here is to approximate the true posterior $p_\theta(z\mid x)$, which is intractable.

        Maximizing this term contributes to the maximization of ELBO, as it ensures that the learned decoder is able to generate data with high likelihood.
    - *Prior matching term* $`D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)`$

        This term measures how closely the encoder $q_\phi(z\mid x)$ matches the prior distribution $p(z)$.

        Minimizing this term (note the minus sign $-$) also contributes to the maximization of ELBO, as it encourages the learned encoder to become similar to the predefined prior distribution, ensuring smoothness and enabling sampling. This makes sense because it allows us, during decoding, to replace the encoder with the prior distribution as a source distribution for input $z$ to the decoder, which aligns with how we defined the decoding process above.

        ($D_{\mathrm{KL}}$ is **KL Divergence**, which has a low value when its two input distributions are similar and a high value when they differ significantly).

> [!TIP]
> VAE doesn't apply MLE directly to $p_\theta(x)$, but indirectly by maximizing ELBO, which involves two terms: reconstruction term for training the decoder $p_\theta(x\mid z)$, and prior matching term for training the encoder $q_\phi(z\mid x)$.

## 2. Conclusion
WIP

## 3. References
#### 3-1. An Introduction to Variational Autoencoders
[Kingma et al. (2019)](https://arxiv.org/abs/1906.02691)

#### 3-2. Understanding Diffusion Models: A Unified Perspective
[Luo et al. (2022)](https://arxiv.org/abs/2208.11970)

#### 3-3. Tutorial on Diffusion Models for Imaging and Vision
[Chan et al. (2024)](https://arxiv.org/abs/2403.18103)
