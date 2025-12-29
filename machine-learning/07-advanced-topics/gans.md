# Generative Adversarial Networks (GANs)

## Table of Contents

1. [Introduction to GANs](#introduction-to-gans)
2. [GAN Architecture](#gan-architecture)
3. [Training Dynamics](#training-dynamics)
4. [Loss Functions](#loss-functions)
5. [Common GAN Variants](#common-gan-variants)
6. [Conditional GANs](#conditional-gans)
7. [Advanced Architectures](#advanced-architectures)
8. [Training Challenges](#training-challenges)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Applications](#applications)

## Introduction to GANs

**Generative Adversarial Networks (GANs)** are a class of generative models that learn to create new data samples similar to training data through an adversarial process.

### Generative Models

**Goal**: Learn the underlying data distribution $p_{data}(x)$ to generate new samples.

**Types of generative models**:
- **Explicit density**: VAEs, autoregressive models, normalizing flows
- **Implicit density**: GANs (don't explicitly model $p_{data}$)

### The Adversarial Idea

GANs use a game-theoretic approach with two networks:

**Generator** ($G$): Creates fake samples
- Input: Random noise $z \sim p_z(z)$
- Output: Fake sample $G(z)$
- Goal: Fool the discriminator

**Discriminator** ($D$): Distinguishes real from fake
- Input: Sample $x$ (real or fake)
- Output: Probability that $x$ is real
- Goal: Correctly classify real and fake

**Analogy**: Counterfeiter (generator) vs detective (discriminator)

## GAN Architecture

### Basic Components

**Generator Network**:
$$G: \mathbb{R}^{d_z} \rightarrow \mathbb{R}^{d_x}$$

```python
# Example generator
def generator(z, hidden_dim=128):
    h1 = Dense(hidden_dim)(z)
    h1 = ReLU()(h1)
    h2 = Dense(hidden_dim)(h1)
    h2 = ReLU()(h2)
    x_fake = Dense(data_dim, activation='tanh')(h2)
    return x_fake
```

**Discriminator Network**:
$$D: \mathbb{R}^{d_x} \rightarrow [0, 1]$$

```python
# Example discriminator
def discriminator(x, hidden_dim=128):
    h1 = Dense(hidden_dim)(x)
    h1 = LeakyReLU()(h1)
    h2 = Dense(hidden_dim)(h1)
    h2 = LeakyReLU()(h2)
    prob = Dense(1, activation='sigmoid')(h2)
    return prob
```

### Information Flow

1. Sample noise $z \sim p_z(z)$ (e.g., $z \sim \mathcal{N}(0, I)$)
2. Generate fake sample: $x_{fake} = G(z)$
3. Sample real data: $x_{real} \sim p_{data}(x)$
4. Discriminator predicts: $D(x_{real})$ and $D(x_{fake})$
5. Update both networks based on discriminator's performance

## Training Dynamics

### Minimax Game

GAN training is formulated as a minimax game:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Interpretation**:
- **Discriminator** wants to maximize $V$: classify real as real (high $D(x)$), fake as fake (low $D(G(z))$)
- **Generator** wants to minimize $V$: fool discriminator (high $D(G(z))$)

### Training Algorithm

**For each training iteration**:

1. **Train Discriminator** (usually $k$ steps):
   - Sample minibatch of real data $\{x^{(1)}, ..., x^{(m)}\}$
   - Sample minibatch of noise $\{z^{(1)}, ..., z^{(m)}\}$
   - Generate fake samples $\{G(z^{(1)}), ..., G(z^{(m)})\}$
   - Update $D$ by ascending gradient:
   $$\nabla_{\theta_D} \frac{1}{m}\sum_{i=1}^m[\log D(x^{(i)}) + \log(1-D(G(z^{(i)})))]$$

2. **Train Generator** (usually 1 step):
   - Sample minibatch of noise $\{z^{(1)}, ..., z^{(m)}\}$
   - Update $G$ by descending gradient:
   $$\nabla_{\theta_G} \frac{1}{m}\sum_{i=1}^m \log(1-D(G(z^{(i)})))$$

### Nash Equilibrium

**Optimal discriminator** (for fixed $G$):

$$D^*_G(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

**Global optimum**: When $p_g = p_{data}$
- Discriminator cannot distinguish: $D^*(x) = 0.5$ everywhere
- Generator perfectly replicates data distribution

## Loss Functions

### Original GAN Loss

**Discriminator loss**:
$$L_D = -\mathbb{E}_{x \sim p_{data}}[\log D(x)] - \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Generator loss (minimax)**:
$$L_G = \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Problem**: Vanishing gradients when discriminator is strong.

### Non-Saturating Loss

**Alternative generator loss**:
$$L_G = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

**Benefits**:
- Stronger gradients early in training
- Same optimal solution
- Most commonly used in practice

### Wasserstein GAN (WGAN)

Uses Earth Mover's Distance (Wasserstein distance):

$$W(p_{data}, p_g) = \inf_{\gamma \in \Pi(p_{data}, p_g)} \mathbb{E}_{(x,y)\sim\gamma}[||x-y||]$$

**WGAN objective**:
$$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

**Key differences**:
- Discriminator (critic) doesn't use sigmoid
- Weight clipping to enforce Lipschitz constraint: $||w|| \leq c$

**Benefits**:
- More stable training
- Meaningful loss metric (correlates with quality)
- Less mode collapse

### WGAN-GP (Gradient Penalty)

Replaces weight clipping with gradient penalty:

$$L = \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))] - \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$$

where $\hat{x} = \epsilon x + (1-\epsilon)G(z)$ is a random interpolation.

**Benefits**:
- More stable than weight clipping
- Better gradient flow
- Higher quality samples

## Common GAN Variants

### Deep Convolutional GAN (DCGAN)

Architecture guidelines for stable training:

**Generator**:
- Use transposed convolutions for upsampling
- Batch normalization in all layers except output
- ReLU activations (except Tanh for output)
- No fully connected layers

**Discriminator**:
- Use strided convolutions for downsampling
- Batch normalization in all layers except input
- LeakyReLU activations
- No fully connected layers except output

```python
# DCGAN Generator example
def dcgan_generator(z, channels=3):
    x = Dense(4*4*1024)(z)
    x = Reshape((4, 4, 1024))(x)
    x = BatchNorm()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(512, 4, strides=2, padding='same')(x)  # 8x8
    x = BatchNorm()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)  # 16x16
    x = BatchNorm()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)  # 32x32
    x = BatchNorm()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(channels, 4, strides=2, padding='same')(x)  # 64x64
    x = Tanh()(x)
    return x
```

### Progressive GAN

Progressively grows generator and discriminator:
- Start with low resolution (4×4)
- Gradually add layers for higher resolution
- Smooth transition using fade-in

**Benefits**:
- More stable training
- Higher quality high-resolution images
- Faster convergence

### StyleGAN / StyleGAN2

**Key innovations**:

1. **Style-based generator**:
   - Mapping network: $z \rightarrow w$
   - Synthesis network controlled by $w$ at each resolution
   - Adaptive instance normalization (AdaIN)

2. **Disentangled latent space**:
   - Better control over generated features
   - Separate control of high-level attributes

3. **Stochastic variation**:
   - Per-pixel noise for fine-grained variation

**StyleGAN2 improvements**:
- Removes artifacts
- Improved normalization
- Path length regularization

## Conditional GANs

**Conditional GAN (cGAN)** - conditions generation on additional information $y$:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x|y)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|y)|y))]$$

### Conditioning Methods

**Label conditioning**:
- Concatenate class label to $z$ for generator
- Concatenate class label to $x$ for discriminator
- Example: Generate digit "5" from MNIST

**Image conditioning**:
- Use image as conditioning input
- Applications: Image-to-image translation

### Pix2Pix

Image-to-image translation with paired data:

**Generator**: U-Net architecture
- Encoder-decoder with skip connections
- Preserves spatial information

**Discriminator**: PatchGAN
- Classifies whether each patch is real or fake
- Focuses on high-frequency details

**Loss**:
$$L = L_{GAN} + \lambda L_{L1}$$

where $L_{L1} = \mathbb{E}_{x,y,z}[||y - G(x,z)||_1]$

**Applications**:
- Edges → photos
- Day → night
- Sketches → realistic images
- Semantic labels → street scenes

### CycleGAN

Unpaired image-to-image translation:

**Key idea**: Cycle consistency
- $G: X \rightarrow Y$ and $F: Y \rightarrow X$
- $F(G(x)) \approx x$ and $G(F(y)) \approx y$

**Loss**:
$$L = L_{GAN}(G) + L_{GAN}(F) + \lambda L_{cycle}$$

$$L_{cycle} = \mathbb{E}_x[||F(G(x)) - x||_1] + \mathbb{E}_y[||G(F(y)) - y||_1]$$

**Applications**:
- Horse ↔ zebra
- Summer ↔ winter
- Photo ↔ painting
- No paired data required!

## Advanced Architectures

### BigGAN

Scaling up GANs for ImageNet:

**Techniques**:
- Larger batch sizes (256-2048)
- Increased model capacity
- Truncation trick (sample $z$ from truncated normal)
- Orthogonal regularization
- Self-attention layers

**Results**: State-of-the-art image quality on ImageNet

### Self-Attention GAN (SAGAN)

Adds self-attention mechanism:

$$\text{Attention}(x) = \text{softmax}(f(x)^T g(x)) h(x)$$

**Benefits**:
- Better modeling of long-range dependencies
- Geometric or structural patterns
- Improved quality for complex scenes

### VQ-GAN

Combines VQVAE with adversarial training:
- Vector quantization for discrete latent codes
- GAN training for high-quality reconstruction
- Used in DALL-E 2 and similar models

## Training Challenges

### Mode Collapse

**Problem**: Generator produces limited variety of samples.

**Types**:
- **Complete collapse**: Single output
- **Partial collapse**: Missing modes from data distribution

**Solutions**:
- Minibatch discrimination (discriminator looks at batch statistics)
- Unrolled GANs (generator considers future discriminator updates)
- Feature matching (match statistics of real and fake)
- WGAN / WGAN-GP (more stable training)

### Vanishing Gradients

**Problem**: When discriminator is too strong, generator gets no learning signal.

**Solutions**:
- Non-saturating loss
- Wasserstein distance
- Careful learning rate tuning
- Less frequent discriminator updates

### Hyperparameter Sensitivity

**Problem**: Small changes in hyperparameters cause training failure.

**Important hyperparameters**:
- Learning rates (often different for G and D)
- Batch size
- Architecture choices
- Number of discriminator updates per generator update

**Solutions**:
- WGAN-GP (more robust)
- Spectral normalization
- Progressive growing
- Extensive tuning

### Training Instability

**Symptoms**:
- Oscillating losses
- Sudden quality drops
- Failure to converge

**Solutions**:
- Two time-scale update rule (TTUR)
- Gradient clipping
- Regularization techniques
- Monitoring multiple metrics

## Evaluation Metrics

### Inception Score (IS)

Measures quality and diversity:

$$IS = \exp(\mathbb{E}_x[KL(p(y|x) || p(y))])$$

**Properties**:
- High score: Sharp predictions, diverse samples
- Limitations: Doesn't compare to real data, can be gamed

### Fréchet Inception Distance (FID)

Compares statistics of real and generated samples in feature space:

$$FID = ||\mu_{real} - \mu_{gen}||^2 + \text{Tr}(\Sigma_{real} + \Sigma_{gen} - 2(\Sigma_{real}\Sigma_{gen})^{1/2})$$

**Properties**:
- Lower is better
- More robust than IS
- Correlates well with human judgment
- Most widely used metric

### Precision and Recall

**Precision**: What fraction of generated samples are realistic?

**Recall**: What fraction of real data modes are covered?

**Trade-off**:
- High precision, low recall: Few high-quality samples (mode collapse)
- Low precision, high recall: Many low-quality samples

### Human Evaluation

**A/B testing**: Humans choose preferred sample

**Realism rating**: Rate how realistic samples are

**Still gold standard** for assessing quality

## Applications

### Image Synthesis

**Face generation**: StyleGAN produces photorealistic faces

**Scene generation**: BigGAN for complex natural scenes

**Art generation**: Create artistic images in various styles

### Image-to-Image Translation

**Super-resolution**: Upscale low-resolution images

**Colorization**: Convert grayscale to color

**Style transfer**: Apply artistic styles to photos

**Semantic synthesis**: Labels → realistic images

### Data Augmentation

Generate synthetic training data:
- Medical imaging (limited data)
- Rare event simulation
- Privacy-preserving synthetic datasets

### Text-to-Image

**DALL-E**: Generate images from text descriptions

**Stable Diffusion**: Open-source text-to-image

**Imagen, Parti**: State-of-the-art text-to-image models

### Video Generation

**Temporal coherence**: Maintain consistency across frames

**Video prediction**: Predict future frames

**Video-to-video translation**: Style transfer for videos

### 3D Generation

**3D-GAN**: Generate 3D objects

**NeRF + GAN**: Neural radiance fields with adversarial training

### Molecular Design

Generate novel molecular structures for drug discovery

## Practical Considerations

### Implementation Tips

**Learning rates**:
- Often use different rates for G and D
- Typical: $lr_D = 4 \times 10^{-4}$, $lr_G = 1 \times 10^{-4}$

**Optimizers**:
- Adam with $\beta_1 = 0.5, \beta_2 = 0.999$ (DCGAN)
- Adam with $\beta_1 = 0, \beta_2 = 0.999$ (BigGAN)

**Discriminator updates**:
- Often update D more frequently than G (e.g., 5:1 ratio)
- WGAN-GP typically uses 5 D updates per G update

**Batch size**:
- Larger is often better (if memory allows)
- Improves stability

### Monitoring Training

**Track metrics**:
- Discriminator and generator losses
- FID score (computed periodically)
- Sample quality (visual inspection)
- Gradient norms

**Warning signs**:
- D loss → 0, G loss → infinity (mode collapse)
- D loss → infinity, G loss → 0 (discriminator failure)
- Stable losses but poor quality (need hyperparameter tuning)

### Example Training Loop

```python
for epoch in range(num_epochs):
    for real_images in dataloader:
        # Train Discriminator
        for _ in range(n_critic):
            noise = torch.randn(batch_size, latent_dim)
            fake_images = generator(noise)

            real_validity = discriminator(real_images)
            fake_validity = discriminator(fake_images.detach())

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

            discriminator.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Train Generator
        noise = torch.randn(batch_size, latent_dim)
        fake_images = generator(noise)
        fake_validity = discriminator(fake_images)

        g_loss = -torch.mean(fake_validity)

        generator.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # Periodic evaluation
    if epoch % eval_interval == 0:
        compute_fid_score(generator)
        save_samples(generator)
```

## Best Practices

1. **Start simple**: DCGAN architecture as baseline
2. **Use proven techniques**: Batch norm, LeakyReLU, etc.
3. **Monitor carefully**: Visual inspection + metrics
4. **Be patient**: GANs can take time to converge
5. **Use pretrained models**: When possible (transfer learning)
6. **Regularize appropriately**: Spectral norm, gradient penalty
7. **Experiment systematically**: Change one thing at a time

## References

- "Generative Adversarial Networks" (Goodfellow et al., 2014) - Original GAN paper
- "Unsupervised Representation Learning with Deep Convolutional GANs" (Radford et al., 2015) - DCGAN
- "Wasserstein GAN" (Arjovsky et al., 2017) - WGAN
- "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017) - WGAN-GP
- "Progressive Growing of GANs" (Karras et al., 2017)
- "A Style-Based Generator Architecture for GANs" (Karras et al., 2019) - StyleGAN
- "Image-to-Image Translation with Conditional Adversarial Networks" (Isola et al., 2017) - Pix2Pix
- "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (Zhu et al., 2017) - CycleGAN
- GANs in Action (Langr & Bok)
