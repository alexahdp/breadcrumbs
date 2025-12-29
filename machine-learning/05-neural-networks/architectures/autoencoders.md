# Autoencoders

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Autoencoder](#basic-autoencoder)
3. [Architecture Variants](#architecture-variants)
4. [Denoising Autoencoders](#denoising-autoencoders)
5. [Sparse Autoencoders](#sparse-autoencoders)
6. [Variational Autoencoders (VAE)](#variational-autoencoders-vae)
7. [Convolutional Autoencoders](#convolutional-autoencoders)
8. [Applications](#applications)
9. [Practical Considerations](#practical-considerations)

## Introduction

**Autoencoders (AE)** are a type of neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct the original data from this representation (decoding).

**Key characteristics:**
- Unsupervised learning (no labels required)
- Learn compressed representations of data
- Encoder-decoder architecture
- Reconstruction-based training objective

**Main components:**
1. **Encoder** - compresses input to latent representation
2. **Latent space (bottleneck)** - compressed representation
3. **Decoder** - reconstructs input from latent representation

**Applications:**
- Dimensionality reduction
- Feature learning
- Data denoising
- Anomaly detection
- Image generation
- Data compression

## Basic Autoencoder

### Architecture

```
Input → Encoder → Latent Code (Bottleneck) → Decoder → Output
  x   →    f    →        z                 →    g    →   x̂
```

**Goal:** Learn $f$ and $g$ such that $x \approx g(f(x))$

### Mathematical Formulation

**Encoder:**

$$\mathbf{z} = f(\mathbf{x}) = \sigma(\mathbf{W}_e\mathbf{x} + \mathbf{b}_e)$$

**Decoder:**

$$\hat{\mathbf{x}} = g(\mathbf{z}) = \sigma(\mathbf{W}_d\mathbf{z} + \mathbf{b}_d)$$

where:
- $\mathbf{x} \in \mathbb{R}^n$ - input
- $\mathbf{z} \in \mathbb{R}^m$ - latent code (typically $m < n$)
- $\hat{\mathbf{x}} \in \mathbb{R}^n$ - reconstruction
- $\sigma$ - activation function

### Loss Function

**Reconstruction loss** (Mean Squared Error):

$$L(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2 = ||\mathbf{x} - \hat{\mathbf{x}}||^2$$

**For binary data** (e.g., black/white images), use Binary Cross-Entropy:

$$L(\mathbf{x}, \hat{\mathbf{x}}) = -\sum_{i=1}^{n}[x_i\log(\hat{x}_i) + (1-x_i)\log(1-\hat{x}_i)]$$

### Training Objective

Minimize the reconstruction error over the dataset:

$$\min_{\theta_e, \theta_d} \frac{1}{N}\sum_{i=1}^{N} L(\mathbf{x}^{(i)}, \hat{\mathbf{x}}^{(i)})$$

where:
- $\theta_e$ - encoder parameters
- $\theta_d$ - decoder parameters
- $N$ - number of training examples

### Example Implementation

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # For normalized data [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# Usage
model = Autoencoder(input_dim=784, encoding_dim=32)  # MNIST: 784 → 32 → 784
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data  # Ignore labels
        img = img.view(img.size(0), -1)  # Flatten

        # Forward pass
        output = model(img)
        loss = criterion(output, img)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Architecture Variants

### Undercomplete Autoencoders

**Bottleneck dimension < Input dimension** ($m < n$)

Forces the network to learn a **compressed representation** of the data.

**Example:** $784 \rightarrow 128 \rightarrow 32 \rightarrow 128 \rightarrow 784$

**Advantage:** Learns meaningful features through dimensionality reduction

### Overcomplete Autoencoders

**Bottleneck dimension ≥ Input dimension** ($m \geq n$)

Without regularization, the network can learn an identity mapping (copying inputs to outputs).

**Solution:** Add regularization:
- Sparsity constraints
- Denoising
- Contractive penalty

### Stacked Autoencoders

**Deep autoencoders** with multiple encoding/decoding layers:

```
Input → E1 → E2 → E3 → Latent → D3 → D2 → D1 → Output
```

**Training strategies:**
1. **Greedy layer-wise pre-training** - train each layer separately
2. **End-to-end training** - train entire network jointly

**Benefits:**
- Learn hierarchical features
- Better representations for complex data

## Denoising Autoencoders (DAE)

**Goal:** Learn robust representations by reconstructing clean data from corrupted input.

### Corruption Process

Add noise to the input:

$$\tilde{\mathbf{x}} = \mathbf{x} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

Or randomly set some inputs to zero (dropout noise):

$$\tilde{x}_i = \begin{cases} 0 & \text{with probability } p \\ x_i & \text{with probability } 1-p \end{cases}$$

### Training Objective

Learn to reconstruct the **original** (clean) data from **corrupted** input:

$$\min_{\theta} \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}[L(\mathbf{x}, f(\tilde{\mathbf{x}}))]$$

### Benefits

- **Robustness** - learns features invariant to noise
- **Better generalization** - prevents memorization
- **Feature extraction** - learns meaningful patterns

### Implementation

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def forward(self, x):
        # Add noise during training only
        if self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x

        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded

# Training
model = DenoisingAutoencoder(784, 32, noise_factor=0.5)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    for img, _ in dataloader:
        img = img.view(img.size(0), -1)

        # Reconstruct clean image from noisy input
        output = model(img)
        loss = criterion(output, img)  # Compare with original

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Sparse Autoencoders

**Goal:** Learn sparse representations where only a few neurons are active at a time.

### Sparsity Constraint

Add a penalty to encourage sparse activations:

$$L_{total} = L_{reconstruction} + \lambda \cdot L_{sparsity}$$

### KL Divergence Sparsity

Encourage average activation $\hat{\rho}_j$ of neuron $j$ to be close to target sparsity $\rho$ (e.g., 0.05):

$$L_{sparsity} = \sum_{j=1}^{m} KL(\rho || \hat{\rho}_j)$$

$$KL(\rho || \hat{\rho}_j) = \rho\log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}$$

where:

$$\hat{\rho}_j = \frac{1}{N}\sum_{i=1}^{N}z_j^{(i)}$$

### L1 Regularization

Simpler approach - add L1 penalty on activations:

$$L_{sparsity} = \sum_{j=1}^{m}|z_j|$$

### Benefits

- **Interpretability** - each neuron learns specific features
- **Disentangled representations** - features are independent
- **Better generalization** - prevents overfitting

### Implementation

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, sparsity_target=0.05):
        super(SparseAutoencoder, self).__init__()
        self.sparsity_target = sparsity_target

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Sigmoid()  # Important for KL divergence
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.encoded = self.encoder(x)
        decoded = self.decoder(self.encoded)
        return decoded

    def kl_divergence_loss(self):
        rho = self.sparsity_target
        rho_hat = torch.mean(self.encoded, dim=0)

        kl = rho * torch.log(rho / rho_hat) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

        return torch.sum(kl)

# Training with sparsity
model = SparseAutoencoder(784, 128, sparsity_target=0.05)
beta = 0.01  # Sparsity weight

for img, _ in dataloader:
    img = img.view(img.size(0), -1)

    output = model(img)

    # Combined loss
    reconstruction_loss = nn.MSELoss()(output, img)
    sparsity_loss = model.kl_divergence_loss()
    loss = reconstruction_loss + beta * sparsity_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Variational Autoencoders (VAE)

**VAE** is a generative model that learns a probabilistic mapping between data and latent space.

**Key difference from AE:** Instead of encoding to a fixed vector, VAE encodes to a **probability distribution**.

### Probabilistic Formulation

**Encoder** outputs parameters of a distribution:

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}(\mathbf{x}), \boldsymbol{\sigma}^2(\mathbf{x})\mathbf{I})$$

where $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are outputs of the encoder network.

**Decoder** reconstructs from sampled $\mathbf{z}$:

$$p_\theta(\mathbf{x}|\mathbf{z})$$

### Reparameterization Trick

To enable backpropagation through stochastic sampling:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

This separates randomness ($\boldsymbol{\epsilon}$) from parameters ($\boldsymbol{\mu}, \boldsymbol{\sigma}$).

### Loss Function

**VAE loss = Reconstruction loss + KL divergence**

$$L_{VAE} = L_{reconstruction} + \beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

**Reconstruction loss:**

$$L_{reconstruction} = -\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]$$

In practice, use MSE or BCE.

**KL divergence** (closed form for Gaussian prior):

$$D_{KL} = -\frac{1}{2}\sum_{j=1}^{m}(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$$

### Implementation

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Training
model = VAE(input_dim=784, latent_dim=20)

for img, _ in dataloader:
    img = img.view(img.size(0), -1)

    # Forward pass
    recon_img, mu, logvar = model(img)

    # Compute loss
    loss = vae_loss(recon_img, img, mu, logvar)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate new samples
with torch.no_grad():
    z = torch.randn(64, 20)  # Sample from prior
    samples = model.decode(z)
```

### Benefits of VAE

1. **Generative model** - can generate new samples
2. **Smooth latent space** - interpolation between points produces meaningful outputs
3. **Probabilistic framework** - principled approach to uncertainty
4. **Disentangled representations** - with β-VAE variant

### β-VAE

Variant that encourages disentangled representations:

$$L_{\beta\text{-}VAE} = L_{reconstruction} + \beta \cdot D_{KL}$$

where $\beta > 1$ (typically 4-10) increases the weight of the KL term.

## Convolutional Autoencoders

For image data, use **convolutional layers** in encoder and **transposed convolutions** (or upsampling) in decoder.

### Architecture

```
Encoder: Conv → Pool → Conv → Pool → ...
Latent: Flattened representation
Decoder: FC → Reshape → ConvTranspose → Upsample → ...
```

### Implementation

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 → 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)                       # 7x7 → 1x1
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),                      # 1x1 → 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 14x14 → 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# For images: input shape (batch, channels, height, width)
model = ConvAutoencoder()
x = torch.randn(32, 1, 28, 28)  # MNIST batch
output = model(x)
```

## Applications

### 1. Dimensionality Reduction

Alternative to PCA:
- Non-linear transformations
- Can capture complex patterns
- Better for visualization

```python
# Use encoder for dimensionality reduction
encoder = model.encoder
reduced_data = encoder(data)  # High-dim → Low-dim
```

### 2. Anomaly Detection

Normal data reconstructs well, anomalies reconstruct poorly:

$$\text{Anomaly Score} = ||x - \hat{x}||^2$$

```python
# Detect anomalies
reconstruction = model(x)
reconstruction_error = torch.mean((x - reconstruction) ** 2, dim=1)
anomalies = reconstruction_error > threshold
```

### 3. Image Denoising

Train denoising autoencoder, then use for cleaning images:

```python
denoised_image = model(noisy_image)
```

### 4. Data Generation (VAE)

Generate new samples by sampling from latent space:

```python
# Sample from prior
z = torch.randn(num_samples, latent_dim)
generated_images = vae.decode(z)
```

### 5. Feature Learning

Use learned representations for downstream tasks:

```python
# Use encoder as feature extractor
features = encoder(data)
# Train classifier on features
classifier = train_classifier(features, labels)
```

### 6. Image Compression

Compress images using the latent representation:

```python
# Compress
compressed = encoder(image)  # Much smaller
# Decompress
reconstructed = decoder(compressed)
```

## Practical Considerations

### Choosing Architecture

**Latent dimension:**
- Too small: underfitting, poor reconstruction
- Too large: overfitting, identity mapping
- Start with 10-20% of input dimension
- Experiment and validate

**Network depth:**
- Deeper networks learn better features
- But harder to train
- Use batch normalization and skip connections for deep networks

### Loss Function Selection

**For continuous data (images):**
- MSE (L2) - blurry reconstructions
- MAE (L1) - sharper but noisier
- Perceptual loss - better visual quality

**For binary data:**
- Binary cross-entropy

**For multimodal data:**
- Combination of losses for different modalities

### Training Tips

**Initialization:**
- Use He/Xavier initialization
- Pre-train layer-by-layer for very deep networks

**Learning rate:**
- Start with 1e-3 (Adam)
- Use learning rate scheduling

**Regularization:**
- Dropout in encoder/decoder
- Weight decay (L2 regularization)
- Early stopping

**Batch normalization:**
- After conv/linear layers, before activation
- Stabilizes training
- Allows higher learning rates

### Hyperparameter Tuning

**For VAE:**
- **β weight:** Balance reconstruction vs KL
  - β too high: blurry reconstructions
  - β too low: posterior collapse (KL → 0)
- **Latent dimension:** 10-100 depending on data complexity

**For Sparse AE:**
- **Sparsity target (ρ):** 0.01-0.1
- **Sparsity weight (β):** 0.01-1.0

**For Denoising AE:**
- **Noise level:** 0.1-0.5 for Gaussian noise
- **Dropout rate:** 0.1-0.5 for dropout noise

### Common Issues

**Poor reconstruction:**
- Bottleneck too small - increase latent dimension
- Network too shallow - add more layers
- Learning rate too high - reduce learning rate

**Overfitting:**
- Add regularization (dropout, weight decay)
- Use denoising autoencoder
- Reduce model capacity
- Get more training data

**VAE posterior collapse:**
- KL term becomes zero, decoder ignores latent code
- Solutions:
  - Reduce β (KL weight)
  - Use KL annealing (start β=0, gradually increase)
  - Use skip connections in decoder

## References

- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science.
- Vincent, P., et al. (2008). Extracting and composing robust features with denoising autoencoders.
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. ICLR.
- Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework.
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) (Carl Doersch)
