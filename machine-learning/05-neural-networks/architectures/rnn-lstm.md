# Recurrent Neural Networks and LSTMs

## Table of Contents

1. [Introduction](#introduction)
2. [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
3. [Training RNNs](#training-rnns)
4. [Problems with Vanilla RNNs](#problems-with-vanilla-rnns)
5. [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
6. [Gated Recurrent Unit (GRU)](#gated-recurrent-unit-gru)
7. [Bidirectional RNNs](#bidirectional-rnns)
8. [Common Architectures](#common-architectures)
9. [Practical Considerations](#practical-considerations)

## Introduction

**Recurrent Neural Networks (RNNs)** are a class of neural networks designed to process **sequential data** by maintaining an internal state (memory) that captures information about previous inputs in the sequence.

**Key characteristics:**
- Process sequences of variable length
- Share parameters across time steps
- Maintain hidden state as memory
- Can theoretically capture long-term dependencies

**Applications:**
- Natural Language Processing (NLP)
- Time series prediction
- Speech recognition
- Music generation
- Video analysis
- Machine translation

## Recurrent Neural Networks (RNN)

### Architecture

Unlike feedforward networks, RNNs have **recurrent connections** that allow information to persist.

**Unrolled RNN through time:**

```
x₀ → [RNN] → h₀ → y₀
      ↓ ↑
x₁ → [RNN] → h₁ → y₁
      ↓ ↑
x₂ → [RNN] → h₂ → y₂
```

The same network cell processes each time step, with hidden state passing forward.

### Mathematical Formulation

At each time step $t$:

$$\mathbf{h}_t = f(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = g(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y)$$

where:
- $\mathbf{x}_t \in \mathbb{R}^d$ - input at time $t$
- $\mathbf{h}_t \in \mathbb{R}^h$ - hidden state at time $t$
- $\mathbf{y}_t \in \mathbb{R}^k$ - output at time $t$
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$ - input-to-hidden weights
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ - hidden-to-hidden (recurrent) weights
- $\mathbf{W}_{hy} \in \mathbb{R}^{k \times h}$ - hidden-to-output weights
- $f$ - activation function (usually $\tanh$ or ReLU)
- $g$ - output activation (softmax for classification, linear for regression)

### Initial Hidden State

The hidden state $\mathbf{h}_0$ is typically initialized to zeros:

$$\mathbf{h}_0 = \mathbf{0}$$

Or learned as a parameter during training.

### Forward Pass

For a sequence of length $T$:

```
for t = 1 to T:
    hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
    yₜ = Wₕᵧhₜ + bᵧ
```

### Parameter Sharing

**Key insight:** The same weight matrices $(\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy})$ are used at every time step.

**Advantages:**
- Can process sequences of any length
- Model size doesn't grow with sequence length
- Same features computed at each time step

### Example Implementation

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        # Weight matrices
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, hidden):
        # x: (batch_size, input_size)
        # hidden: (batch_size, hidden_size)

        combined = torch.cat((x, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

# Using PyTorch's built-in RNN
rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)

# Input: (batch_size, sequence_length, input_size)
x = torch.randn(32, 100, 10)  # 32 sequences of length 100
output, h_n = rnn(x)
```

## Training RNNs

### Backpropagation Through Time (BPTT)

**BPTT** is the training algorithm for RNNs - essentially backpropagation applied to the unrolled network.

**Process:**
1. Unroll the RNN for $T$ time steps
2. Forward pass: compute outputs for all time steps
3. Compute loss across all time steps
4. Backward pass: compute gradients from time $T$ back to time 1
5. Update weights

### Loss Computation

**For sequence prediction:**

$$L = \frac{1}{T}\sum_{t=1}^{T} L_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$

**For sequence classification** (only final output matters):

$$L = L_T(\mathbf{y}_T, \hat{\mathbf{y}}_T)$$

### Truncated BPTT

For very long sequences, full BPTT is computationally expensive.

**Solution:** Truncate backpropagation after $k$ steps ($k < T$):
- Forward pass processes entire sequence
- Backward pass only goes back $k$ steps
- Common values: $k = 20$ to $50$

## Problems with Vanilla RNNs

### Vanishing Gradients

During backpropagation through time, gradients are multiplied by $\mathbf{W}_{hh}$ at each time step:

$$\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$

If eigenvalues of $\mathbf{W}_{hh}$ are less than 1, gradients **vanish exponentially** with sequence length.

**Consequences:**
- Cannot learn long-term dependencies (>10 steps)
- Early layers learn very slowly
- Network "forgets" information from distant past

### Exploding Gradients

If eigenvalues of $\mathbf{W}_{hh}$ are greater than 1, gradients **explode exponentially**.

**Solution:** Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Limited Memory

Vanilla RNNs struggle to remember information for more than 10-20 time steps, making them unsuitable for many real-world tasks.

## Long Short-Term Memory (LSTM)

**LSTMs** were designed to solve the vanishing gradient problem and enable learning of long-term dependencies.

**Key innovation:** Replace simple hidden state with a **cell state** and **gates** that control information flow.

### Architecture

LSTM has:
- **Cell state** $\mathbf{c}_t$ - long-term memory
- **Hidden state** $\mathbf{h}_t$ - short-term memory
- **Three gates:**
  - **Forget gate** $\mathbf{f}_t$ - what to forget from cell state
  - **Input gate** $\mathbf{i}_t$ - what new information to add
  - **Output gate** $\mathbf{o}_t$ - what to output

### Mathematical Formulation

At each time step $t$:

**Forget gate** (what to forget):

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input gate** (what to add):

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate values** (new information):

$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Update cell state**:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output gate** (what to output):

$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden state**:

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where:
- $\sigma$ - sigmoid function (outputs 0 to 1)
- $\odot$ - element-wise multiplication
- $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ - concatenation

### How LSTM Solves Vanishing Gradients

**Cell state gradient flow:**

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \mathbf{f}_t$$

Gradients flow through **addition** (not multiplication), preventing vanishing.

**Uninterrupted gradient highway:** The cell state acts as a "conveyor belt" allowing gradients to flow unchanged.

### LSTM Parameters

For an LSTM layer with input size $d$ and hidden size $h$:

$$\text{Parameters} = 4 \times (h \times (d + h) + h)$$

The factor of 4 is for the four gates (forget, input, candidate, output).

**Example:** Input=100, Hidden=256
- Parameters = $4 \times (256 \times 356 + 256) = 366,592$

### Example Implementation

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # out: (batch, seq_len, hidden_size)
        # Take output from last time step
        out = self.fc(out[:, -1, :])

        return out
```

## Gated Recurrent Unit (GRU)

**GRU** is a simplified variant of LSTM with fewer parameters.

### Architecture

GRU combines forget and input gates into a single **update gate**, and merges cell state and hidden state.

**Components:**
- **Update gate** $\mathbf{z}_t$ - controls how much past information to keep
- **Reset gate** $\mathbf{r}_t$ - controls how much past information to forget
- **Hidden state** $\mathbf{h}_t$ - no separate cell state

### Mathematical Formulation

**Update gate:**

$$\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Reset gate:**

$$\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Candidate activation:**

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Hidden state update:**

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

### GRU vs LSTM

**GRU advantages:**
- Fewer parameters (faster training, less overfitting)
- Simpler architecture
- Often comparable performance to LSTM

**LSTM advantages:**
- More expressive (separate cell state)
- Better for very long sequences
- More control over information flow

**Rule of thumb:** Try GRU first, use LSTM if you need the extra capacity.

### Example Implementation

```python
gru = nn.GRU(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.2
)

x = torch.randn(32, 50, 100)  # (batch, seq_len, features)
output, h_n = gru(x)
```

## Bidirectional RNNs

**Bidirectional RNNs** process sequences in both forward and backward directions.

### Architecture

```
Forward:  h₁→  h₂→  h₃→  h₄→
Input:     x₁    x₂    x₃    x₄
Backward: h₁←  h₂←  h₃←  h₄←
```

At each time step, concatenate forward and backward hidden states:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$$

### Use Cases

**Good for:**
- Text classification (entire sentence available)
- Named entity recognition
- Speech recognition
- Machine translation (encoder)

**Not suitable for:**
- Real-time prediction (need future information)
- Time series forecasting
- Online learning scenarios

### Implementation

```python
bi_lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True  # Key parameter
)

x = torch.randn(32, 50, 100)
output, (h_n, c_n) = bi_lstm(x)

# output: (batch, seq_len, 2 * hidden_size)
# Hidden size is doubled due to bidirectionality
```

## Common Architectures

### Sequence-to-Sequence (Seq2Seq)

**Used for:** Machine translation, summarization, chatbots

**Architecture:**
```
Encoder: Input sequence → Context vector
Decoder: Context vector → Output sequence
```

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # Encode
        encoder_outputs, hidden = self.encoder(src)

        # Decode
        output = self.decoder(trg, hidden)

        return output
```

### Sequence Classification

**Used for:** Sentiment analysis, document classification

```
Input sequence → RNN → Last hidden state → Dense layer → Class probabilities
```

### Sequence Labeling

**Used for:** Named entity recognition, part-of-speech tagging

```
Input sequence → RNN → Hidden states at each step → Dense layer → Labels
```

### Many-to-One

**Input:** Sequence of variable length
**Output:** Single value/vector

**Example:** Sentiment classification

### One-to-Many

**Input:** Single value/vector
**Output:** Sequence

**Example:** Image captioning

### Many-to-Many

**Input:** Sequence
**Output:** Sequence (same or different length)

**Example:** Machine translation, video classification

## Practical Considerations

### Hyperparameters

**Hidden size:**
- Start with 128-256
- Increase for complex tasks (512-1024)
- Diminishing returns beyond certain size

**Number of layers:**
- 1-2 layers often sufficient
- 3-4 layers for complex tasks
- Deeper networks need careful regularization

**Sequence length:**
- Truncate very long sequences
- Use hierarchical models for documents
- Consider using Transformers for very long contexts

### Regularization

**Dropout:**
```python
lstm = nn.LSTM(
    input_size=100,
    hidden_size=256,
    num_layers=2,
    dropout=0.3  # Applied between LSTM layers
)
```

**Recurrent dropout:** Applied to recurrent connections (more advanced)

**Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### Training Tips

**Learning rate:**
- Start with 0.001 (Adam)
- Use learning rate scheduling
- Reduce on plateau

**Batch size:**
- Sequences of different lengths in a batch need padding
- Sort sequences by length for efficiency
- Pack padded sequences:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Pack sequences
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

# Pass through RNN
output, hidden = lstm(packed)

# Unpack
output, _ = pad_packed_sequence(output, batch_first=True)
```

**Teacher forcing** (for seq2seq):
- During training, use ground truth as decoder input
- Helps with training stability
- Gradually reduce during training

### Handling Variable-Length Sequences

**Padding:**
```python
from torch.nn.utils.rnn import pad_sequence

sequences = [torch.randn(10, 5), torch.randn(15, 5), torch.randn(8, 5)]
padded = pad_sequence(sequences, batch_first=True, padding_value=0)
# Shape: (3, 15, 5) - padded to max length
```

**Masking:**
```python
# Create mask for padded positions
mask = (padded != 0).float()

# Apply mask to loss
loss = criterion(output, target)
loss = (loss * mask).sum() / mask.sum()
```

### Common Issues

**Gradient problems:**
- Use gradient clipping
- Try GRU instead of vanilla RNN
- Monitor gradient norms during training

**Slow training:**
- Use packed sequences
- Reduce sequence length
- Use smaller hidden size
- Consider using CuDNN-optimized implementations

**Poor performance:**
- Try bidirectional RNN
- Increase model capacity
- Add attention mechanism
- Consider Transformer architecture

### When to Use RNNs vs Transformers

**Use RNNs/LSTMs when:**
- Sequential processing is natural (online learning)
- Limited computational resources
- Small datasets
- Sequence length is moderate (<500)

**Use Transformers when:**
- Very long sequences
- Large datasets available
- Parallel processing is important
- State-of-the-art performance required

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder.
- Graves, A. (2013). Generating sequences with recurrent neural networks.
- Olah, C. (2015). Understanding LSTM Networks. [Blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch RNN Documentation](https://pytorch.org/docs/stable/nn.html#recurrent-layers)
