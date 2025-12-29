# Natural Language Processing Overview

## Table of Contents

1. [Introduction to NLP](#introduction-to-nlp)
2. [Text Preprocessing](#text-preprocessing)
3. [Text Representation](#text-representation)
4. [Language Models](#language-models)
5. [Word Embeddings](#word-embeddings)
6. [Common NLP Tasks](#common-nlp-tasks)
7. [Transformer Architecture](#transformer-architecture)
8. [Modern NLP Approaches](#modern-nlp-approaches)

## Introduction to NLP

**Natural Language Processing (NLP)** is a field at the intersection of computer science, artificial intelligence, and linguistics, focused on enabling computers to understand, interpret, and generate human language.

### Key Challenges

NLP is challenging because natural language:
- Is ambiguous (same words can have different meanings)
- Has context-dependent meanings
- Contains idioms and expressions
- Varies across languages and dialects
- Evolves over time

### Applications

- **Machine translation**: Google Translate, DeepL
- **Sentiment analysis**: Product reviews, social media monitoring
- **Chatbots and assistants**: Customer service, virtual assistants
- **Text summarization**: News aggregation, document summarization
- **Named entity recognition**: Information extraction
- **Question answering**: Search engines, knowledge bases

## Text Preprocessing

### Basic Steps

**Tokenization** - splitting text into individual units (tokens):
- Word tokenization: "Hello world!" → ["Hello", "world", "!"]
- Sentence tokenization: Splitting documents into sentences
- Subword tokenization: Breaking words into meaningful subunits

**Lowercasing** - converting all text to lowercase:
- "Machine Learning" → "machine learning"
- Helps reduce vocabulary size but may lose information (e.g., "US" vs "us")

**Removing Punctuation and Special Characters**:
- "Hello, world!" → "Hello world"
- May need to keep some punctuation for certain tasks

**Stop Words Removal** - removing common words with little semantic value:
- Words like: "the", "is", "at", "which", "on"
- Can be task-dependent (important for some applications)

**Stemming** - reducing words to their root form:
- "running", "runs", "ran" → "run"
- Fast but may produce non-words

**Lemmatization** - reducing words to their dictionary form:
- "running" → "run", "better" → "good"
- More accurate than stemming but slower

### Advanced Preprocessing

**Part-of-Speech (POS) Tagging** - identifying grammatical roles:
- "The quick brown fox" → [DET, ADJ, ADJ, NOUN]

**Named Entity Recognition (NER)** - identifying proper nouns:
- "Apple released iPhone in California" → [ORG, PRODUCT, LOC]

**Dependency Parsing** - analyzing grammatical structure and relationships between words.

## Text Representation

### Bag of Words (BoW)

Represents text as an unordered collection of words, ignoring grammar and word order:

```python
# Document: "the cat sat on the mat"
# Vocabulary: ["cat", "mat", "on", "sat", "the"]
# BoW vector: [1, 1, 1, 1, 2]
```

**Limitations**:
- Ignores word order and context
- High dimensionality
- Sparse representations

### TF-IDF (Term Frequency-Inverse Document Frequency)

Weights words by their importance in a document relative to a corpus:

$$\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)$$

where:

$$\text{TF}(t,d) = \frac{\text{count of term } t \text{ in document } d}{\text{total terms in document } d}$$

$$\text{IDF}(t) = \log\frac{\text{total number of documents}}{\text{number of documents containing term } t}$$

**Benefits**:
- Downweights common words
- Highlights distinctive terms
- Better than raw counts

### N-grams

Sequences of N consecutive words:
- Unigram (1-gram): ["machine", "learning"]
- Bigram (2-gram): ["machine learning"]
- Trigram (3-gram): ["deep machine learning"]

Captures some local word order information but increases dimensionality exponentially.

## Language Models

**Language Model** - a probabilistic model that predicts the next word given previous words:

$$P(w_t | w_1, w_2, ..., w_{t-1})$$

### N-gram Language Models

Based on Markov assumption that next word depends only on previous N-1 words:

$$P(w_t | w_1, ..., w_{t-1}) \approx P(w_t | w_{t-N+1}, ..., w_{t-1})$$

**Bigram model**:
$$P(w_t | w_{t-1})$$

**Limitations**:
- Cannot capture long-range dependencies
- Sparse for large N
- Fixed context window

### Neural Language Models

Use neural networks to learn continuous representations and capture longer dependencies:
- **Recurrent Neural Networks (RNN)**: Process sequences step by step
- **Long Short-Term Memory (LSTM)**: Better at capturing long-range dependencies
- **Transformer models**: Use attention mechanisms (see below)

## Word Embeddings

**Word Embeddings** - dense vector representations of words that capture semantic meaning:

$$\text{"king"} \rightarrow [0.2, -0.5, 0.7, ..., 0.3] \in \mathbb{R}^d$$

### Word2Vec

Two architectures for learning word embeddings:

**CBOW (Continuous Bag of Words)** - predicts target word from context:
$$P(w_t | w_{t-n}, ..., w_{t-1}, w_{t+1}, ..., w_{t+n})$$

**Skip-gram** - predicts context words from target:
$$P(w_{t+i} | w_t)$$

**Properties**:
- Similar words have similar vectors
- Captures semantic relationships
- Enables vector arithmetic: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$

### GloVe (Global Vectors)

Combines global matrix factorization with local context windows:

$$J = \sum_{i,j=1}^V f(X_{ij})(\vec{w}_i^T \vec{w}_j + b_i + b_j - \log X_{ij})^2$$

where $X_{ij}$ is the co-occurrence count of words $i$ and $j$.

### FastText

Extension of Word2Vec that represents words as bags of character n-grams:
- Handles out-of-vocabulary words
- Works well for morphologically rich languages
- Captures subword information

### Contextual Embeddings

Traditional embeddings assign one vector per word. **Contextual embeddings** generate different vectors based on context:
- ELMo (Embeddings from Language Models)
- BERT (see Transformer Architecture)
- GPT series

## Common NLP Tasks

### Text Classification

Assigning categories to text:
- **Sentiment analysis**: Positive, negative, neutral
- **Topic classification**: Sports, politics, technology
- **Spam detection**: Spam vs not spam

### Sequence Labeling

Assigning labels to each token:
- **Named Entity Recognition (NER)**: Person, location, organization
- **Part-of-Speech Tagging**: Noun, verb, adjective

### Sequence-to-Sequence

Transforming input sequence to output sequence:
- **Machine translation**: English → French
- **Text summarization**: Long text → summary
- **Question answering**: Question + context → answer

### Information Retrieval

Finding relevant documents or passages:
- Search engines
- Document ranking
- Semantic search

## Transformer Architecture

**Transformers** revolutionized NLP by replacing recurrence with attention mechanisms.

### Self-Attention Mechanism

Allows each word to attend to all other words in the sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $Q$ (Query), $K$ (Key), $V$ (Value) are learned projections
- $d_k$ is the dimension of keys

**Benefits**:
- Captures long-range dependencies
- Parallelizable (unlike RNNs)
- No vanishing gradient problem

### Multi-Head Attention

Runs multiple attention operations in parallel:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$

where:
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Allows the model to attend to different representation subspaces.

### Positional Encoding

Since Transformers have no recurrence, position information is added:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### Architecture Components

**Encoder**: Processes input sequence
- Self-attention layers
- Feed-forward networks
- Layer normalization

**Decoder**: Generates output sequence
- Masked self-attention (prevents looking ahead)
- Cross-attention to encoder outputs
- Feed-forward networks

## Modern NLP Approaches

### Pre-training and Fine-tuning

**Transfer learning** paradigm:
1. **Pre-train** on large unlabeled corpus (language modeling)
2. **Fine-tune** on specific downstream task with labeled data

**Benefits**:
- Leverages large-scale unlabeled data
- Improves performance on low-resource tasks
- Reduces training time for specific tasks

### BERT (Bidirectional Encoder Representations from Transformers)

Pre-training objectives:
- **Masked Language Modeling (MLM)**: Predict randomly masked words
- **Next Sentence Prediction (NSP)**: Predict if two sentences are consecutive

**Key innovation**: Bidirectional context (unlike GPT which is unidirectional)

### GPT (Generative Pre-trained Transformer)

- Autoregressive language model (predicts next token)
- Unidirectional (left-to-right)
- Scales well with model size and data

### T5 (Text-to-Text Transfer Transformer)

Frames all NLP tasks as text-to-text:
- Translation: "translate English to German: Hello" → "Hallo"
- Classification: "sentiment: Great movie!" → "positive"

### Recent Trends

**Large Language Models (LLMs)**:
- GPT-3, GPT-4: 175B+ parameters
- PaLM, LLaMA, Claude: Various scales
- Emergent abilities at scale

**Instruction Tuning**: Training models to follow instructions

**Few-shot Learning**: Learning from few examples in context

**Prompt Engineering**: Crafting inputs to elicit desired outputs

## Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class and macro/micro averaged

### Sequence Generation Metrics

**BLEU (Bilingual Evaluation Understudy)** - measures n-gram overlap:
$$BLEU = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** - measures recall of n-grams

**Perplexity** - measures how well a model predicts text:
$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^N \log P(w_i | w_{<i})\right)$$

Lower perplexity indicates better prediction.

## Practical Considerations

### Data Requirements
- Pre-training requires massive datasets (billions of tokens)
- Fine-tuning can work with smaller datasets (thousands of examples)
- Data quality matters more than quantity for specific tasks

### Computational Resources
- Large models require significant GPU/TPU resources
- Inference can be expensive for real-time applications
- Distillation can create smaller, faster models

### Common Libraries

```python
# Transformers library (Hugging Face)
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# SpaCy for preprocessing
import spacy
nlp = spacy.load("en_core_web_sm")

# NLTK for basic NLP
import nltk
from nltk.tokenize import word_tokenize
```

## References

- "Attention Is All You Need" (Vaswani et al., 2017) - Original Transformer paper
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- "Language Models are Few-Shot Learners" (Brown et al., 2020) - GPT-3 paper
- Speech and Language Processing (Jurafsky & Martin)
- Natural Language Processing with Transformers (Tunstall et al.)
