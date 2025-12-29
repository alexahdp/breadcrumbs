# Naive Bayes

## Table of Contents

1. [Introduction](#introduction)
2. [Bayes' Theorem](#bayes-theorem)
3. [Naive Bayes Classifier](#naive-bayes-classifier)
4. [Types of Naive Bayes](#types-of-naive-bayes)
5. [Training and Prediction](#training-and-prediction)
6. [Practical Considerations](#practical-considerations)
7. [Implementation](#implementation)
8. [References](#references)

## Introduction

**Naive Bayes** is a family of probabilistic classifiers based on Bayes' theorem with a "naive" assumption of conditional independence between features. Despite its simplicity, it often performs surprisingly well in practice.

### Key Characteristics

- **Probabilistic** - outputs class probabilities
- **Fast** - training and prediction are very efficient
- **Simple** - easy to implement and understand
- **Robust** - works well with small datasets
- **Scalable** - handles high-dimensional data well

### When to Use

**Best suited for:**
- Text classification (spam detection, sentiment analysis)
- Document categorization
- Real-time prediction (fast inference)
- High-dimensional datasets
- Baseline model for comparison

**Not recommended for:**
- When feature independence assumption is strongly violated
- When accuracy is critical (often outperformed by more complex models)
- Numerical estimation tasks (use regression instead)
- When feature interactions are important

### Advantages and Disadvantages

**Advantages:**
- Very fast training and prediction
- Works well with small training datasets
- Handles high-dimensional data naturally
- Not sensitive to irrelevant features
- Provides probabilistic predictions
- Easy to implement and interpret

**Disadvantages:**
- Independence assumption often unrealistic
- Can be outperformed by discriminative models
- Zero frequency problem (addressed by smoothing)
- Poor probability estimates (though classification is often good)

## Bayes' Theorem

### Fundamental Formula

**Bayes' Theorem** describes the probability of an event based on prior knowledge:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

where:
- $P(A|B)$ - **posterior probability**: probability of $A$ given $B$
- $P(B|A)$ - **likelihood**: probability of $B$ given $A$
- $P(A)$ - **prior probability**: probability of $A$
- $P(B)$ - **marginal probability**: total probability of $B$

### Application to Classification

For classification with features $X = (x_1, x_2, ..., x_n)$ and class $C$:

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

**Goal:** Find the class with maximum posterior probability:

$$\hat{C} = \arg\max_{C} P(C|X) = \arg\max_{C} \frac{P(X|C) \cdot P(C)}{P(X)}$$

Since $P(X)$ is constant for all classes:

$$\hat{C} = \arg\max_{C} P(X|C) \cdot P(C)$$

### Components

**Prior Probability** $P(C)$:
- Probability of class $C$ before seeing features
- Estimated from training data: $P(C) = \frac{n_C}{n}$
- $n_C$ is the number of samples in class $C$
- $n$ is the total number of samples

**Likelihood** $P(X|C)$:
- Probability of observing features $X$ given class $C$
- This is where the "naive" assumption comes in

**Posterior Probability** $P(C|X)$:
- Probability of class $C$ given observed features $X$
- What we want to compute for classification

## Naive Bayes Classifier

### Naive Independence Assumption

The **conditional independence assumption**: given the class, features are independent:

$$P(X|C) = P(x_1, x_2, ..., x_n|C) = \prod_{i=1}^{n} P(x_i|C)$$

This simplifies computation dramatically:

$$P(C|X) \propto P(C) \prod_{i=1}^{n} P(x_i|C)$$

### Classification Rule

**Maximum A Posteriori (MAP) Decision Rule:**

$$\hat{C} = \arg\max_{C} P(C) \prod_{i=1}^{n} P(x_i|C)$$

**Log probability** (for numerical stability):

$$\hat{C} = \arg\max_{C} \left[\log P(C) + \sum_{i=1}^{n} \log P(x_i|C)\right]$$

Using log probabilities:
- Avoids numerical underflow (multiplying many small probabilities)
- Converts products to sums (faster computation)
- More numerically stable

### Why "Naive"?

The independence assumption is rarely true in practice:
- In text: words are not independent (grammar, context)
- In images: pixels are highly correlated
- In tabular data: features often have dependencies

**Despite this**, Naive Bayes often works well because:
- Classification only needs the correct ordering of probabilities
- Errors in probability estimates may cancel out
- Simple model reduces overfitting

## Types of Naive Bayes

### Gaussian Naive Bayes

**For continuous features** assuming normal distribution:

$$P(x_i|C) = \frac{1}{\sqrt{2\pi\sigma_C^2}} \exp\left(-\frac{(x_i - \mu_C)^2}{2\sigma_C^2}\right)$$

where:
- $\mu_C$ is the mean of feature $x_i$ for class $C$
- $\sigma_C^2$ is the variance of feature $x_i$ for class $C$

**Training:** Compute mean and variance for each feature per class:

$$\mu_{C,i} = \frac{1}{n_C}\sum_{x \in C} x_i$$

$$\sigma_{C,i}^2 = \frac{1}{n_C}\sum_{x \in C} (x_i - \mu_{C,i})^2$$

**Use when:**
- Features are continuous
- Features approximately follow normal distribution
- General-purpose continuous data

### Multinomial Naive Bayes

**For discrete count features** (e.g., word counts, frequency data):

$$P(x_i|C) = \frac{N_{C,i} + \alpha}{N_C + \alpha n}$$

where:
- $N_{C,i}$ is the count of feature $i$ in class $C$
- $N_C$ is the total count of all features in class $C$
- $\alpha$ is the smoothing parameter (Laplace/additive smoothing)
- $n$ is the number of features

The probability of document $X = (x_1, ..., x_n)$ with feature counts:

$$P(X|C) = \frac{(\sum_i x_i)!}{\prod_i x_i!} \prod_{i=1}^{n} P(x_i|C)^{x_i}$$

**Use when:**
- Text classification with word counts
- Document categorization
- Features represent frequencies or counts

### Bernoulli Naive Bayes

**For binary features** (feature present/absent):

$$P(x_i|C) = P(i|C)x_i + (1 - P(i|C))(1 - x_i)$$

where:
- $x_i \in \{0, 1\}$ indicates presence/absence of feature $i$
- $P(i|C)$ is the probability that feature $i$ appears in class $C$

**Training:**

$$P(i|C) = \frac{N_{C,i} + \alpha}{N_C + 2\alpha}$$

where $N_{C,i}$ is the number of documents in class $C$ containing feature $i$.

**Use when:**
- Binary features (yes/no, present/absent)
- Text classification with binary term occurrence
- Smaller vocabulary than multinomial variant

**Difference from Multinomial:**
- Bernoulli: penalizes absence of features
- Multinomial: only considers present features

### Categorical Naive Bayes

**For categorical features** (discrete non-count data):

$$P(x_i = v|C) = \frac{N_{C,i,v} + \alpha}{N_C + \alpha n_i}$$

where:
- $x_i = v$ means feature $i$ has value $v$
- $N_{C,i,v}$ is count of samples in class $C$ where feature $i$ equals $v$
- $n_i$ is the number of possible values for feature $i$

**Use when:**
- Categorical features (colors, categories, labels)
- Mixed data types with nominal variables
- Survey data, questionnaire responses

## Training and Prediction

### Training Process

**Step 1:** Calculate class priors:

$$P(C_k) = \frac{n_k}{n}$$

where $n_k$ is the number of samples in class $k$.

**Step 2:** For each feature $i$ and class $C$, estimate $P(x_i|C)$:
- **Gaussian:** compute mean $\mu_{C,i}$ and variance $\sigma_{C,i}^2$
- **Multinomial:** compute feature frequencies
- **Bernoulli:** compute feature occurrence probabilities

**Complexity:** $O(nd)$ where $n$ is samples, $d$ is features.

### Prediction Process

**Step 1:** For each class $C_k$, compute log posterior:

$$\log P(C_k|X) = \log P(C_k) + \sum_{i=1}^{n} \log P(x_i|C_k)$$

**Step 2:** Predict class with maximum log posterior:

$$\hat{C} = \arg\max_{k} \log P(C_k|X)$$

**Step 3:** (Optional) Normalize to get probabilities:

$$P(C_k|X) = \frac{\exp(\log P(C_k|X))}{\sum_{j=1}^{K} \exp(\log P(C_j|X))}$$

**Complexity:** $O(Kd)$ where $K$ is number of classes.

## Practical Considerations

### Handling Zero Probabilities

**Problem:** If a feature value never occurs with a class in training:

$$P(x_i|C) = 0 \Rightarrow P(C|X) = 0$$

**Solution: Laplace Smoothing (Add-One Smoothing)**

For multinomial:

$$P(x_i|C) = \frac{N_{C,i} + \alpha}{N_C + \alpha n}$$

Common choices:
- $\alpha = 1$ (Laplace smoothing)
- $\alpha = 0.1$ (less smoothing)
- $\alpha = 0$ (no smoothing, may cause issues)

Effects:
- Prevents zero probabilities
- Shifts probability mass from frequent to rare events
- Large $\alpha$: more uniform distribution (high bias)
- Small $\alpha$: closer to observed frequencies (low bias)

### Feature Selection

Naive Bayes can handle high-dimensional data, but feature selection can improve performance:

**Techniques:**
1. **Mutual Information** - measures dependence between feature and class
2. **Chi-square test** - tests independence between feature and class
3. **Information Gain** - reduction in entropy
4. **Frequency thresholding** - remove rare features

```python
from sklearn.feature_selection import SelectKBest, chi2

# Select top k features
selector = SelectKBest(chi2, k=1000)
X_selected = selector.fit_transform(X, y)
```

### Handling Continuous Features

**Options for continuous data:**

1. **Gaussian Naive Bayes:**
   - Assumes normal distribution
   - Simple and fast
   - May fail if normality assumption violated

2. **Discretization:**
   - Bin continuous values into discrete intervals
   - Use Multinomial or Categorical NB
   - Loses information but more robust

3. **Kernel Density Estimation:**
   - Non-parametric density estimation
   - More flexible than Gaussian
   - Computationally expensive

### Class Imbalance

**Strategies:**

1. **Prior adjustment:** Manually set class priors:
   ```python
   # Equal priors instead of data-driven
   nb = MultinomialNB(class_prior=[0.5, 0.5])
   ```

2. **Resampling:** Over/undersample before training

3. **Cost-sensitive learning:** Adjust decision threshold based on costs

### Comparison with Other Algorithms

| Aspect | Naive Bayes | Logistic Regression | Decision Trees |
|--------|-------------|---------------------|----------------|
| Training speed | Very fast | Fast | Fast |
| Prediction speed | Very fast | Fast | Very fast |
| Probabilistic | Yes (native) | Yes (native) | Yes (counts) |
| Feature independence | Assumes | Not required | Not required |
| Non-linearity | No | No | Yes |
| Interpretability | High | High | Very high |
| High dimensions | Excellent | Good | Poor |
| Small datasets | Excellent | Good | Poor |

## Implementation

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes
gnb = GaussianNB(
    priors=None,    # Use class frequencies from data
    var_smoothing=1e-9  # Portion of largest variance added for stability
)

gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Model parameters
print(f"Class priors: {gnb.class_prior_}")
print(f"Means per class: {gnb.theta_}")
print(f"Variance per class: {gnb.var_}")
```

### Multinomial Naive Bayes (Text Classification)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Example text data
texts = ["I love this product", "terrible waste of money", ...]
labels = [1, 0, ...]  # 1: positive, 0: negative

# Convert text to word counts
vectorizer = CountVectorizer(
    max_features=5000,  # Limit vocabulary
    ngram_range=(1, 2), # Unigrams and bigrams
    stop_words='english'
)

X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Train Multinomial Naive Bayes
mnb = MultinomialNB(
    alpha=1.0,  # Laplace smoothing parameter
    fit_prior=True,  # Learn class priors from data
    class_prior=None  # Or specify custom priors
)

mnb.fit(X_train, y_train)

# Predictions
y_pred = mnb.predict(X_test)
y_proba = mnb.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Feature log probabilities
feature_names = vectorizer.get_feature_names_out()
for i, class_name in enumerate(mnb.classes_):
    top_features = np.argsort(mnb.feature_log_prob_[i])[-10:]
    print(f"Top features for class {class_name}:")
    print([feature_names[j] for j in top_features])
```

### Bernoulli Naive Bayes

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# Binary term occurrence (not counts)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Train Bernoulli Naive Bayes
bnb = BernoulliNB(
    alpha=1.0,      # Smoothing parameter
    binarize=0.0,   # Threshold for binarizing features
    fit_prior=True
)

bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Categorical Naive Bayes

```python
from sklearn.naive_bayes import CategoricalNB

# For categorical features (encoded as integers)
# Example: color=[0,1,2], size=[0,1], shape=[0,1,2,3]

cnb = CategoricalNB(
    alpha=1.0,  # Smoothing parameter
    fit_prior=True
)

cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Grid search for Multinomial NB
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
    'fit_prior': [True, False]
}

grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
```

### Complement Naive Bayes

```python
from sklearn.naive_bayes import ComplementNB

# Better for imbalanced datasets
cnb = ComplementNB(
    alpha=1.0,
    norm=True  # Apply weight normalization
)

cnb.fit(X_train, y_train)
y_pred = cnb.predict(X_test)

# Often outperforms MultinomialNB on text classification
```

### Spam Classification Example

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Create pipeline
spam_classifier = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Train
spam_classifier.fit(X_train_text, y_train)

# Predict
predictions = spam_classifier.predict(X_test_text)
probabilities = spam_classifier.predict_proba(X_test_text)

# Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['ham', 'spam']))
```

### Custom Prior Probabilities

```python
# When you have domain knowledge about class distribution
custom_priors = [0.7, 0.3]  # 70% class 0, 30% class 1

nb_custom = MultinomialNB(
    alpha=1.0,
    class_prior=custom_priors  # Override data-driven priors
)

nb_custom.fit(X_train, y_train)
```

### Incremental Learning

```python
from sklearn.naive_bayes import MultinomialNB

# Partial fit for streaming data
nb_incremental = MultinomialNB()

# Train on batches
for X_batch, y_batch in data_batches:
    nb_incremental.partial_fit(X_batch, y_batch, classes=[0, 1])

# Continue training with new data
nb_incremental.partial_fit(X_new, y_new)
```

## References

- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- Naive Bayes at Forty: The Independence Assumption in Information Retrieval (Lewis, 1998)
- Pattern Recognition and Machine Learning (Christopher Bishop)
- Introduction to Information Retrieval (Manning, Raghavan, Schütze)
- [On the Optimality of the Simple Bayesian Classifier under Zero-One Loss](https://link.springer.com/article/10.1023/A:1007413511361) (Domingos & Pazzani, 1997)
