# Machine Learning Glossary

A comprehensive reference of machine learning terminology and concepts organized alphabetically.

## A

**Accuracy** - the proportion of correct predictions among all predictions made by a classifier:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Activation Function** - a function applied to a neuron's output to introduce non-linearity. Common examples: ReLU, Sigmoid, Tanh, Softmax.

**Adam Optimizer** - Adaptive Moment Estimation, an optimization algorithm that combines momentum and RMSprop. Computes adaptive learning rates for each parameter.

**Autoencoder (AE)** - a neural network architecture that learns to compress data into a lower-dimensional representation and then reconstruct it.

**AUC (Area Under Curve)** - the area under the ROC curve, measuring the quality of a binary classifier across all classification thresholds. Values range from 0 to 1, where 1 is perfect classification.

## B

**Backpropagation** - an algorithm for training neural networks that computes gradients of the loss function with respect to weights using the chain rule.

**Bagging** - Bootstrap Aggregating, an ensemble method that trains multiple models on different random samples of the training data and averages their predictions.

**Batch Normalization** - a technique that normalizes layer inputs during training to stabilize and accelerate the learning process.

**Batch Size** - the number of training examples used in one iteration of gradient descent.

**Bayesian Optimization** - a method for hyperparameter tuning that builds a probabilistic model of the objective function to find optimal parameters efficiently.

**Bias** - (1) a learnable parameter in neural networks added to weighted inputs; (2) systematic error in predictions (bias-variance tradeoff).

**Bias-Variance Tradeoff** - the balance between a model's ability to minimize bias (error from wrong assumptions) and variance (error from sensitivity to training data fluctuations).

**Binary Classification** - a classification task with exactly two classes (e.g., spam/not spam, positive/negative).

**Binomial Distribution** - a discrete probability distribution of the number of successes in a sequence of $n$ independent Bernoulli trials.

**Boosting** - an ensemble method that sequentially trains models, where each new model focuses on correcting errors made by previous models. Examples: AdaBoost, XGBoost, LightGBM.

## C

**CART (Classification and Regression Trees)** - an algorithm for building decision trees that can perform both classification and regression tasks.

**CatBoost** - Categorical Boosting, a gradient boosting algorithm optimized for handling categorical features.

**Categorical Data** - data that can be divided into specific groups or categories (nominal or ordinal).

**Central Limit Theorem** - states that the distribution of sample means approaches a normal distribution as sample size increases, regardless of the population distribution.

**Classification** - a supervised learning task where the goal is to predict a discrete class label.

**Cluster** - a group of similar data points identified by clustering algorithms.

**Clustering** - an unsupervised learning task that groups similar data points together.

**CNN (Convolutional Neural Network)** - a deep learning architecture designed for processing grid-like data (images), using convolutional layers to extract hierarchical features.

**Confidence Interval** - a range of values within which a population parameter is likely to fall with a specified probability (confidence level).

**Confusion Matrix** - a table showing true positives, true negatives, false positives, and false negatives for a classification model.

**Convolution** - a mathematical operation that applies a filter (kernel) to an input to extract features. Core operation in CNNs.

**Correlation** - a statistical measure of the linear relationship between two variables, typically measured by Pearson or Spearman coefficients.

**Cost Function** - see Loss Function.

**Covariance** - a measure of how much two random variables change together:

$$Cov(X,Y) = E[(X - E[X])(Y - E[Y])]$$

**Cross-Entropy Loss** - a loss function commonly used for classification tasks:

$$L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$$

**Cross-Validation** - a technique for assessing model performance by splitting data into multiple folds and training/testing on different combinations.

## D

**Data Augmentation** - techniques to artificially increase training data size by applying transformations (rotation, flipping, scaling, etc.).

**DBSCAN (Density-Based Spatial Clustering)** - a clustering algorithm that groups points based on density, capable of finding arbitrarily shaped clusters.

**Decision Boundary** - the hypersurface that separates different classes in the feature space.

**Decision Tree** - a tree-structured model that makes predictions by learning decision rules from features.

**Deep Learning** - a subset of machine learning using neural networks with multiple hidden layers.

**Dendrogram** - a tree diagram showing the hierarchical relationship between clusters in hierarchical clustering.

**Dimensionality Reduction** - techniques to reduce the number of features while preserving important information (e.g., PCA, t-SNE).

**Dropout** - a regularization technique that randomly sets a fraction of neuron outputs to zero during training to prevent overfitting.

## E

**Early Stopping** - a regularization technique that stops training when validation performance stops improving.

**Embedding** - a learned dense vector representation of discrete variables (e.g., words, categories).

**Ensemble Learning** - combining multiple models to improve prediction performance. Methods include bagging, boosting, and stacking.

**Epoch** - one complete pass through the entire training dataset.

**Error Function** - see Loss Function.

**Euclidean Distance** - the straight-line distance between two points:

$$d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**Exploding Gradient** - a problem in training deep neural networks where gradients become extremely large, causing unstable updates.

## F

**F1 Score** - the harmonic mean of precision and recall:

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

**False Negative (FN)** - an instance where the model incorrectly predicts the negative class.

**False Positive (FP)** - an instance where the model incorrectly predicts the positive class.

**Feature** - an individual measurable property or characteristic used as input to a model.

**Feature Engineering** - the process of creating new features or transforming existing ones to improve model performance.

**Feature Scaling** - normalizing or standardizing features to a common scale.

**Feedforward Neural Network** - a neural network where information flows in one direction from input to output without cycles.

**Fully Connected Layer** - a neural network layer where every neuron is connected to every neuron in the previous layer. Also called Dense layer.

## G

**GAN (Generative Adversarial Network)** - a framework where two neural networks (generator and discriminator) compete: the generator creates synthetic data while the discriminator tries to distinguish real from fake.

**Gaussian Distribution** - see Normal Distribution.

**Generalization** - a model's ability to perform well on unseen data.

**Gradient** - the vector of partial derivatives of a function with respect to its parameters, indicating the direction of steepest ascent.

**Gradient Descent** - an optimization algorithm that iteratively adjusts parameters in the direction of steepest descent of the loss function:

$$\theta_{new} = \theta_{old} - \alpha \nabla L(\theta)$$

where $\alpha$ is the learning rate.

**Gradient Boosting** - an ensemble method that builds models sequentially, with each new model predicting the residual errors of previous models.

**Grid Search** - a hyperparameter tuning method that exhaustively searches through a manually specified parameter grid.

## H

**Hierarchical Clustering** - a clustering method that builds a hierarchy of clusters, either bottom-up (agglomerative) or top-down (divisive).

**Hyperparameter** - a parameter set before training that controls the learning process (e.g., learning rate, number of layers).

**Hyperparameter Tuning** - the process of finding optimal hyperparameter values for a model.

**Hypothesis** - in machine learning, a candidate model or function that maps inputs to outputs.

## I

**Imbalanced Dataset** - a dataset where class distribution is highly skewed, with one class significantly more frequent than others.

**Imputation** - the process of replacing missing values with substituted values.

**Inference** - using a trained model to make predictions on new data.

**K-Fold Cross-Validation** - a cross-validation technique that splits data into $k$ folds, using each fold once as validation while training on the remaining $k-1$ folds.

**KNN (K-Nearest Neighbors)** - an instance-based algorithm that classifies or regresses based on the majority vote or average of the $k$ nearest training examples.

**Kernel** - (1) in SVM, a function that transforms data into higher dimensions; (2) in CNN, a filter matrix used in convolution operations.

**K-Means** - a clustering algorithm that partitions data into $k$ clusters by minimizing within-cluster variance.

**Kurtosis** - a measure of the "tailedness" of a probability distribution.

## L

**L1 Regularization (Lasso)** - adds the sum of absolute values of weights to the loss function:

$$L_{reg} = L + \lambda \sum_{i}|\theta_i|$$

Tends to produce sparse models (many zero weights).

**L2 Regularization (Ridge)** - adds the sum of squared weights to the loss function:

$$L_{reg} = L + \lambda \sum_{i}\theta_i^2$$

**Label** - the target variable or output in supervised learning.

**Lasso Regression** - linear regression with L1 regularization.

**Latent Variable** - an unobserved variable inferred from observed variables.

**Learning Curve** - a plot showing model performance (training and validation) as a function of training set size or epochs.

**Learning Rate** - a hyperparameter controlling the step size in gradient descent optimization.

**LightGBM** - Light Gradient Boosting Machine, a fast gradient boosting framework using tree-based algorithms.

**Linear Regression** - a regression method that models the relationship between features and target as a linear combination:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n$$

**Logistic Regression** - a classification algorithm that models the probability of class membership using the logistic (sigmoid) function:

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

**Loss Function** - a function measuring the difference between predicted and actual values. The objective is to minimize this function during training.

**LSTM (Long Short-Term Memory)** - a type of recurrent neural network architecture designed to handle long-term dependencies.

## M

**MAE (Mean Absolute Error)** - average of absolute differences between predicted and actual values:

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Manhattan Distance** - the sum of absolute differences between coordinates:

$$d(x,y) = \sum_{i=1}^{n}|x_i - y_i|$$

**Margin of Error** - the maximum expected difference between the true population parameter and a sample estimate.

**Max Pooling** - a downsampling operation in CNNs that takes the maximum value from each patch of the feature map.

**Mean** - the average value of a dataset.

**Median** - the middle value in a sorted dataset.

**Mini-Batch Gradient Descent** - gradient descent using a subset of training data in each iteration.

**MLP (Multi-Layer Perceptron)** - a feedforward neural network with at least one hidden layer.

**Mode** - the most frequently occurring value in a dataset.

**Model** - a mathematical representation learned from data to make predictions or decisions.

**MSE (Mean Squared Error)** - average of squared differences between predicted and actual values:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Multiclass Classification** - classification with more than two classes.

**Multinomial Distribution** - a generalization of binomial distribution for multiple possible outcomes.

## N

**Naive Bayes** - a probabilistic classifier based on Bayes' theorem with the assumption of feature independence:

$$P(y|x_1,...,x_n) = \frac{P(y)P(x_1,...,x_n|y)}{P(x_1,...,x_n)}$$

**Neural Network** - a computational model inspired by biological neural networks, consisting of interconnected nodes (neurons) organized in layers.

**Neuron** - a computational unit in a neural network that receives inputs, applies weights and bias, and passes the result through an activation function.

**Normal Distribution** - a continuous probability distribution with bell-shaped curve:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Normalization** - scaling features to a standard range (e.g., [0, 1]).

## O

**Objective Function** - the function to be optimized (minimized or maximized) during training.

**One-Hot Encoding** - representing categorical variables as binary vectors where only one element is 1.

**Optimizer** - an algorithm used to update model parameters to minimize the loss function (e.g., SGD, Adam, RMSprop).

**Ordinal Data** - categorical data with a meaningful order or ranking.

**Outlier** - a data point significantly different from other observations.

**Overfitting** - when a model learns training data too well, including noise, resulting in poor generalization to new data.

## P

**p-value** - the probability of obtaining results at least as extreme as observed, assuming the null hypothesis is true.

**Parameter** - a variable learned by the model from training data (e.g., weights, biases).

**PCA (Principal Component Analysis)** - a dimensionality reduction technique that transforms data into a new coordinate system where the greatest variance lies on the first coordinates (principal components).

**Perceptron** - the simplest neural network unit that computes a weighted sum of inputs and applies an activation function.

**Polynomial Regression** - regression using polynomial features to model non-linear relationships.

**Pooling** - a downsampling operation in CNNs that reduces spatial dimensions of feature maps.

**Population** - the complete set of all possible observations.

**Precision** - the proportion of true positives among all positive predictions:

$$Precision = \frac{TP}{TP + FP}$$

**Prediction** - the output of a model for a given input.

**Preprocessing** - transforming raw data into a format suitable for machine learning algorithms.

## Q

**Q-Q Plot** - Quantile-Quantile plot, a graphical tool to assess if data follows a particular distribution.

## R

**R² (R-squared, Coefficient of Determination)** - the proportion of variance in the dependent variable explained by the model:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

**Random Forest** - an ensemble of decision trees trained on random subsets of data and features, with predictions averaged or voted.

**Random Search** - a hyperparameter tuning method that randomly samples from the parameter space.

**Recall** - the proportion of true positives among all actual positives:

$$Recall = \frac{TP}{TP + FN}$$

Also called Sensitivity or True Positive Rate.

**Rectified Linear Unit (ReLU)** - an activation function:

$$ReLU(x) = \max(0, x)$$

**Recurrent Neural Network (RNN)** - a neural network with connections forming cycles, designed for sequential data.

**Regression** - a supervised learning task predicting continuous values.

**Regularization** - techniques to prevent overfitting by adding constraints or penalties to the model.

**Reinforcement Learning** - a learning paradigm where an agent learns to make decisions by interacting with an environment and receiving rewards.

**Residual** - the difference between observed and predicted values.

**Ridge Regression** - linear regression with L2 regularization.

**RMSE (Root Mean Squared Error)** - the square root of MSE:

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**RNN (Recurrent Neural Network)** - see Recurrent Neural Network.

**ROC Curve (Receiver Operating Characteristic)** - a plot of True Positive Rate vs False Positive Rate across different classification thresholds.

**RMSprop** - Root Mean Square Propagation, an adaptive learning rate optimization algorithm.

## S

**Sample** - a subset of a population used for analysis.

**Sampling** - the process of selecting a subset from a population.

**SGD (Stochastic Gradient Descent)** - gradient descent using one random sample (or mini-batch) per iteration.

**Sigmoid Function** - an activation function that maps values to (0, 1):

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Skewness** - a measure of asymmetry in a probability distribution.

**Softmax Function** - converts a vector of values into a probability distribution:

$$softmax(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

**Standard Deviation** - the square root of variance, measuring spread of data:

$$\sigma = \sqrt{Var(X)}$$

**Standard Error** - the standard deviation of a sample statistic:

$$SE = \frac{\sigma}{\sqrt{n}}$$

**Standardization** - scaling features to have mean 0 and standard deviation 1 (Z-score normalization).

**Stratified Sampling** - sampling that preserves the proportion of classes from the original dataset.

**Supervised Learning** - learning from labeled data where both input and output are provided.

**Support Vector Machine (SVM)** - a classifier that finds the optimal hyperplane maximizing the margin between classes.

## T

**t-SNE (t-Distributed Stochastic Neighbor Embedding)** - a dimensionality reduction technique particularly suited for visualization of high-dimensional data.

**Tanh (Hyperbolic Tangent)** - an activation function that maps values to (-1, 1):

$$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Test Set** - data held out from training to evaluate final model performance.

**Training Set** - data used to train the model.

**Transfer Learning** - using knowledge gained from one task to improve learning on a related task, often by using pre-trained models.

**True Negative (TN)** - correctly predicted negative instances.

**True Positive (TP)** - correctly predicted positive instances.

## U

**Underfitting** - when a model is too simple to capture underlying patterns in data, resulting in poor performance on both training and test data.

**Unsupervised Learning** - learning from unlabeled data to discover hidden patterns or structures.

## V

**Validation Set** - data used during training to tune hyperparameters and prevent overfitting.

**Vanishing Gradient** - a problem in training deep networks where gradients become extremely small, preventing effective learning in early layers.

**Variance** - a measure of spread of data around the mean:

$$\sigma^2 = E[(X - \mu)^2]$$

**VAE (Variational Autoencoder)** - a generative model that learns a probabilistic mapping between input data and a latent space.

## W

**Weight** - a learnable parameter in a model that determines the strength of connection between neurons or the importance of a feature.

**Weight Decay** - another term for L2 regularization.

**Weight Initialization** - the process of setting initial values for model weights before training.

## X

**XGBoost** - Extreme Gradient Boosting, an optimized gradient boosting framework known for high performance.

## Z

**Z-Score** - a standardized score showing how many standard deviations a value is from the mean:

$$Z = \frac{x - \mu}{\sigma}$$

**Zero-Shot Learning** - the ability of a model to recognize classes it has never seen during training.

## References

- Deep Learning (Goodfellow, Bengio, Courville)
- Pattern Recognition and Machine Learning (Bishop)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)
- Scikit-learn Documentation
- PyTorch Documentation
- TensorFlow Documentation
