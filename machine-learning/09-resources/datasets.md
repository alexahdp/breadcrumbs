# Public Datasets for Machine Learning

A curated collection of publicly available datasets for machine learning practice and research.

## Table of Contents

1. [General ML Datasets](#general-ml-datasets)
2. [Computer Vision](#computer-vision)
3. [Natural Language Processing](#natural-language-processing)
4. [Time Series](#time-series)
5. [Audio and Speech](#audio-and-speech)
6. [Healthcare and Biology](#healthcare-and-biology)
7. [Recommender Systems](#recommender-systems)
8. [Geospatial Data](#geospatial-data)
9. [Dataset Repositories](#dataset-repositories)

## General ML Datasets

### Classic Datasets for Learning

**Iris Dataset**
- **Size**: 150 samples, 4 features
- **Task**: Multi-class classification (3 species)
- **Use for**: Learning classification algorithms
- **Source**: UCI ML Repository / scikit-learn
- **Load**: `sklearn.datasets.load_iris()`

**Wine Dataset**
- **Size**: 178 samples, 13 features
- **Task**: Classification (3 wine types)
- **Use for**: Classification, feature selection
- **Source**: UCI ML Repository / scikit-learn
- **Load**: `sklearn.datasets.load_wine()`

**Boston Housing**
- **Size**: 506 samples, 13 features
- **Task**: Regression (house prices)
- **Use for**: Regression algorithms
- **Source**: scikit-learn (deprecated, use California Housing instead)
- **Note**: Contains ethical concerns, use alternatives

**California Housing**
- **Size**: 20,640 samples, 8 features
- **Task**: Regression (median house value)
- **Use for**: Regression, spatial analysis
- **Source**: scikit-learn
- **Load**: `sklearn.datasets.fetch_california_housing()`

**Titanic Dataset**
- **Size**: 891 samples (train), 418 (test)
- **Task**: Binary classification (survival)
- **Use for**: Classification, handling missing data, feature engineering
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/c/titanic

**Adult Income (Census)**
- **Size**: 48,842 samples, 14 features
- **Task**: Binary classification (income >50K)
- **Use for**: Classification, handling categorical data
- **Source**: UCI ML Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult

### Intermediate Datasets

**Credit Card Fraud Detection**
- **Size**: 284,807 transactions
- **Task**: Binary classification (fraud detection)
- **Use for**: Imbalanced classification, anomaly detection
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/mlg-ulb/creditcardfraud

**Diabetes Dataset**
- **Size**: 768 samples, 8 features
- **Task**: Binary classification (diabetes)
- **Use for**: Medical predictions, imbalanced data
- **Source**: Kaggle / UCI
- **URL**: https://www.kaggle.com/uciml/pima-indians-diabetes-database

**Heart Disease**
- **Size**: 303 samples, 13 features
- **Task**: Binary classification (heart disease)
- **Use for**: Medical classification
- **Source**: UCI ML Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/heart+disease

**Bank Marketing**
- **Size**: 45,211 samples, 16 features
- **Task**: Binary classification (subscription)
- **Use for**: Marketing analytics, imbalanced data
- **Source**: UCI ML Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/bank+marketing

## Computer Vision

### Image Classification

**MNIST**
- **Size**: 70,000 images (60k train, 10k test)
- **Resolution**: 28x28 grayscale
- **Task**: Digit classification (0-9)
- **Use for**: Learning CNNs, benchmarking
- **Source**: Yann LeCun's website / TensorFlow / PyTorch
- **Load**: `torchvision.datasets.MNIST()` or `tensorflow.keras.datasets.mnist`

**Fashion-MNIST**
- **Size**: 70,000 images (60k train, 10k test)
- **Resolution**: 28x28 grayscale
- **Task**: Clothing classification (10 classes)
- **Use for**: More challenging than MNIST
- **Source**: Zalando Research
- **URL**: https://github.com/zalandoresearch/fashion-mnist

**CIFAR-10**
- **Size**: 60,000 images (50k train, 10k test)
- **Resolution**: 32x32 RGB
- **Task**: Object classification (10 classes)
- **Use for**: Learning CNNs, data augmentation
- **Source**: Canadian Institute for Advanced Research
- **Load**: `torchvision.datasets.CIFAR10()`

**CIFAR-100**
- **Size**: 60,000 images (50k train, 10k test)
- **Resolution**: 32x32 RGB
- **Task**: Object classification (100 classes)
- **Use for**: Multi-class classification, transfer learning
- **Source**: CIFAR
- **Load**: `torchvision.datasets.CIFAR100()`

**ImageNet (ILSVRC)**
- **Size**: 1.2M training images, 50k validation, 100k test
- **Resolution**: Variable (typically 224x224 after preprocessing)
- **Task**: Object classification (1000 classes)
- **Use for**: Transfer learning, pre-training models
- **Source**: Stanford University
- **URL**: http://www.image-net.org/
- **Note**: Requires registration

**STL-10**
- **Size**: 13,000 labeled images, 100,000 unlabeled
- **Resolution**: 96x96 RGB
- **Task**: Object recognition (10 classes)
- **Use for**: Unsupervised learning, semi-supervised learning
- **URL**: https://cs.stanford.edu/~acoates/stl10/

**Caltech-101 / Caltech-256**
- **Size**: 9,144 images (101) / 30,607 images (256)
- **Resolution**: Variable
- **Task**: Object recognition
- **Use for**: Transfer learning, object detection
- **URL**: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

### Object Detection

**COCO (Common Objects in Context)**
- **Size**: 330K images, 1.5M object instances
- **Task**: Object detection, segmentation, keypoints
- **Use for**: Object detection, instance segmentation
- **Source**: Microsoft
- **URL**: https://cocodataset.org/

**Pascal VOC**
- **Size**: ~11,500 images with 27,450 annotated objects
- **Task**: Object detection, segmentation
- **Use for**: Object detection benchmarking
- **URL**: http://host.robots.ox.ac.uk/pascal/VOC/

**Open Images**
- **Size**: 9M images with 16M bounding boxes
- **Task**: Object detection, visual relationships
- **Use for**: Large-scale object detection
- **URL**: https://storage.googleapis.com/openimages/web/index.html

### Face Recognition

**Labeled Faces in the Wild (LFW)**
- **Size**: 13,000 face images
- **Task**: Face verification, recognition
- **Use for**: Face recognition benchmarking
- **URL**: http://vis-www.cs.umass.edu/lfw/

**CelebA**
- **Size**: 202,599 celebrity images
- **Task**: Face attribute recognition, generation
- **Use for**: GANs, face attributes
- **URL**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

### Medical Imaging

**ChestX-ray14**
- **Size**: 112,120 frontal-view X-ray images
- **Task**: Disease classification (14 diseases)
- **Use for**: Medical image classification
- **URL**: https://nihcc.app.box.com/v/ChestXray-NIHCC

**ISIC Skin Lesion**
- **Size**: 33,126 dermoscopic images
- **Task**: Skin cancer classification
- **Use for**: Medical classification, melanoma detection
- **URL**: https://www.isic-archive.com/

**BraTS (Brain Tumor Segmentation)**
- **Size**: ~300 MRI scans
- **Task**: Brain tumor segmentation
- **Use for**: Medical image segmentation
- **URL**: http://braintumorsegmentation.org/

## Natural Language Processing

### Text Classification

**IMDb Movie Reviews**
- **Size**: 50,000 reviews (25k train, 25k test)
- **Task**: Sentiment analysis (binary)
- **Use for**: Text classification, sentiment analysis
- **Source**: Stanford
- **Load**: `tensorflow.keras.datasets.imdb`

**20 Newsgroups**
- **Size**: ~20,000 documents
- **Task**: Multi-class text classification (20 topics)
- **Use for**: Document classification, clustering
- **Source**: scikit-learn
- **Load**: `sklearn.datasets.fetch_20newsgroups()`

**AG News**
- **Size**: 120,000 training samples, 7,600 test
- **Task**: News categorization (4 classes)
- **Use for**: Text classification
- **URL**: https://www.kaggle.com/amananandrai/ag-news-classification-dataset

**Yelp Reviews**
- **Size**: 6.99M reviews
- **Task**: Sentiment analysis, rating prediction
- **Use for**: Sentiment analysis, review classification
- **URL**: https://www.yelp.com/dataset

**Spam Classification**
- **Size**: 5,572 SMS messages
- **Task**: Binary classification (spam/ham)
- **Use for**: Text classification, NLP basics
- **URL**: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

### Question Answering

**SQuAD (Stanford Question Answering)**
- **Size**: 100,000+ question-answer pairs
- **Task**: Reading comprehension, QA
- **Use for**: Question answering systems
- **URL**: https://rajpurkar.github.io/SQuAD-explorer/

**Natural Questions (Google)**
- **Size**: 307,373 training examples
- **Task**: Question answering
- **Use for**: Open-domain QA
- **URL**: https://ai.google.com/research/NaturalQuestions

### Machine Translation

**WMT (Workshop on Machine Translation)**
- **Task**: Machine translation (various language pairs)
- **Use for**: Neural machine translation
- **URL**: http://www.statmt.org/wmt21/

**Multi30k**
- **Size**: 31,000 images with captions in English, German, French
- **Task**: Image captioning, translation
- **URL**: https://github.com/multi30k/dataset

### Named Entity Recognition

**CoNLL-2003**
- **Task**: Named entity recognition (PER, LOC, ORG, MISC)
- **Use for**: NER benchmarking
- **URL**: https://www.clips.uantwerpen.be/conll2003/ner/

**OntoNotes**
- **Size**: 1M words
- **Task**: NER, coreference resolution
- **URL**: https://catalog.ldc.upenn.edu/LDC2013T19

## Time Series

**Stock Market Data**
- **Source**: Yahoo Finance, Alpha Vantage, Quandl
- **Task**: Price prediction, trend analysis
- **Use for**: Time series forecasting
- **Access**: `yfinance` library (Python)

**Air Quality UCI**
- **Size**: 9,358 hourly measurements
- **Task**: Air quality prediction
- **Use for**: Time series forecasting, regression
- **URL**: https://archive.ics.uci.edu/ml/datasets/Air+Quality

**Electricity Load**
- **Size**: 370 consumers, 26,304 time points
- **Task**: Load forecasting
- **Use for**: Time series forecasting
- **URL**: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

**M4 Competition**
- **Size**: 100,000 time series
- **Task**: Forecasting competition
- **Use for**: Time series forecasting benchmarking
- **URL**: https://www.m4.unic.ac.cy/

**ECG5000**
- **Size**: 5,000 ECG recordings
- **Task**: Heartbeat classification
- **Use for**: Time series classification, medical data
- **URL**: http://www.timeseriesclassification.com/

**Bike Sharing**
- **Size**: Daily counts (731 days) and hourly counts (17,379 hours)
- **Task**: Demand prediction
- **Use for**: Regression, time series
- **URL**: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

## Audio and Speech

**LibriSpeech**
- **Size**: 1,000 hours of speech
- **Task**: Speech recognition
- **Use for**: ASR systems
- **URL**: http://www.openslr.org/12/

**Common Voice (Mozilla)**
- **Size**: 9,000+ hours in 60+ languages
- **Task**: Speech recognition
- **Use for**: Multilingual ASR
- **URL**: https://commonvoice.mozilla.org/

**UrbanSound8K**
- **Size**: 8,732 audio clips of urban sounds
- **Task**: Sound classification (10 classes)
- **Use for**: Audio classification
- **URL**: https://urbansounddataset.weebly.com/

**GTZAN Genre**
- **Size**: 1,000 audio tracks (30 sec each)
- **Task**: Music genre classification (10 genres)
- **Use for**: Audio classification, music analysis
- **URL**: http://marsyas.info/downloads/datasets.html

**Speech Commands**
- **Size**: 105,000 utterances of 35 words
- **Task**: Keyword spotting
- **Use for**: Voice command recognition
- **Source**: Google
- **URL**: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

## Healthcare and Biology

**Genomics Datasets**
- **1000 Genomes Project**: Human genetic variation
- **URL**: https://www.internationalgenome.org/

**Cancer Genome Atlas (TCGA)**
- **Task**: Cancer genomics
- **URL**: https://www.cancer.gov/tcga

**PhysioNet**
- **Content**: Medical research data (ECG, EEG, etc.)
- **URL**: https://physionet.org/

**MIMIC-III**
- **Size**: 40,000+ ICU patients
- **Task**: Clinical predictions
- **Use for**: Healthcare analytics
- **URL**: https://mimic.physionet.org/
- **Note**: Requires training and approval

**Drug Review Dataset**
- **Size**: 161,297 reviews
- **Task**: Sentiment analysis, rating prediction
- **URL**: https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset

## Recommender Systems

**MovieLens**
- **Sizes**: 100K, 1M, 10M, 20M, 25M ratings
- **Task**: Movie recommendation
- **Use for**: Collaborative filtering, recommendation systems
- **URL**: https://grouplens.org/datasets/movielens/

**Netflix Prize**
- **Size**: 100M ratings
- **Task**: Movie rating prediction
- **Note**: Original competition, data available with restrictions
- **URL**: https://www.kaggle.com/netflix-inc/netflix-prize-data

**Amazon Product Reviews**
- **Size**: 142.8M reviews
- **Task**: Product recommendation, sentiment analysis
- **URL**: https://nijianmo.github.io/amazon/index.html

**Book-Crossing**
- **Size**: 1.1M ratings on 270K books
- **Task**: Book recommendation
- **URL**: http://www2.informatik.uni-freiburg.de/~cziegler/BX/

**Last.fm**
- **Size**: Music listening data
- **Task**: Music recommendation
- **URL**: https://grouplens.org/datasets/hetrec-2011/

## Geospatial Data

**OpenStreetMap**
- **Content**: Global map data
- **Task**: Geographic analysis, routing
- **URL**: https://www.openstreetmap.org/

**Landsat Satellite Imagery**
- **Content**: Earth observation data
- **Task**: Remote sensing, land use classification
- **URL**: https://www.usgs.gov/landsat-missions

**Natural Earth**
- **Content**: Vector and raster map data
- **URL**: https://www.naturalearthdata.com/

**NYC Taxi Trip Data**
- **Size**: Millions of taxi trips
- **Task**: Demand prediction, route optimization
- **URL**: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

**Uber Movement**
- **Content**: Urban transportation data
- **Task**: Travel time analysis
- **URL**: https://movement.uber.com/

## Dataset Repositories

### Major Repositories

**Kaggle Datasets**
- **URL**: https://www.kaggle.com/datasets
- **Content**: 50,000+ datasets across all domains
- **Features**: Competitions, notebooks, community

**UCI Machine Learning Repository**
- **URL**: https://archive.ics.uci.edu/ml/
- **Content**: 580+ datasets
- **Features**: Classic ML datasets, well-documented

**Google Dataset Search**
- **URL**: https://datasetsearch.research.google.com/
- **Content**: Aggregates datasets from various sources
- **Features**: Search engine for datasets

**Papers with Code Datasets**
- **URL**: https://paperswithcode.com/datasets
- **Content**: Research datasets with benchmark results
- **Features**: Links to papers and code

**AWS Open Data Registry**
- **URL**: https://registry.opendata.aws/
- **Content**: Large-scale datasets on AWS
- **Features**: Free to access, cloud-native

**Microsoft Research Open Data**
- **URL**: https://msropendata.com/
- **Content**: Research datasets from Microsoft
- **Features**: Various domains, cloud storage

**Hugging Face Datasets**
- **URL**: https://huggingface.co/datasets
- **Content**: NLP and multimodal datasets
- **Features**: Easy loading with `datasets` library
- **Load**: `from datasets import load_dataset`

**TensorFlow Datasets**
- **URL**: https://www.tensorflow.org/datasets
- **Content**: Ready-to-use datasets for TensorFlow
- **Load**: `tensorflow_datasets.load()`

**PyTorch Datasets**
- **URL**: https://pytorch.org/vision/stable/datasets.html
- **Content**: Datasets for PyTorch
- **Load**: `torchvision.datasets`

### Government Data

**Data.gov (US)**
- **URL**: https://www.data.gov/
- **Content**: US government open data

**European Data Portal**
- **URL**: https://data.europa.eu/
- **Content**: European open data

**World Bank Open Data**
- **URL**: https://data.worldbank.org/
- **Content**: Development data

**WHO Data**
- **URL**: https://www.who.int/data
- **Content**: Global health data

### Domain-Specific

**OpenML**
- **URL**: https://www.openml.org/
- **Content**: Machine learning datasets and experiments
- **Features**: Reproducible ML research

**Stanford Network Analysis Project (SNAP)**
- **URL**: https://snap.stanford.edu/data/
- **Content**: Network/graph datasets

**Computer Vision Datasets (Roboflow)**
- **URL**: https://public.roboflow.com/
- **Content**: Annotated image datasets
- **Features**: Various formats, preprocessing

**Financial Data (Quandl)**
- **URL**: https://www.quandl.com/
- **Content**: Financial and economic data

## Dataset Selection Tips

### Choosing the Right Dataset

**For Learning:**
- Start small: Iris, Wine, MNIST
- Clear problem definition
- Well-documented
- Standard benchmarks

**For Research:**
- Recent datasets from Papers with Code
- Challenge datasets (Kaggle competitions)
- Domain-specific repositories
- Novel problem formulations

**For Production:**
- Similar to target distribution
- Sufficient size (thousands to millions)
- Representative of real-world scenarios
- Updated regularly

### Data Quality Checklist

- **Size**: Adequate number of samples for your task
- **Balance**: Check class distribution for classification
- **Missing values**: Understand extent and patterns
- **Documentation**: Clear description of features and target
- **License**: Ensure you can use it for your purpose
- **Versioning**: Use specific versions for reproducibility
- **Bias**: Be aware of potential biases in data collection

### Ethical Considerations

- **Privacy**: Ensure data doesn't contain sensitive personal information
- **Consent**: Data collected with proper consent
- **Fairness**: Check for demographic biases
- **Transparency**: Understand data collection methodology
- **Purpose**: Use data only for appropriate purposes

## Loading Datasets in Python

### Scikit-learn

```python
from sklearn.datasets import load_iris, fetch_california_housing

# Built-in datasets
iris = load_iris()
X, y = iris.data, iris.target

# Downloaded datasets
housing = fetch_california_housing()
```

### TensorFlow

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Keras datasets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# TensorFlow Datasets
ds = tfds.load('cifar10', split='train', shuffle_files=True)
```

### PyTorch

```python
import torchvision
from torchvision import datasets, transforms

# MNIST
mnist = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.ToTensor())

# CIFAR-10
cifar = datasets.CIFAR10(root='./data', train=True, download=True,
                         transform=transforms.ToTensor())
```

### Hugging Face

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset('imdb')
dataset = load_dataset('squad', split='train')

# Load from CSV/JSON
dataset = load_dataset('csv', data_files='my_file.csv')
```

### Pandas (CSV/Excel)

```python
import pandas as pd

# CSV
df = pd.read_csv('data.csv')

# Excel
df = pd.read_excel('data.xlsx')

# From URL
url = 'https://example.com/data.csv'
df = pd.read_csv(url)
```

## References

- Kaggle: https://www.kaggle.com/
- UCI ML Repository: https://archive.ics.uci.edu/ml/
- Papers with Code: https://paperswithcode.com/
- TensorFlow Datasets: https://www.tensorflow.org/datasets
- Hugging Face Datasets: https://huggingface.co/datasets
