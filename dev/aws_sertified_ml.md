## AWS Sertified Machine Learning exam notes

#### S3
There are few types of S3 services:
 - standard
 - IA
 - One-zone IA

Glacier:
 - Instant retrieval (millisecond retrieval, minimum storage duration - 90 days)
 - Flexible retrieval (minimum storage duration - 90 days)
    - Expedited (1-5 minutes)
    - Standard (3-5 hours)
    - Bulk (5-12 hours) - free retrieve
 - Deep archive (minimum storage duration - 180 days)
    - Standard (12 hours)
    - Bulk (48 hours)

Intelligent Tiering
 - Frequent Access (automatic) (default)
 - Infrequent Access (automatic) (for objects not requested for 30 days)
 - Archive Instant Access Tier (automatic) (for objects not requested for 90 days)
 - Archive Access Tier (optional) (for objects not requested from 90 days to 700+ days)
 - Deep Archive Access Tier (optional) (for objects not requested from 180 days to 700+ days)


#### Kinesis Data Stream
For data streaming (like Apache Kafka)

#### Kinesis Data Firehose
For data ingestion (putting data into different sources: S3, Redshift, Elasticsearch)

#### Kinesis Data Analytics
 - RANDOM_CUT_FOREST - for anomaly detection on numeric columns (uses recent history)
 - HOTSPOT - detects relatively dense regions in your data

Two types:
 - SQL
 - Apache Flink - more powerful

#### Glue
Glue Data Catalog
 - metadata repository for all tables
 - automated schema inference
 - schemas versioning
 - integration with athena, redshift spectrum

Glue Data Crawlers
 - go over your data to infer schemas
 - works with JSON, parquet, CSV, ...

Glue ETL
 - transform data, clean data, enrich data
   - generate code in python or scala, you can modify it
   - can provide your own spark or pyspark scripts
   - target can be S3, RDS, Redshift, or Glue data catalog
 - fully managed, pay for resources
 - jobs are run on spark serverless

Glue ETL transformations:
 - DropFields, DropNullFields
 - Filter
 - Join
 - Map

ML Transformations:
 - find matches

Format conversions

Glue is based on the Apache Spark

#### AWS Data Pipeline

For moving data
Destinations:
 - S3
 - RDS
 - DynamoDB
 - Redshift
 - EMR
It is orchestration service
With data pipeline you can move data from your RDS or DynamoDB to S3 or another target
It is run on the EC2 (or EMR) and you can control it (you have more deep control)


#### AWS Batch
Based on Docker images
 - dynamic provision of instances

(serverless: you don't provision or control resources, just pay for underlying EC2 instances)

### Data Engineering Summary
Here's a quick summary of all the services we've mentioned

 * Amazon S3: Object Storage for your data
 * VPC Endpoint Gateway: Privately access your S3 bucket without going through the public internet
 * Kinesis Data Streams: real-time data streams, need capacity planning, real-time applications
 * Kinesis Data Firehose: near real-time data ingestion to S3, Redshift, ElasticSearch, Splunk
 * Kinesis Data Analytics: SQL transformations on streaming data
 * Kinesis Video Streams: real-time video feeds
 * Glue Data Catalog & Crawlers: Metadata repositories for schemas and datasets in your account
 * Glue ETL: ETL Jobs as Spark programs, run on a serverless Spark Cluster
 * DynamoDB: NoSQL store
 * Redshift: Data Warehousing for OLAP, SQL language
 * Redshift Spectrum: Redshift on data in S3 (without the need to load it first in Redshift)
 * RDS / Aurora: Relational Data Store for OLTP, SQL language
 * ElasticSearch: index for your data, search capability, clickstream analytics
 * ElastiCache: data cache technology
 * Data Pipelines: Orchestration of ETL jobs between RDS, DynamoDB, S3. Runs on EC2 instances
 * Batch: batch jobs run as Docker containers - not just for data, manages EC2 instances for you
 * DMS: Database Migration Service, 1-to-1 CDC replication, no ETL
 * Step Functions: Orchestration of workflows, audit, retry mechanisms

**Briefly mentioned, covered by Frank Kane:**
 * EMR: Managed Hadoop Clusters
 * Quicksight: Visualization Tool
 * Rekognition: ML Service
 * SageMaker: ML Service
 * DeepLens: camera by Amazon
 * Athena: Serverless Query of your data

#### AWS QuickSite
There is Spice under the hood
ML possibilities:
 - anomaly detection
 - forecasting
 - auto-narratives

Anti-patterns:
 - highly-formatted canned reports
 - ETL

QuickSite is a good choice for ad-hoc queries, analyzis and visualization


#### AWS EMR
Managed Hadoop framework based on EC2
Includes:
 - spark
 - Hbase
 - Preso
 - Flink
 - Hive
 - EMR Notebooks
 - ...

**EMR cluster consists of:**
 - master-node - manages tasks
 - core-node - hosts HDFS data and run tasks 
 - task-node - run tasks (good choice for spot-instances)

Cluster can be:
 - transient - good way for spot-instances
 - long-running - good way for reserved instances

**EMR storages:**
 - HDFS (distributes data between EC2 nodes, default chunk-size: 128Mb), good for performance
 - EMRFS - access S3 if it is were HDFS
 - Local filesystem (instant store)
 - EBS for HDFS

Can add and remove nodes automatically

#### Hadoop
 * Mapreduce <-> Spark (alternative)
 * YARN (yet another resource negotiator)
 * HDFS
(Spark is newer, smarter and more effective)

Spark components:
 - spark core
 - spark streaming
 - spark SQL
 - MLlib
 - GraphX

**Zeppelin+Spark**
 - allows to run spark tasks interactively from browser
 - can execute SQL queries direcly from SparkSQL
 - like jupyter

**EMR Notebook**
Similar to Zeppelin, but has more integrations with AWS
- provision clusters from notebook
- notebooks backed up to S3


#### Feature engineering
SMOTE - Synthetic minority over-sampling technique - generate synthetic examples of some classes using nearest neighbors

AWS Random cut forest - for outliers detection/cleaning

#### Sage maker Ground Truth
 - it like Yandex Toloka - for data labeling
 - also it creates its own model and tries to label data. If it is not confident in class, it will assign the example to the assesor

#### Activation functions
 - classification -> softmax
 - RNN's -> Tanh
 - anything else:
   - start with RELU
   - try leaky RELU
   - PRELU
   - Swish for really deep networks

Specialized CNN architectures:
 - **LeNet5** - good for handwriting recognition
 - **AlexNext** - image classification
 - **GoogLeNet** - image classification (deeper, better performance)
 - **ResNet** - image classification (deeper, better performance)

RNN for:
 - time-series
 - music
 - texts (translation)

**RNN topologies:**
 - *Sequence-to-Sequence* - predict stock prices based on historical data
 - *Sequence-to-vector* - words in a sentence to a sentiment
 - *Vector-to-Sequence* - create caption from an image
 - *Encoder-Decoder* - Sequence->Vector->Sequence (translation)

LSTM - long-short term memory
GRU - Gated Recurrent Unit

**Huggins Face** - a collection of pre-trained models

**Transfer learning approaches:**
 - continue training pretrained model (change last freezed layer)
 - add new layer
 - retrain from scratch

EMR supports Apache MXNet

Appropriate instances for deep learning:
 - P3: 8 Tesla V100
 - P2 16 K80
 - G3: 4 M60

**Batch size**
 - small batch sizes tend to not get stuck in local minima
 - large batch sizes can converge at wrong solution at random
 - large learning rates can overshoot the correct solution
 - small learning rates increse learning time

#### Regularization techniques
In case of overfitting you can try to do your model simpler
 - Dropout
 - Early stopping (if we have achieved enough accuracy, we can don't execute all epochs)
 - L1 - sum of weights
   - performs feature selection
   - computationally inefficient
   - sparse output
   - better to use when some features can be not important
 - L2 - sum of squared weights
   - all features remain considered, just weighted
   - computationally efficient
   - dense output

#### Vanishing gradients problem
 - small changes - very low step size, in deep deep networks the step value may be less and less in deeper layers
 - too big changes - 

**Solutions**
 - break your network into few layers and train each layers separately
 - LSTM - can tackle this problem
 - ResNet - designed to deal with this problem
 - ReLu - better choise for activation function

**Gradient checking** - debugging technique
 - numerically check gradients computed during training

#### Confusion Matrix
just heatmap

#### Metrics list
**Recall** - percent of positives rightly predicted (completness, sensetivity, true positive rate)
Recall = TP / (TP + FN)
Good choice for:
 - fraud detection

**Precision** - percent of relevant results (correct positives, when you care about false positives)
Precision = TP / (TP + FP)
Good choice for:
 - medical screening
 - drug testing

**Specificity** - true negative rate
Specificity = TN / (TN + FP)

**F1-score**
F1 = 2FP / (2TP + FP + FN)
or
F1 = 2 * (Precision*Recall / (Precision + Recall))

**RMSE** - for numeric values

**ROC curve** - Receiver Operating Characteristic curve. Plot of true positive rates vs false positive rates at various threshold settings
The more ROC is bent to the left upper corner, the better

**AUC** - the area under ROC curve. Equals to probability that a classifier will rank a randomly chosen positive instance higher then a randomly chosen negative one.
ROC-AUC = 0.5 - useless classifier
ROC-AUC = 1.0 - perfect classifier
Commonly used to compare classifiers

#### Ensemble methods
 - Bagging
 - Boosting

#### SageMaker
Can be used for:
 - fetch, clean and prepare data
 - train and evaluate a model
 - deploy model, evaluate results in production

Input formats:
 - RecordIO protobuf
 - CSV
 - File or Pipe

Models:
 - **XGBoost**
 - **Seq2Seq** (can't use many machines, but can use multiple GPUs)
 - **DeepAR**
   - for forecasting time-series data
   - finds sequences and seasonality)
   - always include entire data for training, testing and inference
   - hyperparamethers (context_length, ...)
   - can use CPU and GPU
   - single or multimachine
 - **BlazingText**
   - for text classification
   - word2vec
 - **Object2Vec**
   - only one machine
   - supports multiple GPU
 - **Object Detection**
   - finds bounds, classes and confidence level
   - uses a CNN with Single Shot multibox Detector (SSD) algorithm
   - supports multi-machines, multi-GPU
 - **Image Classification**
   - for full images classes
   - uses ResNet under the hood
   - multi-GPU
   - multi-machines
 - **Semantic Segmentation**
   - pixel-level object classification
   - produces a segmentation mask
   - built on MXNet Gluon and Gluon CV
   - choice for 3 algorithms:
     - fully-convolutional network (FCN) 
     - pyramid-scene-parsing (PSP)
     - DeepLabV3
   - only GPU nodes
   - single machine
   - use ml.*.xlarge EC2 instances
 - **Random Cut Forest**
   - for anomaly detection
   - unsupervised
   - detects unexpected spikes in time-series data, breaks in periods, unclassifiable datapoints
   - assigns anomaly score to each point
   - hyperparameters:
      - num_trees
      - num_samples_per_tree
   - no-GPU, no multi-machine, very easy algorithm
 - **Neural Topic Model**
   - to organize documents into topics
   - classify or summarize documents based on topics
   - unsupervised (neural variational inference)
   - hyperparameters:
     - topics number
   - GPU or CPU
 - **LDA** - Latent Dirichlet Allocation
   - unsupervised
   - can be used for things other than words (customers clusterization, ...)
   - you specify number of topics
   - CPU-based
 - **KNN**
   - supervised (predict class by classes of nearest neighbors)
   - can be used for regression
   - CPU or GPU
   - CPU - for law latency
   - GPU - for high throughtput
 - **K-means**
   - unsupervised clustering
   - CPU or GPU
 - **PCA**
   - dimensionality reduction
   - unsupervised
   - two modes: regular (for sparse data) and randomized (for large number of observations - uses approximation approach)
   - CPU or GPU
 - **Factorization Machines**
   - classification or regression sparse data
   - use cases: recommender system
   - supervised
   - limited to pair-wise interactions
   - data format: RecordIO-protobuf
   - CPU - recommended or GPU - works only with dense data
 - **IP Insights**
   - usupervised learning of IP adresses patterns
   - identifies suspisious behaviors from IP adresses
   - input: csv only
   - uses neural network under the hood
   - automatically generates negative pairs during training
   - CPU or GPU, multi-GPU
 - **Reinforcement Learning**
   - terms: environment, state, action, Q-function, reward, observation
   - Markov-desicion-process provides a framework for modeling decision making in situations where outcomes are partly random and partly under the control (MDP is a discrete time stochastic control process)
   - SageMaker uses deep-learning with tensorflow and MXNet
   - multi-machine, multi-core
   - 
 - **Automatic Model Tuning**
   - you define ranges of hyperparameters and metrics
   - it learns as it goes, so it doesn't need to try every possible combination of parameters
   - don't try to optimize too many hyperparameters at once
   - limit parameters ranges to small ranges
   - use logarithmic scales when it's possible
   - don't run too many training jobs concurrently
   - check the conrrectness of the metric
 - **SageMaker Debbugger**
   - supports:
     - tensorflow
     - PyTorch
     - MXNet
     - XGBoost
     - SageMaker generic estimator
 - **SageMaker Autopilot** (AutoML)
   - automates:
     - algorithm selection
     - data preprocessing
     - model tuning
     - all infastructure
   - also you can choose:
     - do all (clean data, create model, train, ...)
     - create a template with recommendations for you
   - problem types:
     - binary classification
     - multiclass classification
     - regression
   - algorithm types:
     - Linear Learner
     - XGBoost
     - Deep Learning
   - input: CSV
   - integrates with SageMaker Clarify
 - **SageMaker Model Monitor**
   - get alerts on quality deviations
   - visualize data-drift
   - detect anomalies and outliers
   - detect new features
   - integrates with SageMaker Clarify
   - integrates with tensorboard, QuickSight and Tableu
 - **SageMaker JumpStart**
   - for creating model from ready standard use-case
 - **SageMaker Data Wrangler**
   - import, transform, analyze, exoirt data with SageMaker Studio
 - **SageMaker Feature Store**
   - find and discover features with Studio
   - online/offline
 - **SageMaker Edge Manager**
   - for edge devices
 - **SageMaker Canvas**
   - no-code solution
   - for classification or regression
   - automatic data cleaning
 - **SageMaker Training Compiler**
   - compiles and optimizes training jobs for GPUs
   - converts models into hardware optimized instructions


#### Services
 - **Textract**
   - OCR
   - extract text from the image
   - supports tables, forms
 - **Kendra**
   - enterprise search with natural language
   - combines data from file-systems, share-point, intranet, S3, ...
 - **Comprehend**
   - for NLP and text-analytics
   - extract key phrases, entities, sentiments, topics, language and document classifications
   - can be trained on your data
 - **Translate**
   - translates text
   - can use custom terminology (you can upload your dictionary)
 - **Transcribe**
   - speech-to-text
   - transcribes all the file (mp3, flac, ...)
   - can recognize number of speakers
   - automatoc language determination
   - custom vocabularies
   - there is streaming transcribing
 - **Polly**
   - text-to-speech
   - many voices and languages
   - you customize pronunciation of specific terms
   - supports speech marks (useful for lips synchronization)
   - Lexicons - for customizing pronunciations
 - **Rekognition**
   - computer-vision
   - object and scene detection
   - facial analyzing
   - image moderation
   - face comparison
   - text in image
   - video analysis
     - people pathing
   - custom labels (provide your traning set)
 - **Forecast**
   - time-series analysis
   - algorithms:
     - *CNN-QR* (Quantile Regression)
       - best for large datasets with hundreds of data-series
       - accepts related historical time series data & metadata
       - computationally exprensive 
     - *DeepAR+*
       - recurrent neural network
       - best for large datasets
       - accepts related forward-looking timeseries & metadata
     - *Prophet*
       - additive model with non-linear trends and seasonality
     - *NPTS* (non-parametric time-series)
       - good for sparse data. Has variants for seasonal / climatological forecasts
     - *ARIMA* (autoregressive integrated moving average)
       - commonly used for simple datasets (<100 timeseries)
     - *ETS* (exponential smoothing)
       - commonly used for simple datasets (<100 
 - **Lex**
   - natural language chat-bot
   - based on utterants and intents
   - lambda functions are called to fulfill the intent
   - slots can be used to get extra information
 - **Personalize**
   - recommendation engine
   - black-box
   - you must provide an explicit schema in Avro format
   - two main APIs:
     - getPersonolizedRecommendations (recommended products, simital items)
     - getPersonolizedRanking (rank a list of provided records)
   - realtime or batch
   - contextual recommendations
   - intelligent user segmentation
   - hyperparameter optimization
 - **DeepRacer**
   - very specific thing
   - for racing cars
   - reinforcement learning
 - **DeepLense**
   - deep learning-enabled video-camera
   - edge-device
 - **Lookout**
   - detecting defects
   - fully automated service
 - **Monitron**
   - end-to-end system for industrial monitoring
 - **TorchServe** 
   - for deploying PyTorch models
 - **Neuron**
   - SDK for ML inference specifically on AWS Inferentia chips
 - **Panorama**
   - brings computer vision to your IP-cameras
 - **Fraud Detector**
   - upload your custom data
   - access risks from new accounts
   - for online payments
 - **CodeGuru**
   - automatic code-review
 - **Contact Lens for Amazon Connect**
   - support for call-centers
   - ingests audio data from recorded calls
   - sentiment analysis
   - categorize calls automatically
   - theme detection
 - **Augmented AI**
   - human review for ML predictions
   - builds workflows for reviewing low-confidence predictions
   - integrated with Textract and Rekognition
   - integreates with SageMaker

#### Examples:
 - build you Alexa: Transcribe => Lex => Polly
 - Make a universal translator: Transcribe => Translate => Polly
 - Build a Jeff Bezos Detector: Rekognition
 - are people on the phone happy: Transcribe => Comprehend

**SageMaker Neo**
 - train once, run anywhere
 - for compiling and running code at the edge devices
 - optimizes code for specific devices

**Elastic Inference**
 - sidecar GPU instance for inferencing
 - allows to low costs
 - works for deep learning (Tensorflow, PyTorch, MxNet)

SageMaker maintains auto-scaling
 - based on target metrics

**SageMaker Serverless**
 - serverless for your models

**SageMaker Inference Recommender**
 - recommends optimal instances for inferencing
 - automates load testing
 - instance recommendations - takes about 45 minutes
 - endpoint recommendations - takes about 2 hours (customized load test)

**Inference Pipeline**
 - linear sequence 2-15 containers