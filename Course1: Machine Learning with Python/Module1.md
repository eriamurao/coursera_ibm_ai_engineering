### Introduction to Machine Learning

### Machine Learning in Action

#### An Overview of Machine Learning

Artificial Intelligence (AI)

- simulates cognitive abilities of humans
- comprises of:
  - Natural Language Processing
  - Computer Vision
  - Generative AI
  - Machine Learning
  - Deep Learning

Machine Learning (ML)

- subset of AI that uses algorithms
- requires feature engineering

Deep Learning (DL)

- a ML algorithm that uses multi-layered neural networks
- automatically extracts features from big data

How ML Works

- teaches computers to:
  - learn from data
  - identify patterns
  - make decisions
- algorithms:
  - use computation methods for learning
  - does not rely on a fixed algorithm

ML Paradigms

- *Supervised Learning* models train on labeled data
- *Unsupervised Learning* works without labels
- *Semi-supervised Learning* works iteratively
- *Reinforcement Learning* learns from feedback

ML Techniques

- *Classification* predicts class or category of a class
- *Regression* predicts continuous values
- *Clustering* groups similar data (Unsupervised Learning technique)
- *Association* finds items or events that co-occur
- *Anomaly Detection* discovers abnormal and unusual cases
- *Sequence Mining* predicts next event
- *Dimension Reduction* reduces size of data or features
- *Recommendation Systems* associate people's preferences

Applications of ML

- Detect if a patient has a benign or malignant tumor
- Netflix and Amazon generating suggestions
- Banks to decide whether to approve or deny loans
- Telecommunication companies to segment customers
- Recognize animals in images

#### Machine Learning Model Lifecycle

- Problem Definition
- Data Collection
  - Extract, Transform, Load (ETL) process
- Data Preparation
- Model Development and Evaluation
- Model Deployment

**The model is iterative**

#### A Day in the Life of Machine Learning Engineer

- Problem Definition
  - Purchased
  - Recommended
- Data Collection
  - User data
  - Production data
  - Other data
- Data Preparation
  - Data have errors, different formatting, missing
  - Clean the data
  - Remove extreme values
  - Fix format
  - Create new features
  - Exploratory data analysis
  - split data into training and test sets
- Model Development and Evaluation
  - explore existing frameworks
  - content-based filtering
  - collaborative filtering
  - test model with unseen data
- Model Deployment
  - continue to track model performance
  - retrain model based on new information

#### Data Scientist VS AI Engineer

Data Scientist

Use Case: Data storyteller

- use descriptive use case by EDA or clustering
- use predictive use case by regression and classification

Data: structured data or tabular data

- 100s to 100k data
- involves cleaning the data or adding new features

Models: many models that they choose

- scope is narrow
  - less params, less compute, and less train time

Process: Use Case > Data > Model > Deploy

- topics like FE, CV, HPT

AI Engineer

Use case: AI systems builder

- prescriptive use case by decision optimization and recommendation eng
- generative use cases by intelligence assistance and chatbots

Data: unstructured data

- billions and trillions tokens

Model: Foundation Model

- scope is wide
  - more params, more compute, and more train time

Process: Model > Prompt Engineering > Embed

- topics like Chaining, PEFT, RAG, Agents

#### Tools for Machine Learning

Data

- a collection of raw facts, figures, or information
- used to draw insights, inform decisions, and fuel advanced technologies
- central to every ML algorithm
- used to discover patterns and make predictions

ML Tools

- provide functionalities for ML pipelines
- includes modules for:
  - data preprocessing and building
  - evaluating
  - optimizing
  - implementing ML models
- examples:
  - Pandas library: data manipulation and analysis
  - Scikit-learn library: supervised and unsupervised learning algorithms for linear regression

Machine Learning Programming Language

- a programming language for building ML models and decoding hidden patterns in data
- examples:
  - Python
    - analyzing and processing data
    - developing ML models
  - R
    - statistical learning
    - data exploration and ML
  - Julia
    - parallel and distributed numerical computing support
  - Scala
    - processing big data and building ML pipelines
  - Java
    - supporting scalable ML application
  - JavaScript
    - running ML models in web browsers

Uses of machine learning tools:

- store and retrieve data
- work with plots, graphs and dashboards
- explore, visually inspect and clean data
- prepare data for developing ML models

Types of machine learning tools

- Data processing and analysis
  - process, store, and interact with data
  - examples:
    - Postgres: relational database management system
    - Hadoop: batch processing solution for big data
    - Spark: distributed data processing framework
    - Apache Kafka: distributed real-time streaming analytics
    - Pandas: data wrangling and preprocessing
    - Numpy: numerical computing tools
- Data visualization
  - understand and visualize data structure
  - examples:
    - Matplotlib: customizable plots and interactive visualizations
    - Seaborn: attractive statistical graphics
    - ggplot2: building and adding elements in layers
    - Tableau: interactive data visualization dashboards
- Machine learning ecosystem
  - create and tune ML models
  - examples:
    - NumPy: numerical computations on data arrays
    - Pandas: data analysis, visualization, cleaning, and preparation
    - SciPy: computing for optimization, integration and linear regression
    - Scikit-learn: suite of classification, regression, clustering and dimensional reduction
- Deep learning
  - designing, training and testing neural network-based models
  - examples:
    - TensorFlow: numerical computing and large-scale ML
    - Keras: implementing neural networks
    - Theano: defining, optimizing, and evaluating mathematical expressions involving arrays
    - PyTorch: computer vision, NLP, and experimentation
- Computer vision
  - object detection
  - image classification
  - facial recognition
  - image segmentation
  - examples:
    - OpenCV: Real-time computer vision applications
    - Scikit-image: image processing algorithms
    - TorchVision: popular data sets, image loading, pre-trained deep learning architectures, and image transformations
- Natural language processing
  - help build applications that understand, interpret, and generate human language
  - examples:
    - NLTK: Text processing, tokenization, and stemming
    - TextBlob: Part of speech tagging, noun phrase extraction, sentiment analysis, and translation
    - Stanza: pre-trained models for tasks such as part of speech tagging, named entity recognition and dependency parsing
- Generative AI
  - leverage AI to generate new content based on input data
  - examples:
    - Hugging Face transformers: text generation, language translation, and sentiment analysis
    - ChatGPT: Text generation, building chatbots, etc.
    - DALL-E: generating images from text descriptions
    - PyTorch: Uses deep learning to create generative models such as GANs and transformers

#### Scikit-learn Machine Learning Ecosystem

ML Tools:

- Data collection
- Preprocessing
- Model training
- Model evaluation
- Model deployment
- Monitoring

ML Ecosystem

- interconnected tools, frameworks, libraries, platforms, and processes
- supports developing, deploying, and managing ML models

Python tools and libraries

- Numpy:
  - foundational ML support with computations on data arrays
- Pandas:
  - data analysis, visualization, cleaning and preparing for ML using DataFrames
- SciPy:
  - scientific computing for optimizations, integration, and linear regression
- Matplotlib:
  - customizable tools for visualizations
- Scikit-learn:
  - building classical ML models

Scikit-learn

- free ML library for Python
- classification, regression, clustering, and dimensionality reduction algorithms
- designed to work with Numpy and SciPy
- Excellent documentation and community support
- Constantly evolving
- Enables easy implementation of ML models

ML Pipeline Tasks

- all pipeline tasks are implemented in scikit-learn:
  - data preprocessing tasks
  - train or test splitting
  - model setup and fitting
  - hyperparameter tuning with cross-validation
  - prediction
  - evaluation
  - exporting the model to be used in production
