# Notes

- [Notes](#notes)
  - [Week 1: Explore use case and analyze dataset](#week-1-explore-use-case-and-analyze-dataset)
    - [Tools](#tools)
    - [Resources](#resources)
    - [Snippets](#snippets)
  - [Week 2: Statistical Bias](#week-2-statistical-bias)
    - [Causes](#causes)
    - [Metrics for bias](#metrics-for-bias)
    - [Feature importance: SHAP](#feature-importance-shap)
  - [Week 3 : AutoML](#week-3--automl)
    - [Model Building challenges](#model-building-challenges)
    - [AutoML](#automl)
    - [TF-IDF](#tf-idf)
      - [TF - Term Frequency](#tf---term-frequency)
      - [IDF - Inverse-document frequency](#idf---inverse-document-frequency)
    - [Sagemaker autopilot](#sagemaker-autopilot)
  - [Week 4: Built-in algorithms](#week-4-built-in-algorithms)
    - [Use cases](#use-cases)
    - [Evolution of text analysis algorithms](#evolution-of-text-analysis-algorithms)

## Week 1: Explore use case and analyze dataset

> Ultimate goal of Data Science -> Knowledge Distillation

- AI = technique for computers to mimic human behaviour
- ML = subset of AI, statistical methods and algos that are able to learn from data, without being explicitly programmed.
- DL = subset of ML, artificual nerual networks to learn from data
- DS = an interdisciplinary field that combines business and domain knowledge with math, statistics, data visualization, and programming skills.
- Practical DS = utilizing the power of cloud to run ds and ml on steroids for massive datasets
- Scale up = boost compute power
- Scale out = distributed model training in parallel across multiple CPUs

### Tools

- boto3 -> AWS SDK for python to create, configure and manage AWS
- AWS S3 and Athena -> ingest, query and store data
- AWS Glue Crawlers -> Catalog data in its schema
- AWS Sagemaker Data Wrangler and Clarify -> bias detection
- [AWS data Wrangler](https://github.com/awslabs/aws-data-wrangler)

### Resources

- [Optimize Python ETL by extending Pandas with AWS Data Wrangler](https://aws.amazon.com/blogs/big-data/optimize-python-etl-by-extending-pandas-with-aws-data-wrangler/)

### Snippets

```sql
SELECT CARDINALITY(SPLIT(review_body, ' ')) as num_words
FROM dsoaws_deep_learning.reviews
```

```python
summary = df["num_words"].describe(
    percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

df["num_words"].plot.hist(
    xticks=[0, 16, 32, 64, 128, 256], bins=100,
    range = [0, 256]).axvline(x=summary["100%"], c="red")
)
```

## Week 2: Statistical Bias

Training data is biased if it does not comprehensively represent the problem space

Why? Some elements of dataset more heavily weighted/represented

### Causes

- Activitiy bias
  - human generated content like social media
  - very small percentage of population are active on social media, so data collected is not representative of everyone
- Societal bias
  - introduced because of preconceived notion that exist in society, all of us have unconscious bias
- Selection bias
  - user selection is used as training data to further improve the model -> feedback loops
  - I watch one kdrama on netflix because it's about startups, and now netflix is recommending me tons of kdramas, when in reality I watched the show because I like startups.
- Data drift (shift)
  - happens when data distribution significantly varies from training data used to train the model
  - ex: loan approval ML system breaking years after deployment because interest rates have risen significantly
  - types
    1. covariant drift: distribution of features change
    2. prior probability drift: distribution of target change
    3. concept drift: relationship between features and target change

Statistical Bias = tendency for a measure to over/under estimate a parameter

Tools to detect statistical bias

- sagemaker data wrangler
- sagemaker clarify

### Metrics for bias

> Properly measuring statistical data bias is key to build fair and successful ML models.

Difference in proportion of labels (DPL)

- detect an imbalance of positive outcomes between different facet values.
- ex: does products with the category "accecories" have higher positive ratings than products with category "tech"?

Class Imbalance (CI)

- Measures the imbalance in the number of members between different attribute values
- ex: building a sentiment classifier to determine whether product reviews have positive, neutral or negative sentiments. From the star rating column, there is a disproportionate amount of the ratings are five stars (50% of the total ratings).
- Models trained on an unequal class distribution like this one tend to be biased towards the majority class which can be harmful when the classification of minority classes are more valued than that of the majority class

[More metrics for measuring pretraining bias](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.html)

### Feature importance: SHAP

- Explain individual features that make up the dataset
- evaluate how useful/valuable the feature is relative to other feature

Shapley values based on game theory

Allow you to attribute outcome of game to individual players of game

Transfer that to ML, individual players are the features and outcome is the ml prediction

You can provide local vs global explanations

- Local: how individual feature contribuets to final model
- Global: How data in it's entirety contributes to outcome

It considers all possible combinations of feature values + outcomes of ML model. Time intensive but can guarantee consistency and local accuracy

[shap docs](https://shap.readthedocs.io/en/latest/)

## Week 3 : AutoML

### Model Building challenges

> nature of ML makes it difficult to iterate quickly (limited by compute/human resources)

- creating ML model involves multiple iterations that increases time-to-market
- requires specialize skill sets from existing teams
- ML experiments take much longer than traditional development life cycles (long time to get model performance and run experiments)

### AutoML

- reduce time-to-market
- lower ML barrier
- iterate quickly with automation
- save resources for more challenging use-cases

### TF-IDF

- Evaluates how relevant a word is to a document/collection of documents

Bag-of-Words is typical approach of putting words in bag -> sample without replacement, tally the count of words.

To measure word importance:

- increase proportionally to number of times word appear in document
- decrease proportionally by the frequency of word in corpus

#### TF - Term Frequency

Idea: Relative frequency of a word in a document, against all other words in the document

$tf(t, d) = \frac{f_{t, d}}{\Sigma_{t' \in d} f_{t' \in d}}$

t - term, d - document, D - corpus

It's possible that terms would appear more in longer documents that shorter ones.

#### IDF - Inverse-document frequency

Idea: Look at whole corpus to measure how important is a term

$idf(t, D) = log(\frac{|D|}{|d \in D : t \in d})$

Common words like "if", "the", will appear many times but they have little importance, so you scale down the weight of frequent terms while you scale up the rare ones.

Caveat: might lead to division by zero

example:

Document containing 200 words, the word "food" appear 10 times.

tf("food", d) = 10/200 = 0.05

If corpus has 1 million documents, and word "food" appear in 1000 of these documents

idf("food", D) = log( 1M / 1000 ) = 3

So, tf-idf weight = 0.05 \* 3 = 0.15

### Sagemaker autopilot

- Sagemaker's implementation of AutoML
- has full transparency into data transformation, hyperparameter (provides notebooks for experiments) feature importance, model performance, etc.

Running SageMaker AutoPilot generates two notebooks:

- Data exploration
- Candidate definition

## Week 4: Built-in algorithms

### Use cases

1. Feature Engineering (Dimensionality reduction)
   - drop weak features (color for predicting mileage)
   - PCA: reduce number of features while retaining as much info as possible
2. Binary/multi-class classification
   - predict labels (spam or not spam)
   - XGBoost: Extreme Gradient Boosting
     - Boosting: a sequential process where each model attempts to correct errors of previous models (assign more weights to false classifications), to form a strong learner at the end
     - performs really well because of its robust handling of a variety of data types, relationships, and distributions, and the variety of hyperparameters that you can fine tune.
3. Anomaly Detection
   - Detect abnormal behaviour
   - Random cut forest (RCF)
     - an unsupervised algorithm for detecting anomalous data points, it assings an anomaly score for each data point (low being normal and high being an anomaly)
4. Clustering
   - group customers in spending groups from transactions
   - K-means: unsupervised learning model that group similar objects together into k clusters where each data point belong to the cluster with the nearest mean
5. Topic Modelling
   - Latent Dirichlet Allocation: generative probability model that attempts to provide a model for the distribution of input and outputs based on latent varirables.
   - Latent variables are variables that are not directly observed but inferred from other variables (direct opposite of discriminant models which learns how input map to outputs)
   - Neural Topic Modelling

### Evolution of text analysis algorithms

1. Simpel bag of words model (1950s)
1. [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf) (2013)
   - converts text into vectors (embeddings)
   - use as inputs to ML algos - KNN or clustering
   - model architectures:
     - continuous bag-of-words (CBOW) : predict current word from window of surrounding context words
     - continuous skip grams : use current word to predict surrounding context words
   - Challenge: out of vocabulary issue: for words that are not present in text data, model will discard it
1. [GloVe](https://aclanthology.org/D14-1162.pdf) (2014)
   - use regression to learn word representation through unsupervised learning
1. [FastText](https://arxiv.org/pdf/1607.04606v2.pdf)
   - Extension of Word2Vec, breaks word into character sets of length n (n-grams) which helps with out of vocabulary issue
   - ex: "amazon" -> "a", "am", "ama", "amaz", "amazo", "amazon"
   - embedding for a word is the aggregate of the embedding of each n-gram with the word
1. [Transformers](https://arxiv.org/abs/1706.03762) (2017)
   - concept of attention refers to one model component capturing correlation between input and output
   - attention would macp each word from model output to words in input sequence, assigning weights depending on importance towards predicted word
   - self-attention mechanism focuses on capturing relationship between all words in the input sequence and significantly improved NLU tasks
1. [BlazingText](https://dl.acm.org/doi/pdf/10.1145/3146347.3146354) (2017)
   - highly optimized implementation of Word2Vec and text classiifcation algorithms with multiple CPU or GPUs
1. [ElMo](https://arxiv.org/pdf/1802.05365v2.pdf) (2018)
   - word vectors are learned from deep bidirectional language model
   - combines forward and backward language model - better captures syntax and semantics across different liguistic context
1. [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018)
   - trained on large unlabled text corpus
   - predicts from left to right, uni-directional
1. [BERT](https://arxiv.org/abs/1810.04805) (2018)
   - bidirectional
   - learns representation from unlabeled text from L to R and R to L
