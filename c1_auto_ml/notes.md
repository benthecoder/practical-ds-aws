# Notes

- [Notes](#notes)
  - [Week 1: Explore use case and analyze dataset](#week-1-explore-use-case-and-analyze-dataset)
    - [Tools](#tools)
    - [Snippets](#snippets)
  - [Week 2: Statistical Bias](#week-2-statistical-bias)
    - [Causes](#causes)
    - [Metrics for bias](#metrics-for-bias)
    - [Feature importance: SHAP](#feature-importance-shap)
  - [Resources](#resources)

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

## Resources

- [Optimize Python ETL by extending Pandas with AWS Data Wrangler](https://aws.amazon.com/blogs/big-data/optimize-python-etl-by-extending-pandas-with-aws-data-wrangler/#:~:text=AWS%20Data%20Wrangler%20is%20an,the%20extraction%20and%20load%20steps)
