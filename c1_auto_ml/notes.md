# Notes

## Week 1

Ultimate goal of Data Science -> Knowledge Distillation

- AI = technique for computers to mimic human behaviour
- ML = subset of AI, statistical methods and algos that are able to learn from data, without being explicitly programmed.
- DL = subset of ML, artificual nerual networks to learn from data
- DS = an interdisciplinary field that combines business and domain knowledge with math, statistics, data visualization, and programming skills.
- Practical DS = utilizing the power of cloud to run ds and ml on steroids for massive datasets
- Scale up = boost compute power
- Scale out = distributed model training in parallel across multiple CPUs

Tools

- AWS S3 and Athena -> ingest, query and store data
- AWS Glue Crawlers -> Catalog data in its schema
- AWS Sagemaker Data Wrangler and Clarify -> bias detection
- [AWS data Wrangler](https://github.com/awslabs/aws-data-wrangler)

```sql
SELECT CARDINALITY(SPLIT(review_body, ' ')) as num_words
FROM dsoaws_deep_learning.reviews
```

### Data viz

```python
summary = df["num_words"].describe(
    percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

df["num_words"].plot.hist(
    xticks=[0, 16, 32, 64, 128, 256], bins=100,
    range = [0, 256]).axvline(x=summary["100%"], c="red")
)
```

## Lab

Scenario: Build NLP model which takes product reviews as input and clasify sentiment of reviews into positive, neutral, or negative.

1. Ingest data into Amazon S3 bucket
2. Explore data with Athena and Glue
3. Analyze data with Python and SQL

## Resources

- [Optimize Python ETL by extending Pandas with AWS Data Wrangler](https://aws.amazon.com/blogs/big-data/optimize-python-etl-by-extending-pandas-with-aws-data-wrangler/#:~:text=AWS%20Data%20Wrangler%20is%20an,the%20extraction%20and%20load%20steps)
