# udacity-data-scientist-nanodegree-capstone-project

# Blog post

https://medium.com/@jsohi/what-type-of-users-cancel-their-music-subscriptions-1a6a42ccebae

## Motivation

This is the capstone project for the Udacity Data Scientist Nanodegree program. In this project, I used Spark (PySpark) to predict customer churn for a fictional company called Sparkify (with data provided by Insight Data Science). I went through three ML model iterations until I got an acceptable F1 score. This is a measure of model perfomance (a harmonic mix of precision and recall). Traditional accuracy was not a good metric in this case because the dataset was unbalanced (only a few users churn relative to users who do not)

## File/directory tree

```
- README.md # this file
- transform_raw_to_user.py - take the raw data and transform to user level for modeling
- Sparkify.ipynb # working through 1st model iteration with sample data
- final.ipynb # the intermediate and final model selected which is GBT

- data - entire folder is gitignored due to large file size
|- mini_sparkify_event_data.json  # raw data to process
|- TRANSFORMED_mini_sparkify_event_data.csv  # data that has been transformed

- models # different model iterations (different ML classification algorithms, parameters, features, etc)
|- final_model # one folder per model version
```

## Libraries used

* pyspark (v2.4.5)
* plotly
* pandas
* pathLib 

## Analysis results

I found a Gradient Boosting Tree algorithm performed the best with this data and was able to make predictions with an F1 score of: **0.71**
