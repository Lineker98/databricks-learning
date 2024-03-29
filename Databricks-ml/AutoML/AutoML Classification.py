# Databricks notebook source
from pyspark.sql.types import DoubleType, StringType, StructType, StructField

schema = StructType([
  StructField("age", DoubleType(), True),
  StructField("workclass", StringType(), True),
  StructField("fnlwgt", DoubleType(), True),
  StructField("education", StringType(), True),
  StructField("education_num", DoubleType(), True),
  StructField("marital_status", StringType(), True),
  StructField("occupation", StringType(), True),
  StructField("relationship", StringType(), True),
  StructField("race", StringType(), True),
  StructField("sex", StringType(), True),
  StructField("capital_gain", DoubleType(), True),
  StructField("capital_loss", DoubleType(), True),
  StructField("hours_per_week", DoubleType(), True),
  StructField("native_country", StringType(), True),
  StructField("income", StringType(), True)
])
census_df = spark.read.format("csv").schema(schema).load("/databricks-datasets/adult/adult.data")

# COMMAND ----------

census_df.display()

# COMMAND ----------

census_df.count()

# COMMAND ----------

census_df.columns

# COMMAND ----------

train_df, test_df = census_df.randomSplit([0.99, 0.01], seed=42)

# COMMAND ----------

display(train_df)

# COMMAND ----------

from databricks import automl

# COMMAND ----------

summary = automl.classify(train_df, target_col="income", timeout_minutes=5, experiment_name="AutoML Classification Python API")

# COMMAND ----------

print(summary)

# COMMAND ----------

import mlflow

test_df = test_df.toPandas()
y_test = test_df["income"]
x_test = test_df.drop("income", axis=1)

model_uri = summary.best_trial.model_path

# run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(x_test)
test_df["income_predicted"] = predictions
display(test_df)

# COMMAND ----------

#predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type='string')
#display(test_df.withColumn('income_predicted', predict_udf))

# COMMAND ----------

import sklearn.metrics
 
model = mlflow.sklearn.load_model(model_uri)
sklearn.metrics.plot_confusion_matrix(model, x_test, y_test)

# COMMAND ----------


