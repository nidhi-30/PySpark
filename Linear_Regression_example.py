# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('lrex').getOrCreate()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

training = spark.read.format('libsvm').load('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/sample_linear_regression_data.txt')

# COMMAND ----------

training.show()

# COMMAND ----------

lr = LinearRegression(featuresCol = 'features', labelCol = 'label', predictionCol = 'prediction')

# COMMAND ----------

lrModel = lr.fit(training)

# COMMAND ----------

lrModel.coefficients

# COMMAND ----------

lrModel.intercept

# COMMAND ----------

training_summary = lrModel.summary

# COMMAND ----------

training_summary.r2

# COMMAND ----------

training_summary.rootMeanSquaredError

# COMMAND ----------

all_data = spark.read.format('libsvm').load('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/sample_linear_regression_data.txt')

# COMMAND ----------

train_data, test_data = all_data.randomSplit([0.7, 0.3])

# COMMAND ----------

train_data.describe().show()

# COMMAND ----------

test_data.describe().show()

# COMMAND ----------

correct_model = lr.fit(train_data)

# COMMAND ----------

test_results = correct_model.evaluate(test_data)

# COMMAND ----------

test_results.rootMeanSquaredError

# COMMAND ----------

unlabled_data = test_data.select('features')

# COMMAND ----------

unlabled_data.show()

# COMMAND ----------

predictions = correct_model.transform(unlabled_data)

# COMMAND ----------

predictions.show()

# COMMAND ----------


