# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('cruise').getOrCreate()

# COMMAND ----------

data = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/cruise_ship_info.csv', inferSchema = True, header = True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.show()

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol = 'Cruise_line', outputCol = 'Cruise_indexed')

# COMMAND ----------

new_data = indexer.fit(data).transform(data)

# COMMAND ----------

new_data.printSchema()

# COMMAND ----------

new_data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['Age', 'Tonnage', 'passengers', 'length', 'cabins', 'passenger_density', 'Cruise_indexed'],
                           outputCol = 'features')

# COMMAND ----------

output = assembler.transform(new_data)

# COMMAND ----------

output.printSchema()

# COMMAND ----------

output.show()

# COMMAND ----------

final_data = output.select('features', 'crew')

# COMMAND ----------

final_data.show()

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7, 0.3])

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

lr = LinearRegression(labelCol = 'crew')

# COMMAND ----------

lr_model = lr.fit(train_data)

# COMMAND ----------

test_results = lr_model.evaluate(test_data)

# COMMAND ----------

test_results.residuals.show()

# COMMAND ----------

test_results.rootMeanSquaredError

# COMMAND ----------

test_results.r2

# COMMAND ----------

test_results.meanAbsoluteError

# COMMAND ----------

test_results.meanSquaredError

# COMMAND ----------

unlabeled_data = test_data.select('features')

# COMMAND ----------

unlabeled_data.show()

# COMMAND ----------

predictions = lr_model.transform(unlabeled_data)

# COMMAND ----------

predictions.show()

# COMMAND ----------

from pyspark.sql.functions import corr

# COMMAND ----------

data.describe().show()

# COMMAND ----------

data.select(corr('crew', 'passengers')).show()

# COMMAND ----------


