# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('lr_example').getOrCreate()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# COMMAND ----------

data = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/Ecommerce_Customers.csv', inferSchema = True, header = True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.head()

# COMMAND ----------

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership'],
                           outputCol = 'features')

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

output.printSchema()

# COMMAND ----------

output.head()

# COMMAND ----------

final_data = output.select('features', 'Yearly Amount Spent')

# COMMAND ----------

final_data.show()

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7, 0.3])

# COMMAND ----------

train_data.describe().show()

# COMMAND ----------

lr = LinearRegression(labelCol = 'Yearly Amount Spent')

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

final_data.describe().show()

# COMMAND ----------

unlabeled_data = test_data.select('features')

# COMMAND ----------

unlabeled_data.show()

# COMMAND ----------

predictions = lr_model.transform(unlabeled_data)

# COMMAND ----------

predictions.show()

# COMMAND ----------


