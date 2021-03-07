# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('rfproject').getOrCreate()

# COMMAND ----------

data = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/dog_food.csv', inferSchema = True, header = True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['A', 'B', 'C', 'D'], outputCol = 'features')

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

final_data = output.select('features', 'Spoiled')

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7, 0.3])

# COMMAND ----------

rfc = RandomForestClassifier(labelCol = 'Spoiled', featuresCol = 'features')

# COMMAND ----------

rfc_model = rfc.fit(final_data)

# COMMAND ----------

rfc_feat = rfc_model.featureImportances

# COMMAND ----------

rfc_feat

# COMMAND ----------


