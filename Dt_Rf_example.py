# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('tree').getOrCreate()

# COMMAND ----------

data = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/College.csv', inferSchema = True, header = True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

data.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols = ['Apps', 'Accept', 'Enroll', 'Top10perc', 'Top25perc', 'F_Undergrad', 
                                         'P_Undergrad', 'Outstate', 'Room_Board', 'Books', 'Personal', 'PhD', 
                                         'Terminal', 'S_F_Ratio', 'perc_alumni', 'Expend', 'Grad_Rate'], outputCol = 'feature')

# COMMAND ----------

output = assembler.transform(data)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol = 'Private', outputCol = 'PrivateIndex')

# COMMAND ----------

output_fixed = indexer.fit(output).transform(output)

# COMMAND ----------

output_fixed.printSchema()

# COMMAND ----------

final_data = output_fixed.select('feature', 'PrivateIndex')

# COMMAND ----------

train_data, test_data = final_data.randomSplit([0.7, 0.3])

# COMMAND ----------

from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier, RandomForestClassifier)

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

dtc = DecisionTreeClassifier(labelCol = 'PrivateIndex', featuresCol = 'feature')
rfc = RandomForestClassifier(labelCol = 'PrivateIndex', featuresCol = 'feature')
gbt = GBTClassifier(labelCol = 'PrivateIndex', featuresCol = 'feature')

# COMMAND ----------

dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

# COMMAND ----------

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

my_bi_eval = BinaryClassificationEvaluator(labelCol = 'PrivateIndex')

# COMMAND ----------

print('DTC')
print(my_bi_eval.evaluate(dtc_preds))

# COMMAND ----------

print('RFC')
print(my_bi_eval.evaluate(rfc_preds))

# COMMAND ----------

print('GBT')
print(my_bi_eval.evaluate(gbt_preds))

# COMMAND ----------

my_bi_eval2 = BinaryClassificationEvaluator(labelCol = 'PrivateIndex', rawPredictionCol = 'prediction')

# COMMAND ----------

print('GBT')
print(my_bi_eval2.evaluate(gbt_preds))

# COMMAND ----------


