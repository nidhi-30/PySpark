# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('recommend').getOrCreate()

# COMMAND ----------

from pyspark.ml.recommendation import ALS

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

data = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/movielens_ratings.csv', inferSchema = True, header = True)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.show()

# COMMAND ----------

data.describe().show()

# COMMAND ----------

train_data, test_data = data.randomSplit([0.8,0.2])

# COMMAND ----------

als = ALS(maxIter = 5, regParam = 0.01, userCol = 'userId', itemCol = 'movieId', ratingCol = 'rating')

# COMMAND ----------

model = als.fit(train_data)

# COMMAND ----------

prediction = model.transform(test_data)

# COMMAND ----------

prediction.show()  

# COMMAND ----------

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')

# COMMAND ----------

rmse = evaluator.evaluate(prediction)

# COMMAND ----------

print('RMSE')
print(rmse)

# COMMAND ----------

single_user = data.filter(data['userId'] == 11).select(['movieId', 'userId'])

# COMMAND ----------

single_user.show()

# COMMAND ----------

recommendation = model.transform(single_user)

# COMMAND ----------

recommendation.orderBy('prediction', ascending = False).show()

# COMMAND ----------


