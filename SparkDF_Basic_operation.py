# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("ops").getOrCreate()

# COMMAND ----------

df = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/appl_stock.csv', inferSchema = True, header = True)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.head()

# COMMAND ----------

df.filter("Close < 500").select(['Open', 'Close']).show()

# COMMAND ----------

df.filter((df['Close'] < 200) & (df['Open'] > 200)).show()

# COMMAND ----------

result = df.filter(df['Low'] == 197.16).collect()

# COMMAND ----------

row = result[0]

# COMMAND ----------

row.asDict()

# COMMAND ----------


