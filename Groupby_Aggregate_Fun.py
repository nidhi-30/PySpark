# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('aggs').getOrCreate()

# COMMAND ----------

df = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/sales_info.csv', inferSchema = True, header = True)

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.groupBy('Company').count().show()

# COMMAND ----------

df.agg({'Sales': 'max'}).show()

# COMMAND ----------

group_data = df.groupBy("Company")

# COMMAND ----------

group_data.agg({'Sales':'max'}).show()

# COMMAND ----------

from pyspark.sql.functions import countDistinct, avg, stddev

# COMMAND ----------

df.select(countDistinct('Sales')).show()

# COMMAND ----------

df.select(avg('Sales').alias('Average Sales')).show()

# COMMAND ----------

df.select(stddev('Sales')).show()

# COMMAND ----------

from pyspark.sql.functions import format_number

# COMMAND ----------

sales_std = df.select(stddev('Sales').alias('std'))

# COMMAND ----------

sales_std.select(format_number('std', 2).alias('std')).show()

# COMMAND ----------

df.orderBy('Sales').show()

# COMMAND ----------

df.orderBy(df['Sales'].desc()).show()

# COMMAND ----------


