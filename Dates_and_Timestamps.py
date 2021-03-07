# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('dates').getOrCreate()

# COMMAND ----------

df = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/appl_stock-1.csv', inferSchema = True, header = True)

# COMMAND ----------

df.head()

# COMMAND ----------

df.select(['Date', 'Open']).show()

# COMMAND ----------

from pyspark.sql.functions import dayofmonth, dayofyear, hour, month, year, weekofyear, format_number, date_format

# COMMAND ----------

df.select(dayofyear(df['Date'])).show()

# COMMAND ----------

df.select(year(df['Date'])).show()

# COMMAND ----------

newdf = df.withColumn("Year", year(df['Date']))

# COMMAND ----------

result = newdf.groupBy("Year").mean().select(["Year", "avg(Close)"])

# COMMAND ----------

new = result.withColumnRenamed("avg(Close)", "Average Closing Price")

# COMMAND ----------

new.select(["Year", format_number(("Average Closing Price"), 2).alias("Avg Close")]).show()

# COMMAND ----------


