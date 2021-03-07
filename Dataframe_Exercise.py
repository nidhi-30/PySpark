# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName("Exercise").getOrCreate()

# COMMAND ----------

df = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/walmart_stock.csv', inferSchema = True, header = True)

# COMMAND ----------

df.columns

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.head(5)

# COMMAND ----------

newdf = df.describe()

# COMMAND ----------

newdf.printSchema()

# COMMAND ----------

from pyspark.sql.functions import format_number

# COMMAND ----------

newdf.select(newdf['Summary'],format_number(newdf['Open'].cast('float'), 2).alias("Open"),
            format_number(newdf['High'].cast('float'), 2).alias("High"),
            format_number(newdf['Low'].cast('float'), 2).alias("Low"),
            format_number(newdf['Close'].cast('float'), 2).alias("Close"),
            newdf['Volume'].cast('int').alias("Volume")).show()

# COMMAND ----------

dfnew = df.withColumn("HV Ratio", df['High']/df['Volume'])

# COMMAND ----------

dfnew.select("HV Ratio").show()

# COMMAND ----------

df.show()

# COMMAND ----------

df.orderBy(df['High'].desc()).select('Date').collect()[0][0]

# COMMAND ----------

from pyspark.sql.functions import mean, max, min

# COMMAND ----------

df.select(mean(df['Close'])).show()

# COMMAND ----------

df.select(min('Volume'), max('Volume')).show()

# COMMAND ----------

df.filter(df['Close'] <  60).count()

# COMMAND ----------

(df.filter(df['High'] > 80).count()/ df.count()) * 100

# COMMAND ----------

from pyspark.sql.functions import corr, year

# COMMAND ----------

df.select(corr('High', 'Volume')).show()

# COMMAND ----------

res = df.groupBy(year(df['Date'])).max()

# COMMAND ----------

res.select('year(Date)', 'max(High)').show()

# COMMAND ----------

from pyspark.sql.functions import month

# COMMAND ----------

res1 = df.groupBy(month(df['Date'])).avg()

# COMMAND ----------

res1.select('month(Date)', 'avg(CLose)').orderBy('month(Date)').show()

# COMMAND ----------


