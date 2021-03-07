# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Basics').getOrCreate()

# COMMAND ----------

df = spark.read.json('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/Data/people.json')

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.columns

# COMMAND ----------

df.describe().show()

# COMMAND ----------

from pyspark.sql.types import StructField, StringType, IntegerType, StructType

# COMMAND ----------

data_schema = [StructField('age', IntegerType(), True),
              StructField('name', StringType(), True)]

# COMMAND ----------

final_struc = StructType(fields = data_schema)

# COMMAND ----------

df = spark.read.json('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/Data/people.json', schema = final_struc)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.show()

# COMMAND ----------

type(df['age'])

# COMMAND ----------

type(df.select('age'))

# COMMAND ----------

df.select('age').show()

# COMMAND ----------

type(df.head(2)[0])

# COMMAND ----------

 df.select(['age', 'name']).show()

# COMMAND ----------

df.withColumn('newage', df['age']).show()

# COMMAND ----------

df.withColumn('double_age', df['age']*2).show()

# COMMAND ----------

df.show()

# COMMAND ----------

df.withColumnRenamed('age', 'my_new_age').show()

# COMMAND ----------

df.createOrReplaceTempView("people")

# COMMAND ----------

results = spark.sql("SELECT * FROM people")

# COMMAND ----------

results.show()

# COMMAND ----------

new_results = spark.sql("SELECT * FROM people WHERE age=30")

# COMMAND ----------

new_results.show()

# COMMAND ----------


