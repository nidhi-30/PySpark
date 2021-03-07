# Databricks notebook source
from pyspark import SparkContext

# COMMAND ----------

from pyspark.streaming import StreamingContext

# COMMAND ----------

ssc = StreamingContext(sc, 1)

# COMMAND ----------

lines = ssc.socketTextStream('localhost', 9999)

# COMMAND ----------

words = lines.flatMap(lambda line: line.split(' '))

# COMMAND ----------

pairs = words.map(lambda word : (word, 1))

# COMMAND ----------

word_counts = pairs.reduceByKey(lambda num1, num2 : num1+num2)

# COMMAND ----------

word_counts.pprint()

# COMMAND ----------

ssc.start()

# COMMAND ----------


