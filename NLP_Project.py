# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('nlp').getOrCreate()

# COMMAND ----------

data = spark.read.csv('dbfs:/FileStore/shared_uploads/pateln@lakeheadu.ca/SMSSpamCollection', inferSchema = True, sep = '\t')

# COMMAND ----------

data.show()

# COMMAND ----------

data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c1', 'text')

# COMMAND ----------

from pyspark.sql.functions import length

# COMMAND ----------

data = data.withColumn('length', length(data['text']))

# COMMAND ----------

data.show()

# COMMAND ----------

data.groupBy('class').mean().show()

# COMMAND ----------

from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer)

# COMMAND ----------

tokenizer = Tokenizer(inputCol = 'text', outputCol = 'token_text')
stop_remove = StopWordsRemover(inputCol = 'token_text', outputCol = 'stop_token')

# COMMAND ----------

count_vec = CountVectorizer(inputCol = 'stop_token', outputCol = 'c_vec')
idf = IDF(inputCol = 'c_vec', outputCol = 'tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol = 'class', outputCol = 'label')

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

clean_up = VectorAssembler(inputCols = ['tf_idf', 'length'], outputCol = 'features')

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

# COMMAND ----------

nb = NaiveBayes()

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

data_prep_pipe = Pipeline(stages = [ham_spam_to_numeric, tokenizer, stop_remove, count_vec, idf, clean_up])

# COMMAND ----------

cleaner = data_prep_pipe.fit(data)

# COMMAND ----------

clean_data = cleaner.transform(data)

# COMMAND ----------

clean_data.columns

# COMMAND ----------

clean_data = clean_data.select('label', 'features')

# COMMAND ----------

clean_data.show()

# COMMAND ----------

train_data, test_data = clean_data.randomSplit([0.7,0.3])

# COMMAND ----------

spam_detector = nb.fit(train_data)

# COMMAND ----------

test_result = spam_detector.transform(test_data)

# COMMAND ----------

test_result.show()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

acc_eval = MulticlassClassificationEvaluator()

# COMMAND ----------

acc = acc_eval.evaluate(test_result)

# COMMAND ----------

print('Acc of NB')
print(acc)

# COMMAND ----------


