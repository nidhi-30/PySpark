# Databricks notebook source
# API Key = wHhAACQ7sRGxVTXoqK6asbUCD
# Key Secret = 2aJKigkrlIFcpJn4wZeNzWXkZ0AGY6YwksqgHba8fLonERXjDC

# COMMAND ----------

# Access token: 2565362018-s3oHoNNkrZwJUVlm3EmAcJNZMCnHGAebGN5XCPj
# Access token secret: 7dy9nJhm2c3i6IkZZRjEp8XvlQK8k2KQvHTdEzabTRoDV

# COMMAND ----------

import matplotlib
import seaborn

# COMMAND ----------

# !pip install tweepy

# COMMAND ----------

import tweepy
from tweepy import OAuthHandler, Stream

# COMMAND ----------

from tweepy.streaming import StreamListener

# COMMAND ----------

import socket
import json

# COMMAND ----------

consumer_key = 'wHhAACQ7sRGxVTXoqK6asbUCD'
consumer_secret = '2aJKigkrlIFcpJn4wZeNzWXkZ0AGY6YwksqgHba8fLonERXjDC'
access_token = '2565362018-s3oHoNNkrZwJUVlm3EmAcJNZMCnHGAebGN5XCPj'
access_secret = '7dy9nJhm2c3i6IkZZRjEp8XvlQK8k2KQvHTdEzabTRoDV'

# COMMAND ----------

class TweetListener(StreamListener):
  
  def __init__(self, csocket):
    self.client_socket = csocket
    
  def on_data(self, data):
    try:
      msg = json.loads(data)
      print(msg['text'].encode('utf-8'))
      self.client_socket.send(msg['text'].encode('utf-8'))
      return True
    except BaseException as e:
      print("ERROR", e)
      return True
    
    
  def on_error(self, status):
    print(status)
    return True

# COMMAND ----------

def sendData(c_socket):
  auth = OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_secret)
  
  twitter_stream = Stream(auth, TweetListener(c_socket))
  twitter_stream.filter(track = ['guitar'])

# COMMAND ----------

if __name__ = '__main__':
  s = socket.socket()
  host = '127.0.0.1'
  port = 5555
  s.bind((host, port))
  
  print('listening on port 5555')
  
  s.listen(5)
  c, addr = s.accept()
  
  sendData(c)

# COMMAND ----------

 
