import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis
from langchain.document_loaders import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import uuid
import logger
import numpy as np
import json

def upload_disarm_data():
  try:
    client = redis.Redis(
      host=os.environ["REDIS_HOST"],
      port=os.environ["REDIS_PORT"],
      password=os.environ["REDIS_PASSWORD"])
  except ValueError as e:
    raise ValueError(f"Your redis connected error: {e}")
  
  loader = CSVLoader(file_path='./merged_file.csv')
  
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  docs = text_splitter.split_documents(documents)
  docs = docs[0:400]
  
  embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
  
  index_name = 'disarm'
  metadatas = None
  embeddings = embedding.embed_documents([x.page_content for x in docs])
  
  dim = len(embeddings[0])
  # Constants
  vector_number = len(embeddings)  # initial number of vectors
  # name of the search index if not given
  if not index_name:
    index_name = uuid.uuid4().hex
  prefix = f"doc:{index_name}"  # prefix for the document keys
  distance_metric = (
    "COSINE"  # distance metric for the vectors (ex. COSINE, IP, L2)
  )
  content = TextField(name="content")
  metadata = TextField(name="metadata")
  content_embedding = VectorField(
    "content_vector",
    "FLAT",
    {
      "TYPE": "FLOAT32",
      "DIM": dim,
      "DISTANCE_METRIC": distance_metric,
      "INITIAL_CAP": vector_number,
    },
  )
  fields = [content, metadata, content_embedding]
  
  # Check if index exists
  try:
    client.ft(index_name).info()
    logger.info("Index already exists")
  except:  # noqa
    # Create Redis Index
    client.ft(index_name).create_index(
      fields=fields,
      definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
    )
  
    pipeline = client.pipeline()
    for i, text in enumerate(docs):
      key = f"{prefix}:{i}"
      metadata = metadatas[i] if metadatas else {}
      pipeline.hset(
        key,
        mapping={
          "content": text.page_content,
          "content_vector": np.array(embeddings[i], dtype=np.float32).tobytes(),
          "metadata": json.dumps(metadata),
        },
      )
    pipeline.execute()
print("Data Done")

