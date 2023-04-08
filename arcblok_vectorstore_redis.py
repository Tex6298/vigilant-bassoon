import redis
from langchain.vectorstores.redis import Redis


class ArcBlokRedis(Redis):

  def __init__(self, host, port, password, index_name, embedding_function):
    self.redis_host = host
    self.redis_port = port
    self.redis_password = password
    # Connect to redis instance
    self.client = redis.Redis(host=self.redis_host,
                              port=self.redis_port,
                              password=self.redis_password)

    self.embedding_function = embedding_function
    self.index_name = index_name
