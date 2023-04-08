import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from sentence_transformers import SentenceTransformer
import os



#----------------------------------------------------------------------------------------------------
#------------------------------------------helper functions-----------------------------------------

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def load_vectors(client:Redis, tweet_metadata, vector_dict, vector_field_name):
    p = client.pipeline(transaction=False)
    for index in tweet_metadata.keys():    
        #hash key
        key='Tweet:'+ str(index)+ ':' + tweet_metadata[index]['keywords']
        #hash values
        tweets_metadata = tweet_metadata[index]
        keywords_vector = vector_dict[index].astype(np.float32).tobytes()
        tweets_metadata[vector_field_name]=keywords_vector
        # HSET
        p.hset(key, mapping=tweets_metadata)

    p.execute()

def create_flat_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2'):
    redis_conn.ft().create_index([
        VectorField(vector_field_name, "FLAT", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "BLOCK_SIZE":number_of_vectors }),
        TagField("keywords"),
        # TextField("is_quote_tweet"),
        # TextField("primary_key"),
        # TextField("conversation_id"),
        # TextField("is_retweet")        
    ])


def create_hnsw_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2',M=40,EF=200):
    redis_conn.ft().create_index([
        VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "M": M, "EF_CONSTRUCTION": EF}),
        TagField("keywords"),
        # TextField("is_quote_tweet"),
        # TextField("primary_key"),
        # TextField("conversation_id"),
        # TextField("is_retweet")   
    ])    



#----------------------------------------------------------------------------------------------------
#----------------------------------- Clean data, Connect to Redis and Transformer ------------------

# Set the path to the directory containing the CSV files
#path = r"C:\Users\marty\Documents\PythonScripts\Whisperhackathon\Hackathon3\data"

# Use the OS module to change the current working directory to the directory containing the CSV files
#os.chdir(path)

#call and clean csv data
all_columns_df = pd.read_csv("translated.csv")
all_columns_df.reset_index(drop=True, inplace=True)  
all_columns_df = all_columns_df[['conversation_id', 'favorite_count', 'full_text', 'is_quote_tweet', 'is_retweet', 'primary_key', 'keywords']]
all_columns_df = all_columns_df.astype(str)
all_columns_df[['is_quote_tweet', 'is_retweet']] = all_columns_df[['is_quote_tweet', 'is_retweet']].replace('True', 'Yes')
all_columns_df[['is_quote_tweet', 'is_retweet']] = all_columns_df[['is_quote_tweet', 'is_retweet']].replace('False', 'No')
all_columns_df[['is_quote_tweet', 'is_retweet']] = all_columns_df[['is_quote_tweet', 'is_retweet']].replace('nan', 'No')

# connect to define Redis connection parameters
try:
    redis_conn = redis.Redis(
      host=os.environ["REDIS_HOST"],
      port=os.environ["REDIS_PORT"],
      password=os.environ["REDIS_PASSWORD"])
    print ('Connected to redis')
except ValueError as e:
  raise ValueError(f"Your redis connected error: {e}")

# call model
if not model:
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
else:
   pass


# Use Model to vectorise Keywords
keywords =  [tweet_metadata[i]['keywords']  for i in tweet_metadata.keys()]
keywords_vectors = [ model.encode(sentence) for sentence in keywords]

# define tweet metadata as dictionary 
tweet_metadata = all_columns_df.to_dict(orient='index')



#---------------------------------------------------------------------------------------------------------------
#------------------------------------------ Index and query the data--------------------------------------

KEYWORDS_EMBEDDING_FIELD='keyword_vector'
TEXT_EMBEDDING_DIMENSION=784
NUMBER_TWEETS=1000

print ('Loading and Indexing + ' +  str(NUMBER_TWEETS) + ' tweets')

#flush all data
redis_conn.flushall()

#create flat index & load vectors
create_flat_index(redis_conn, KEYWORDS_EMBEDDING_FIELD,NUMBER_TWEETS,TEXT_EMBEDDING_DIMENSION,'COSINE')
load_vectors(redis_conn,tweet_metadata,keywords_vectors,KEYWORDS_EMBEDDING_FIELD)


#------------------------------------------------------------------------------------------------------------
#--------------------------------------Query FLAT index --------------------------------------

topK=5
tweet_query='rwanda drc'
#product_query='cool way to pimp up my cell'

#vectorize the query
query_vector = model.encode(tweet_query).astype(np.float32).tobytes()

#prepare the query
q = Query(f'*=>[KNN {topK} @{KEYWORDS_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','keywords_vectors').dialect(2)
params_dict = {"vec_param": query_vector}


#Execute the query
results = redis_conn.ft().search(q, query_params = params_dict)

#Print similar products found
for tweet in results.docs:
    print ('***************Similar tweets  found ************')
    print (color.YELLOW + 'Score = ' +  color.END  + tweet.vector_score)
   # print (color.YELLOW + 'conversation_id = ' +  color.END  + tweet.conversation_id)
    print (color.BOLD + 'keywords = ' +  color.END + tweet.keywords)