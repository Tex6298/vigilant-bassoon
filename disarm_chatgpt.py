import redis
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import json
import os
import numpy as np
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from arcblok_vectorstore_redis import ArcBlokRedis
from langchain.embeddings.openai import OpenAIEmbeddings


class Disarm_GPT(object):
  NUM_VECTORS = 300

  def __init__(self):
    embeddings = OpenAIEmbeddings()
    # Connect to redis instance
    self.redis_client = ArcBlokRedis(
      host=os.environ["REDIS_HOST"],
      port=os.environ["REDIS_PORT"],
      password=os.environ["REDIS_PASSWORD"],
      index_name='disarm',
      embedding_function=embeddings.embed_query)

    # Initialize sentence transformer model
    self.model = SentenceTransformer(
      'sentence-transformers/all-distilroberta-v1')
    # Connect to the OpenAI API
    openai.api_key = os.environ['OPENAI_API_KEY']

    self._template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the most recent state of the union address.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
      self._template)

    self.template = """You are an AI assistant for answering questions about disinformation campaign and misinformation on social media.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Dude, I'm, like, not sure." Don't try to make up an answer.
If the question is not about disinformation campaign, conspiracy theories, or misinformation on social media, politely inform them that you are tuned to only answer questions about disinformation campaign and misinformation on social media.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
    self.QA_PROMPT = PromptTemplate(template=self.template,
                                    input_variables=["question", "context"])

  def get_chain(self):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
      llm,
      self.redis_client,
      qa_prompt=self.QA_PROMPT,
      condense_question_prompt=self.CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain


