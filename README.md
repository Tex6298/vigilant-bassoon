This repository contains an MVP application for performing QA over the TTPs of the DISARM Framework and applying them in reasoning about social media data provided by an OSINT researcher. It leverages GPT from OpenAI, LangChain, Redis vector indexes & similarity search, and a modified dataset of the DISARM Framework TTPs, as well as a collated sample dataset of tweets from past influence operations. 

Our team presentation for the hackathon project is accessible at [,..,]()

To run this code locally, you will need an API key from OpenAI as well as credentials for a Redis database. You must rename the file `.env.template` to `.env` and fill in the `OPENAI_API_KEY`, `REDIS_HOST`, `REDIS_PASSWORD`, and `REDIS_PORT` environmental variable values with your corresponding private values. 

The [DISARM Framework](https://github.com/DISARMFoundation/DISARMframeworks) is owned by the DISARM Foundation and is licensed under [CC-BY-4.0](https://github.com/DISARMFoundation/DISARMframeworks/blob/main/LICENSE.md)