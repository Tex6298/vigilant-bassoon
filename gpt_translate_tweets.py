import numpy as np
import pandas as pd
import openai

#----------------------------------------------------------------------------------------------------------
#-------------------------------------Functinos -------------------------------------------------


# set openai API key
def set_openai_key(api_key):
    openai.api_key = api_key



# define a function to auto-truncate long text fields
def auto_truncate(val):
    return val[:MAX_TEXT_LENGTH]


def gpt3_algorithm_v2(text, prompt):
  # Completion function call engine: text-davinci-002

    Platformresponse = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{prompt}:{text}.",
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        )

    return Platformresponse.choices[0].text


#----------------------------------------------------------------------------------------------------------
#-------------------------------------Load Tweet data -------------------------------------------------


MAX_TEXT_LENGTH = 512
NUMBER_TWEETS = 1000
# load the product data and truncate long text fields
all_columns_df = pd.read_csv("data/combineddatasets.csv", converters={'full_text': auto_truncate})
all_columns_df['primary_key'] = all_columns_df['created_at'] + '-' + all_columns_df['full_text']
all_columns_df['full_text'].replace('', np.nan, inplace=True)
all_columns_df.dropna(subset=['full_text'], inplace=True)
all_columns_df = all_columns_df.sample(n=NUMBER_TWEETS)
all_columns_df.reset_index(drop=True, inplace=True)


prompt = "Repeat the Tweet, if necessary, translate it to English leaving all hashtags unchanged"

api_key = os.environ["OPENAI_API_KEY"]
set_openai_key(api_key)

#Loop through dataframe
for index, row in all_columns_df.iterrows():
    #extract column to be translated
    text_to_translate = all_columns_df.loc[index, 'full_text']
    #translate using OpenAI API
    translated_text = gpt3_algorithm_v2(text_to_translate)
    #add translated text to new column
    all_columns_df.at[index, 'translated_text'] = translated_text
    if index % 100 == 0:
        print(index)
#return dataframe with original and translated columns
    all_columns_df = all_columns_df

#----------------------------------------------------------------------------------------------------------
#-------------------------------------Tidy df and save to CSV -------------------------------------------------



all_columns_df['translated_text'] = all_columns_df['translated_text'].str.replace('\n\n', '')
all_columns_df['primary_key'] = all_columns_df['created_at'] + '-' + all_columns_df['translated_text']
all_columns_df.reset_index(drop=True, inplace=True)
all_columns_df.to_csv('data/translated.csv', index=False)


