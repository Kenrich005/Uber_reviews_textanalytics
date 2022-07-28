import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter
import statsmodels.api as sm
import sklearn
import xgboost
import googletrans
from googletrans import *
from langdetect import detect
import plotly.express as px               
from plotly.subplots import make_subplots 
import plotly.graph_objects as go         
from wordcloud import WordCloud
import re
import string
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_numeric
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))



# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND PAGE ICON
st.set_page_config(layout="wide", page_icon=":taxi:")

header = st.container()
dataset = st.container()
trans_emoticon = st.container()
preprocessing = st.container()
ngram = st.container()

with header:
    
    st.title('Text Analytics for Uber Data !')
    st.markdown("---")
    st.subheader('Expectations from the Project')
    st.text(' - Analyze text reviews of Uber cabs’ US services')
    st.text(' - Relate whether and which different features of these reviews impact overall ratings')
    st.text(' - Pinpoint possible areas of improvement')
    
    
with dataset:
    
    st.sidebar.image("https://github.com/Kenrich005/Uber_reviews_textanalytics/blob/6ace3968785f4cfcaca57fbc589238935940bc86/ISB_Logo_JPEG.jpg?raw=true", use_column_width=True)
    st.sidebar.title("Uber Dataset")
    st.sidebar.text('This dataset was provided by ISB')
    
    
# Enter stuff here !!!!    
 
with trans_emoticon:
          
    st.subheader('Rating distribution')
    df = pd.read_csv('uber_reviews_itune_translated_emoji.csv')
    
        
    # Make table
    
    st.text("On preliminary analysis of the dataset, multiple languages and emoticons were detected.")
    st.text("Hence, Review column was translated to 'English' in order to maintain consistency.")
    st.text("Let's check if the data has been translated and if emoticons are removed.")
    
    df_lang_count  = df.groupby(by=["Language"]).size().reset_index(name="Counts")
    fig_table = go.Figure(data=go.Table(columnwidth=[2,1,5,1,1],header=dict(values=list(df.columns), fill_color='orange', align='left'),
                cells=dict(values=[df.Title, df.Rating, df.Review, df.Date, df.Language], align='left')))
    fig_table.update_layout(margin=dict(l=5, r=5, b=10, t=10))
    st.write(fig_table)
    
    #Converting Date into datetime format
    df['Date'] =  pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    
    st.markdown("###### Note:")
    st.text("The dataset has been completely translated in English using googletrans api")
    st.text("Emoticons have been removed using regular expressions")
    st.text("Columns like 'Author_Name', 'Author_URL' & 'App_Version' are dropped")
    st.text(" - This ensures user privacy")
    st.text(" - The dropped columns do not add significant value for sentiment analysis")
    
    
    
    # Rating distribution
    st.subheader("How have the customer's rated us?")
    df_rating_count  = df.groupby(by=["Rating"]).size().reset_index(name="Counts")
    fig_bar1 = px.bar(data_frame=df_rating_count, x="Rating", y="Counts", color='Rating', text_auto=True)
    
    st.plotly_chart(fig_bar1)
    
    st.markdown('###### As could be observed from the above distribution, most of the people seem unsatisfied with the services.')
    
    
    
with preprocessing:
   
    def clean_text(Review_to_clean):
    
        # Replacing @handle with the word USER
        Review_handle = Review_to_clean.str.replace(r'@[\S]+', 'user')
    
        # Replacing the Hash tag with the word HASH
        Review_hash = Review_handle.str.replace(r'#(\S+)','hash')
        
        # Replacing Two or more dots with one
        Review_dot = Review_hash.str.replace(r'\.{2,}', ' ')
        
        # Removing all the special Characters
        Review_special = Review_dot.str.replace(r'[^\w\d\s]',' ')
    
        # Removing all the non ASCII characters
        Review_ascii = Review_special.str.replace(r'[^\x00-\x7F]+',' ')
    
        # Removing the leading and trailing Whitespaces
        Review_space = Review_ascii.str.replace(r'^\s+|\s+?$','')
    
        # Replacing multiple Spaces with Single Space
        Dataframe = Review_space.str.replace(r'\s+',' ')
        
            
        return Dataframe
    
    df['Review'] = clean_text(df['Review'])
    df['Review'] = df['Review'].apply(str)

  
with ngram:
    
    st.subheader("Let's check for N-grams as per the rating.")
    st.write("In the fields of computational linguistics and probability, an n-gram is a continuous sequence of n items from a given sample of text or speech.")
    st.write("The items can be phonemes, syllables, letters, words or base pairs according to the application. The n-grams typically are collected from a text or speech corpus.")
    
  
    # Removing the stop words before plotting

    df['Review'] = df['Review'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
    
    
    # Rating with ngram.
    df_data_pos = " ".join(df.loc[df['Rating'] == 1, 'Review'])
    token_text_pos = word_tokenize(df_data_pos)
    gram_pos = ngrams(token_text_pos, 2)
    frequency_pos = Counter(gram_pos)
    
       
    df = pd.DataFrame(frequency_pos.most_common(20))
    
    plt.rcParams['figure.figsize'] = [12, 6]
    sns.set(font_scale = 0.8, style = 'whitegrid')
    
    # plotting
    word_count = sns.barplot(x = df[1], y = df[0], color = 'hotpink')
    word_count.set_title("Word Count Plot")
    word_count.set_ylabel("Words")
    word_count.set_xlabel("Count");
    st.pyplot()
    
    
    
    wordcloud = WordCloud(width = 1000, height = 500, background_color = 'white').generate(df_data_pos)
    plt.figure(figsize=(10, 4))
    plt.imshow(wordcloud)
    plt.axis('off');
    st.pyplot()
    
    
    # Kenny in the above code, you may change the Rating parameter (1-5) and parameter for ngram to change the Uni, Bi, Tri by changine the number. 

    
    

    
    
    
    
    
    

    
    