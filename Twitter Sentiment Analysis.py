#!/usr/bin/env python
# coding: utf-8

# Please install langdetect, textblob, wordcloud and nltk before running the script

# In[1]:


# Importing necessary libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import matplotlib.pyplot as plt

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.types import FloatType
from pyspark.ml.feature import Tokenizer, HashingTF, StopWordsRemover
from pyspark.ml.clustering import KMeans

from textblob import TextBlob
from langdetect import detect
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[2]:


# Initialize Spark Session
spark = SparkSession.builder     .appName("Tweet Data Processing")     .master("local[*]")     .config("spark.executor.memory", "10g")     .config("spark.driver.memory", "10g")     .getOrCreate()


# ## Reading in the Data

# In[3]:


# Read the CSV file with multiline option enabled
dfdt = spark.read.format("csv")     .option("header", "true")     .option("quote", "\"")     .option("escape", "\"")     .option("multiline", True)     .load("hashtag_donaldtrump.csv")
dfdt.cache()
dfjb = spark.read.format("csv")     .option("header", "true")     .option("quote", "\"")     .option("escape", "\"")     .option("multiline", True)     .load("hashtag_joebiden.csv")
dfjb.cache()


# Partitioning the data to run processs in parallel

# In[4]:


dfdt = dfdt.repartition(10)
dfjb = dfjb.repartition(10)


# ## Data Cleaning

# In[5]:


# Change the dtype of column created_at from string to datetime
dfdt = dfdt.withColumn("created_at", F.to_timestamp(dfdt["created_at"]))
dfdt = dfdt.withColumn("collected_at", F.to_timestamp(dfdt["collected_at"]))
dfdt = dfdt.withColumn("tweet_id", dfdt["tweet_id"].cast("decimal(18,0)"))
dfdt = dfdt.withColumn("likes", dfdt["likes"].cast("int"))
dfdt = dfdt.withColumn("retweet_count", dfdt["retweet_count"].cast("int"))
dfdt = dfdt.withColumn("user_id", dfdt["user_id"].cast("int"))
dfdt = dfdt.withColumn("user_join_date", F.to_timestamp(dfdt["user_join_date"]))
dfdt = dfdt.withColumn("user_followers_count", dfdt["user_followers_count"].cast("int"))
dfdt = dfdt.withColumn("lat", dfdt["lat"].cast("float"))
dfdt = dfdt.withColumn("long", dfdt["long"].cast("float"))

dfjb = dfjb.withColumn("created_at", F.to_timestamp(dfjb["created_at"]))
dfjb = dfjb.withColumn("collected_at", F.to_timestamp(dfjb["collected_at"]))
dfjb = dfjb.withColumn("tweet_id", dfjb["tweet_id"].cast("decimal(38,0)"))
dfjb = dfjb.withColumn("likes", dfjb["likes"].cast("int"))
dfjb = dfjb.withColumn("retweet_count", dfjb["retweet_count"].cast("int"))
dfjb = dfjb.withColumn("user_id", dfjb["user_id"].cast("int"))
dfjb = dfjb.withColumn("user_join_date", F.to_timestamp(dfjb["user_join_date"]))
dfjb = dfjb.withColumn("user_followers_count", dfjb["user_followers_count"].cast("int"))
dfjb = dfjb.withColumn("lat", dfjb["lat"].cast("float"))
dfjb = dfjb.withColumn("long", dfjb["long"].cast("float"))


# In[6]:


def clean_text(tweet):
    """
    Cleans the tweet text by performing various preprocessing steps.
    """
    # Convert it to Lowercase
    tweet = tweet.lower()

    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Use the sub() method to replace URLs with the specified replacement text
    tweet = url_pattern.sub("", tweet)

    # Removing Newline characters
    tweet = tweet.replace("\n", " ")
    tweet = " ".join([word for word in tweet.split(" ") if not word.startswith(("#","@"))])

    # Remove emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)

    # Remove special characters
    tweet = re.sub(r'[^a-zA-Z0-9]+', ' ', tweet)

    # Rectify a few words
    tweet = tweet.replace("donaldtrump","donald trump")
    tweet = tweet.replace("joebiden","joe biden")

    # Removing extra white spaces
    tweet = re.sub(' +', ' ', tweet.strip())

    if tweet!="":
        return tweet
    else:
        return None

# Register the function as a UDF
clean_text_udf = udf(clean_text, StringType())

# Apply the UDF to the tweet column for Trump data
dfdt_cleaned = dfdt.withColumn("cleaned_tweet", clean_text_udf("tweet"))

# Deleting all the null records
dfdt_cleaned = dfdt_cleaned.dropna(subset=["cleaned_tweet"])

# Apply the UDF to the tweet column for Biden data
dfjb_cleaned = dfjb.withColumn("cleaned_tweet", clean_text_udf("tweet"))

# Deleting all the null records
dfjb_cleaned = dfjb_cleaned.dropna(subset=["cleaned_tweet"])


# In[7]:


# Define a function to detect the language of a tweet
def detect_language(tweet):
    """
    Detects the language of a given tweet.
    """
    try:
        return detect(tweet)
    except:
        return None

# Register the UDF
detect_language_udf = udf(detect_language, StringType())

# Apply the UDF to the cleaned Trump dataframe and create a new column called "language"
dfdt_cleaned = dfdt_cleaned.withColumn("language", detect_language_udf(F.col("cleaned_tweet")))

# Apply the UDF to the cleaned Biden dataframe and create a new column called "language"
dfjb_cleaned = dfjb_cleaned.withColumn("language", detect_language_udf(F.col("cleaned_tweet")))


# In[8]:


# Replace country names for consistency
dfdt_cleaned = dfdt_cleaned.withColumn("country", F.regexp_replace(F.col("country"), "United States of America", "United States"))
dfjb_cleaned = dfjb_cleaned.withColumn("country", F.regexp_replace(F.col("country"), "United States of America", "United States"))


# In[9]:


# Filter English tweets from Trump data
dfdt_filtered = dfdt_cleaned.filter((F.col("language") == "en") & (F.col("country") == "United States"))
# Filter English tweets from Biden data
dfjb_filtered = dfjb_cleaned.filter((F.col("language") == "en") & (F.col("country") == "United States"))


# In[10]:


dfdt.unpersist()
dfjb.unpersist()


# In[11]:


# Cache filtered dataframes for optimization
dfdt_filtered.cache()
dfjb_filtered.cache()


# ## Sentiment Analysis

# In[12]:


def get_polarity(text):
    """
    Calculates the polarity (sentiment) of a given text.
    """
    return TextBlob(text).sentiment.polarity

# Register the UDF
polarity_udf = udf(get_polarity, FloatType())

# Add polarity column to Trump data
dfdt_polarized = dfdt_filtered.withColumn("polarity", polarity_udf(F.col("cleaned_tweet")))
# Assign categorical labels based on sentiment score for Trump data
dfdt_polarized = dfdt_polarized.withColumn('label',
                                           F.when(dfdt_polarized.polarity > 0, 'positive').otherwise(
                                           F.when(dfdt_polarized.polarity < 0, 'negative').otherwise('neutral')))

# Add polarity column to Biden data
dfjb_polarized = dfjb_filtered.withColumn("polarity", polarity_udf(F.col("cleaned_tweet")))
# Assign categorical labels based on sentiment score for Biden data
dfjb_polarized = dfjb_polarized.withColumn('label',
                                           F.when(dfjb_polarized.polarity > 0, 'positive').otherwise(
                                           F.when(dfjb_polarized.polarity < 0, 'negative').otherwise('neutral')))


# In[13]:


# Group and count sentiment labels for Trump data
dt_label_counts = dfdt_polarized.groupBy("label").count().orderBy("label")
dt_label_counts = dt_label_counts.withColumn("label",F.col("label").cast("string"))
dt_label_counts = dt_label_counts.withColumn("count",F.col("count").cast("integer"))

# Group and count sentiment labels for Biden data
jb_label_counts = dfjb_polarized.groupBy("label").count().orderBy("label")
jb_label_counts = jb_label_counts.withColumn("label",F.col("label").cast("string"))
jb_label_counts = jb_label_counts.withColumn("count",F.col("count").cast("integer"))

dt_labels = [row['label'] for row in dt_label_counts.select("label").collect()]
dt_counts = [row['count'] for row in dt_label_counts.select("count").collect()]

jb_labels = [row['label'] for row in jb_label_counts.select("label").collect()]
jb_counts = [row['count'] for row in jb_label_counts.select("count").collect()]


# In[15]:


# Plot sentiment distribution for Trump data
plt.bar(dt_labels,dt_counts)
plt.xlabel('Sentiments for Trump')
plt.ylabel('Number of Sentiments')
plt.show()


# In[16]:


# Plot sentiment distribution for Biden data
plt.bar(jb_labels,jb_counts)
plt.xlabel('Sentiments for Biden')
plt.ylabel('Number of Sentiments')
plt.show()


# In[17]:


def preprocess_text(df):
    """
    Pre-processes text by tokenizing, removing stop words, and applying TF-IDF.
    """
    tokenizer = Tokenizer(inputCol="cleaned_tweet", outputCol="tokeized_tweets")
    stp_wrds = stopwords.words('english')
    stp_wrds.remove("nor")
    stp_wrds.remove("not")
    stop_words = StopWordsRemover(stopWords=stp_wrds, inputCol="tokeized_tweets", outputCol="filtered_tweets")
    hashingTF = HashingTF(numFeatures=1024, inputCol="filtered_tweets", outputCol="features")
    preprocessed_df = tokenizer.transform(df)
    preprocessed_df = stop_words.transform(preprocessed_df)
    return hashingTF.transform(preprocessed_df)

# Pre-process the text for Trump data
dfdt_preprocessed = preprocess_text(dfdt_filtered)

# Pre-process the text for Biden data
dfjb_preprocessed = preprocess_text(dfjb_filtered)


# ## Clustering

# In[18]:


# Perform KMeans clustering for Trump data
Sum_of_squared_distances_dt = []
K = range(2,8)
for k in K:
    km_dfdt = KMeans(k=k)
    km_dfdt = km_dfdt.fit(dfdt_preprocessed)
    ssd_dfdt = km_dfdt.summary.trainingCost
    Sum_of_squared_distances_dt.append(ssd_dfdt)


# In[19]:


plt.plot(K, Sum_of_squared_distances_dt, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k for Donald Trump Tweets')
plt.show()


# Selecting k=6 for Donald Trump data

# In[20]:


# Perform KMeans clustering for Biden data
Sum_of_squared_distances_jb = []
K = range(2,8)
for k in K:
    km_dfjb = KMeans(k=k)
    km_dfjb = km_dfjb.fit(dfdt_preprocessed)
    ssd_dfjb = km_dfjb.summary.trainingCost
    Sum_of_squared_distances_jb.append(ssd_dfjb)


# In[26]:


plt.plot(K, Sum_of_squared_distances_jb, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k for Joe Biden Tweets')
plt.show()


# Selecting k=6 for Joe Biden data

# In[22]:


# Assign optimal cluster count for Trump data
dfdt_km = KMeans(k=6)
dfdt_km = dfdt_km.fit(dfdt_preprocessed)


# In[23]:


# Assign optimal cluster count for Biden data
dfjb_km = KMeans(k=6)
dfjb_km = dfjb_km.fit(dfjb_preprocessed)


# In[24]:


# Assign cluster labels to Trump tweets
clustered_dfdt = dfdt_km.transform(dfdt_preprocessed)

# Assign cluster labels to Biden tweets
clustered_dfjb = dfjb_km.transform(dfdt_preprocessed)


# In[25]:


# Define a window spec ordered by some column that makes sense to sample by, e.g., `created_at`
windowSpec = Window.partitionBy("prediction").orderBy("created_at")

# Add a row number within each partition
df_with_row_number = clustered_dfdt.withColumn("row_num", row_number().over(windowSpec))

# Filter to get 4 tweets per cluster for Trump data
df_filtered = df_with_row_number.filter(df_with_row_number.row_num <= 4)

# Select the columns of interest for Trump data
df_result = df_filtered.select("prediction", "tweet")

# Show the results for Trump data
df_result.show(24,truncate=False)

# Define a window spec ordered by some column that makes sense to sample by, e.g., `created_at`
windowSpec = Window.partitionBy("prediction").orderBy("created_at")

# Add a row number within each partition
df_with_row_number = clustered_dfjb.withColumn("row_num", row_number().over(windowSpec))

# Filter to get 4 tweets per cluster for Biden data
df_filtered = df_with_row_number.filter(df_with_row_number.row_num <= 4)

# Select the columns of interest for Biden data
df_result = df_filtered.select("prediction", "cleaned_tweet")

# Show the results for Biden data
df_result.show(24,truncate=False)


# In[27]:


from pyspark.sql.functions import udf
from pyspark.sql.types import MapType, StringType, IntegerType

# Define your word counting function
def word_counter(tweet):
    dict1 = {}
    for word in tweet.split():
        if word in dict1:
            dict1[word] += 1
        else:
            dict1[word] = 1
    return dict1

# Register the function as a UDF
word_counter_udf1 = udf(word_counter, MapType(StringType(), IntegerType()))
clustered_dfjb_with_words = clustered_dfjb.filter("prediction == 0").withColumn("word_counts", word_counter_udf1("cleaned_tweet"))
word_counter_udf2 = udf(word_counter, MapType(StringType(), IntegerType()))
clustered_dfdt_with_words = clustered_dfdt.filter("prediction == 0").withColumn("word_counts", word_counter_udf2("cleaned_tweet"))


def merge_dicts(dict1, dict2):
    """Merge two dictionaries by summing the values of matching keys."""
    from collections import defaultdict
    result = defaultdict(int)
    for d in (dict1, dict2):  # Loop through both dictionaries
        for key, value in d.items():
            result[key] += value
    return dict(result)
from pyspark.sql.functions import udf, col
from pyspark.sql.types import MapType, StringType, IntegerType
import functools

# UDF to merge dictionaries across all rows
merge_dicts_udf = udf(merge_dicts, MapType(StringType(), IntegerType()))

# Convert column of dictionaries to a list of dictionaries
dicts_list_jb = [row['word_counts'] for row in clustered_dfjb_with_words.select("word_counts").collect()]
dicts_list_dt = [row['word_counts'] for row in clustered_dfdt_with_words.select("word_counts").collect()]

# Reduce the list using the merge_dicts function
resultant_dict_jb = functools.reduce(merge_dicts, dicts_list_jb)
resultant_dict_dt = functools.reduce(merge_dicts, dicts_list_dt)


# In[33]:


for word in ["the", "to", "and", "a", "of", "for", "in", "on", "is", "that", "with", "as", "it", "this", "by", "be", "are", "was", "will", "or", "which", "have", "has", "i", "you", "he", "she", "they", "we", "their", "at", "from"]:
    if word in resultant_dict_jb:
        del resultant_dict_jb[word]
    if word in resultant_dict_dt:
        del resultant_dict_dt[word]


# In[34]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(resultant_dict_dt)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()


# In[35]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(resultant_dict_jb)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()

