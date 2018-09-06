from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer,CountVectorizer, RegexTokenizer,StopWordsRemover
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import explode,col,concat, expr, when, lower, udf,collect_list,concat_ws
import operator
from pyspark.ml.linalg import Vectors, VectorUDT
from nltk.stem.porter import PorterStemmer
import pandas as pd
pd.set_option('display.width', 300)

# For benchmarking purpose to select limited number of tweets in multiple of 100k, Select between 0-10

multiplier = 10



#

# User Profiling & Clustering through Topic_Modeling

def Topic_Modeling(tweet_df):
    
    # Initializing Model for tokenizing the tweets for each user

    tokenizer = Tokenizer(inputCol="concat_ws( , collect_list(tweet))", outputCol="words")

    regexTokenizer = RegexTokenizer(inputCol="concat_ws( , collect_list(tweet))", outputCol="tokens", pattern="\\W+",minTokenLength=4)

    # udf for counting tokens

    countTokens = udf(lambda words: len(words), IntegerType())

    # Tokenizing the data

    regexTokenized = regexTokenizer.transform(tweet_df)

    regexTokenized.select("user_id","concat_ws( , collect_list(tweet))", "tokens") \
       .withColumn("token_count", countTokens(col("tokens"))).show()

    #print(regexTokenized.select("user_id","concat_ws( , collect_list(tweet))", "tokens") \
    #    .withColumn("token_count", countTokens(col("tokens"))).toPandas().head())
    
    # Definig udf for steming use nltk porter stemmer

    p_stemmer = PorterStemmer()
    def stem(x):
        stemmed_tokens = [p_stemmer.stem(i) for i in x]
        return stemmed_tokens   
    stem_udf = udf(lambda x:stem(x), ArrayType(StringType()))

    # Stemming tokens
    stemmedTokens.withColumn("Stemmed_tokens", stem_udf('tokens'))

    token_count.select("user_id","concat_ws( , collect_list(tweet))", "tokens","Stemmed_tokens").show()

    # Defining model for stopwords

    stopWords_remover = StopWordsRemover(inputCol="Stemmed_tokens", outputCol="filtered")
    default_StopWords=remover.getStopWords()
    default_StopWords.append("https")

    # Removing Stopwords

    filtered_df = stopWords_remover.transform(regexTokenized)

    filtered_df.withColumn("Pre_tokens", countTokens(col("Stemmed_tokens"))).withColumn("Post_tokens", countTokens(col("filtered"))).show()

    #print(filtered_df.withColumn("Pre_tokens", countTokens(col("Stemmed_tokens"))).withColumn("Post_tokens", countTokens(col("filtered"))).toPandas().head())
    
    # Defining model to convert text documents to vectors of token counts

    countVect = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=1000, minDF=5)

    model = countVect.fit(filtered_df)

    vectorizer  = model.transform(filtered_df).select("user_id", "features")
    vectorizer.show(10)
    
    # Initializing LDA topic Modeling

    lda = LDA(k=10,maxIter=10)
    lda_model=lda.fit(vectorizer)

    #topics = lda_model.topicsMatrix()

    ll = lda_model.logLikelihood(vectorizer)
    lp = lda_model.logPerplexity(vectorizer)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound on perplexity: " + str(lp))
    
    # Describe topics.
    topics = lda_model.describeTopics(5)

    print("The topics described by their top-weighted terms:")
    
    topics.show(truncate=False)

    # UDF for formating the topics for desired usage

    zip_ = udf(lambda x, y: list(zip(x, y)),ArrayType(StructType([
      # Adjust types to reflect data types
      StructField("first", IntegerType()),
      StructField("second", DoubleType())])))
    
    topics_df=topics.withColumn("tmp", zip_("termIndices", "termWeights")).withColumn("tmp", explode("tmp"))\
    				.select("topic", col("tmp.first").alias("termIndices"), col("tmp.second").alias("termWeights"))

    # Extracting documents vocabulary

    vocab=model.vocabulary

    # UDF for extracting words for term indices asssigned to each topic 
    words_ = udf(lambda x:vocab[x] )
    topics_df=topics_df.withColumn("Words", words_('termIndices'))
    topics_df=topics_df.groupBy("topic").agg(collect_list(col("Words")).alias("Words"),collect_list(col("termIndices")).alias("termIndices"),collect_list(col("termWeights")).alias("termWeights"))
    print("The topics described by their top-weighted terms:")
    print(topics_df.toPandas())
    

    # Shows the result
    transformed = lda_model.transform(vectorizer)
    transformed.show(truncate=False)

    # UDF to extract top topic for each user
    
    max_value =  udf(lambda x:max(x).item())
    max_index = udf(lambda x:x.argmax().item())

    User_Topics_df=transformed.withColumn("Topic_Prob", max_value("topicDistribution")).withColumn("Topic", max_index("topicDistribution")).select("user_id", "Topic_Prob","Topic","topicDistribution")
    
    User_Topics_df.show(truncate=False)

    # Number of users for each topic assigned to them
    User_Topics_df.groupBy("Topic").count().show()

  	# saving userprofile in csv output
    user_profile=User_Topics_df.join(topics_df,User_Topics_df.Topic == topics_df.topic)
    user_profile.toPandas().to_csv('user_profile.csv')    
    

if __name__ == "__main__":

    # Initializing spark session
    spark = SparkSession \
        .builder \
        .appName("User Profiling") \
        .config("spark.executor.memory", "3g") \
        .getOrCreate()
    
    # Importing cleaned data
    df= spark.read.json("tweet_sf.json")

    # Lower-casing text 
    df=df.withColumn("text", lower(col("text")))\
    #					.limit(100000*multiplier)

    df.registerTempTable("MainTable")
    tweet_df = spark.sql("SELECT user_id AS user_id, text AS tweet FROM MainTable WHERE text IS NOT NULL")

    # Grouping tweets for each user

    tweet_df=tweet_df.groupBy("user_id").agg(concat_ws(' ',collect_list("tweet")))    

    Topic_Modeling(tweet_df)
 	
 	# Terminating spark session
    spark.stop()		