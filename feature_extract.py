import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold

cleaned_df = pd.read_csv('resources/cleaned_tweets.csv')

tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
tf_idf_vectorizer.fit(cleaned_df['cleaned_text'])
tf_idf_df = tf_idf_vectorizer.transform(cleaned_df['cleaned_text'])
tf_idf_df = pd.DataFrame(tf_idf_df.toarray(),columns=tf_idf_vectorizer.get_feature_names_out())
print('Shape before VarianceThreshold',tf_idf_df.shape)

var_thr = VarianceThreshold(threshold=0.0005)
var_thr.fit(tf_idf_df)
var_df = tf_idf_df[var_thr.get_feature_names_out()]

print('Shape after VarianceThreshold',var_df.shape)
print(len(tf_idf_vectorizer.get_feature_names_out())-len(var_thr.get_feature_names_out()),'features removed.')

var_df.to_csv('resources/var_tf_idf.csv', encoding='utf-8', index=False)