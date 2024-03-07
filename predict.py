import streamlit as st
import pickle
import warnings
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ordered_set import OrderedSet
import re

import nltk
nltk.download('punkt')
# nltk.download('stopwords')

warnings.filterwarnings('ignore')
from PIL import Image

tf_idf_vect = pickle.load(open('resources/tfidf_vect.pkl','rb'))
var_thr = pickle.load(open('resources/var_thr.pkl','rb'))

dtc_model = pickle.load(open('resources/dtc_model.pkl','rb'))
knn_model = pickle.load(open('resources/knn_model.pkl','rb'))
nb_model = pickle.load(open('resources/nb_model.pkl','rb'))
svm_model = pickle.load(open('resources/svm_model.pkl','rb'))

# stop_dict = set(stopwords.words('English'))

def clean_text(text):
    ps = PorterStemmer()
    t = text
    t = re.sub(r'[^a-zA-Z]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    t = t.lower()
    t = t.strip()
    t = word_tokenize(t)
    # t = list(OrderedSet(t) - stop_dict)
    t = [word for word in t if len(word)>2]
    t = [ps.stem(w) for w in t]
    t = ' '.join(t)
    
    return t

def feature_extraction_selection(text):
    tf_idf_df = tf_idf_vect.transform(text)
    tf_idf_df = pd.DataFrame.sparse.from_spmatrix(tf_idf_df,columns=tf_idf_vect.get_feature_names_out())
    tf_idf_df = tf_idf_df[tf_idf_vect.get_feature_names_out()]
    # var_df = tf_idf_df[var_thr.get_feature_names_out()]

    return tf_idf_df

def app():
    text = st.text_input('text','')

    dtc_result = ''
    knn_result = ''
    nb_result = ''
    svm_result = ''

    if st.button('Click here to predict'):
        cleaned_text = feature_extraction_selection([clean_text(text)])
        print('dtc')
        dtc_result = dtc_model.predict(cleaned_text)
        print('knn')
        knn_result = knn_model.predict(cleaned_text)
        print('nb')
        nb_result = nb_model.predict(cleaned_text)
        print('svm')
        svm_result = svm_model.predict(cleaned_text)
        st.balloons()
    st.success('Decision Tree Classifier: {}'.format(dtc_result))
    st.success('KNN: {}'.format(knn_result))
    st.success('Naive Bayes: {}'.format(nb_result))
    st.success('Support Vector Machines: {}'.format(svm_result))

if __name__ == '__main__':
    app()