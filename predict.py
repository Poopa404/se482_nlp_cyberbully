import streamlit as st
import pickle
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
from PIL import Image

tfidf_vect = pickle.load(open('resources/tfidf_vect.pkl','rb'))
var_thr = pickle.load(open('resources/var_thr.pkl','rb'))

dtc_model = pickle.load(open('resources/dtc_model.pkl','rb'))
knn_model = pickle.load(open('resources/knn_model.pkl','rb'))
nb_model = pickle.load(open('resources/nb_model.pkl','rb'))
svm_model = pickle.load(open('resources/svm_model.pkl','rb'))

def feature_extraction_selection(text):
    tfidf_df = tfidf_vect.transform(text)
    return tfidf_df

def app():
    text = st.text_input('text','')

    result = ''
    if st.button('Click here to predict'):
        result = feature_extraction_selection(text)
        st.balloons()
    st.success('The output is {}'.format(result))

if __name__ == '__main__':
    app()