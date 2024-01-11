import numpy as np
import pandas as pd
import pickle
import streamlit as st

# load data
data_ = pd.read_csv("test.csv")

class_pred = pd.read_csv("Deploy_data.csv")

st.title("Quora Question pair Similarity")
idx = st.slider('Select an index to display question pair', 0, 1000)
q1 = data_.iloc[idx]["question1"]
q1_ = str(q1)
q2 = data_.iloc[idx]["question2"]
q2_ = str(q2)

st.subheader("These are the two questions to find if, Duplicate or not")
st.markdown("Question 1 :{}".format(q1_))
st.markdown("Question 2 :{}".format(q2_))

def predict_duplicate(idx):
    dup_class = class_pred['y_pred'].values[idx]
    probability = class_pred['probability'].values[idx]
    if dup_class == 0:
        return 'Not Duplicate', probability
    else:
        return 'Duplicate', probability



if st.button('Predict Duplicate or not'):
    class_pred, score = predict_duplicate(idx)
    st.success("The question pairs are : {}".format(class_pred.upper()))
    st.success("Prediction made with probability : {}".format(round(score,3)))
