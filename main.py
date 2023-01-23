import os
import re
import logging
import uvicorn
from data_preprocessing import text_preprocess, tokenize, remove_stopwords, lemmatization, removelt2wordslength
import pandas as pd
from fastapi import FastAPI, HTTPException, APIRouter
import tensorflow
import config
from data_validation import Textrequest, Output
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from config import decode_text
from keras.models import load_model


model = load_model("/home/rahul/Downloads/NLP/lstm/lstm.h5")

app = FastAPI(
    title=config.APP_TITLE, version=config.APP_VERSION, debug=config.IS_DEBUG, description=config.DESCRIPTION
)

# The Homepage of Api
@app.get('/')
async def index():
    return "Welcome to Classification of text Cybersecurity  & Notcybersecurity API"

@app.post("/predict-text")
def predict_sentiment(review: Textrequest):

    df = pd.DataFrame(review.sentence, columns=['text1'])
    print(df)

    # Text Preprocessing
    text_prpocess_obj = text_preprocess()
    df.text1 = df.text1.swifter.apply(lambda x: text_prpocess_obj.preprocessText(x))

    # Tokenization of words
    df.text1 = df.text1.swifter.apply(lambda x: tokenize(x))

    # Remove Stopwords
    df.text1 = df.text1.swifter.apply(lambda x: remove_stopwords(x))

    # Lemmatization
    lemmatization_obj = lemmatization()
    df.text1 = df.text1.swifter.apply(lambda x: lemmatization_obj.lemmatizing_space(x))

    # remove words from a string length between 2
    df.text1 = df.text1.swifter.apply(lambda x: removelt2wordslength(x))

    # TOKENIZATION
    tokk = Tokenizer()
    tokk.fit_on_texts(df.text1.values)
    seq = tokk.texts_to_sequences(df.text1.values)
    seqmatrix = pad_sequences(seq, padding='post', maxlen=config.MAX_SEQUENCE_LENGTH)


    # perform prediction
    prediction = model.predict(seqmatrix)
    Y_pred_othertext = [decode_text(score) for score in prediction]

    # prediction_probability = model.predict_proba([seqmatrix])
    #
    # output = Y_pred_othertext
    #
    # # show results
    # result = {"prediction": Y_pred_othertext, "Probability": output}
    return Y_pred_othertext


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)


