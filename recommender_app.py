import pickle
import gradio as gr
import xgboost as xgb
from google.colab import drive

model = pickle.load(open('models/xgb.sav', 'rb'))

def pred(commentaire):
    sentiment = model.predict([commentaire])
    return sentiment


demo = gr.Interface(fn=pred, inputs="text", outputs="text")
demo.launch()
