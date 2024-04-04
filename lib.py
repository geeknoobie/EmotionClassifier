# emotion_classifier.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests

class EmotionClassifier:
    def __init__(self, model_url='https://github.com/geeknoobie/Emotionmodel/raw/main/model.keras',
                       tokenizer_url='https://github.com/geeknoobie/Emotionmodel/raw/main/tokenizer.pkl',
                       index_to_label_url='https://github.com/geeknoobie/Emotionmodel/raw/main/index_to_label.pkl'):
        self.model = self.load_model_from_url(model_url)
        self.tokenizer = self.load_tokenizer_from_url(tokenizer_url)
        self.index_to_label = self.load_index_to_label_from_url(index_to_label_url)

    def load_model_from_url(self, url):
        response = requests.get(url)
        with open('model.keras', 'wb') as f:
            f.write(response.content)
        return load_model('model.keras')

    def load_tokenizer_from_url(self, url):
        response = requests.get(url)
        with open('tokenizer.pkl', 'wb') as f:
            f.write(response.content)
        with open('tokenizer.pkl', 'rb') as f:
            return pickle.load(f)

    def load_index_to_label_from_url(self, url):
        response = requests.get(url)
        with open('index_to_label.pkl', 'wb') as f:
            f.write(response.content)
        with open('index_to_label.pkl', 'rb') as f:
            return pickle.load(f)

    def predict_emotion(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        sequence_padded = pad_sequences(sequence, maxlen=self.model.input_shape[1])
        prediction = self.model.predict(sequence_padded)[0]
        predicted_index = np.argmax(prediction)
        return self.index_to_label[predicted_index]
