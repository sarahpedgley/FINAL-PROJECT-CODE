from models import EnsembleModel, NaiveBayesModel, LogisticRegressionModel, SVMModel, DictionaryAlgorithm
from vectorizer import Vectorizer
from models import Model
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import os
import nltk
import string
import re
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True) 
nltk.download('stopwords', quiet=True) 

class GenreClassifier:
    def __init__(self, model: Model, vectorizer: Vectorizer, genre_labels: List[str]):
        self.model = model
        self.vectorizer = vectorizer
        self.genre_labels = genre_labels

    def load_training_data(self, directory: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []

        file_to_genre = {
            "Fantasy": [
                "the wonderful wizard of oz.txt",
                "alice's adventures in wonderland.txt",
                "a journey to the centre of the earth.txt",
                "peter pan.txt",
                "the marvelous land of oz.txt",
                "through the looking-glass.txt",
                "the wind in the willows.txt",
                "gulliver's travels.txt",
                "the lost world.txt",
                "the mabinogion.txt"
                
            ],
            "Sci-fi": [
                "the war of the worlds.txt",
                "twenty thousand leagues under the sea.txt",
                "the time machine.txt",
                "RUR.txt",
                "the eyes have it.txt",
                "the island of doctor moreau.txt",
                "the hanging stranger.txt",
                "the iron heel.txt",
                "a princess of mars.txt",
                "the scarlet plague.txt"
            ],
            "Horror": [
                "carmilla.txt",
                "metamorphosis.txt",
                "dracula.txt",
                "the castle of otranto.txt",
                "the mysteries of udolpho.txt",
                "the trial.txt",
                "25 ghost stories.txt",
                "the jewel of seven stars.txt",
                "frankenstein.txt",
                "jekyll and hyde.txt"
            ],
            "Thriller": [
                "the hound of the baskervilles.txt",
                "the lonely house.txt",
                "benighted.txt",
                "knock three-one-two.txt",
                "the mill of silence.txt",
                "the three just men.txt",
                "ferdinand.txt",
                "the fall of the house of usher.txt",
                "the man who couldn't sleep.txt",
                "the picture of dorian gray.txt"
            ],
            "Mystery": [
                "a study in scarlet.txt",
                "the murder of roger ackroyd.txt",
                "the memoirs of sherlock holmes.txt",
                "the murder on the links.txt",
                "the return of sherlock holmes.txt",
                "the mysterious affair at styles.txt",
                "the draycott murder mystery.txt",
                "the moonstone.txt",
                "the shadow of the wolf.txt",
                "the valley of fear.txt"
            ],
            "Romance": [
                "pride and prejudice.txt",
                "the blue castle.txt",
                "middlemarch.txt",
                "the lady of the lake.txt",
                "the romance of lust.txt",
                "wuthering heights.txt",
                "emma.txt",
                "northanger abbey.txt",
                "the mill on the floss.txt",
                "the return of the native.txt"
            ],
        }

        try:
            for genre, files in file_to_genre.items():
                for filename in files:
                    file_path = os.path.join(directory, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            texts.append(file.read().strip())
                            labels.append(genre)
                    except FileNotFoundError:
                        print(f"File '{filename}' not found.")
                    except Exception as e:
                        print(f"An error occurred while reading file '{filename}': {e}")
            if not texts:
                print("No valid .txt files found in the directory.")
                exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            exit(1)

        return texts, labels
    
    def preprocess(self, text: str) -> str:
        
        #lowercase
        text = text.lower()
        
        #change numbers to words - again remove because it may be helpful for sci-fi
        x = inflect.engine()
        temp_string = text.split()
        new_string = []
        for word in temp_string:
            if word.isdigit():
                temp = x.number_to_words(word)
                new_string.append(temp)
            else:
                new_string.append(word)

        text = ' '.join(new_string)
        
        #remove punctuation/special characters ?? or is this useful for e.g romance ?
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        #remove whitespace
        text = " ".join(text.split())
        
        #tokenise and remove stopwords
        from nltk.corpus import stopwords
        word_tokens = word_tokenize(text)
        stopwords = set(stopwords.words("english")) 
        filtered = [word for word in word_tokens if word not in stopwords]
        text = " ".join(filtered)
        
        #lemmatisation
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
        text = " ".join(lemmas)

        if text is None:
            return ""
        return text
    
    def train(self, X: List[str], y: List[str]) -> None:
        preprocessed_texts = [self.preprocess(text) for text in X]
        X_vectorized = self.vectorizer.fit_transform(preprocessed_texts)
        self.model.fit(X_vectorized, y)
    
    def predict(self, text: str) -> str:
        if isinstance(self.model, DictionaryAlgorithm):
            text_raw = text
            prediction = self.model.predict([text_raw])
        else:
            text_raw = self.preprocess(text)
            text_vectorized = self.vectorizer.transform([text_raw])
            prediction = self.model.predict(text_vectorized)
        return prediction[0]
    
    def evaluate(self, X: List[str], y: List[str]) -> float:
        X_vectorized = self.vectorizer.transform(X)
        predictions = self.model.predict(X_vectorized)
        return accuracy_score(y, predictions)