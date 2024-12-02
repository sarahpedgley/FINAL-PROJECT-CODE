from models import EnsembleModel
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

class GenreClassifier:
    def __init__(self, model: Model, vectorizer: Vectorizer, genre_labels: List[str]):
        self.model = model
        self.vectorizer = vectorizer
        self.genre_labels = genre_labels

    def load_training_data(self, directory: str) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []

        file_to_genre = {
            "the wonderful wizard of oz.txt": "fantasy",
            "the war of the worlds.txt": "sci-fi",
            "carmilla.txt": "horror",
            "the hound of the baskervilles.txt": "thriller",
            "a study in scarlet.txt": "mystery",
            "pride and prejudice.txt": "romance",
        }

        try:
            for filename, genre in file_to_genre.items():
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts.append(file.read().strip())
                    labels.append(genre)
            if not texts:
                print("No .txt files found in the directory.")
                exit(1)
        except FileNotFoundError:
            print(f"Error: Directory '{directory}' not found.")
            exit(1)
        except Exception as e:
            print(f"An error occurred while loading files: {e}")
            exit(1)

        return texts, labels

    
    def preprocess(self, text: str) -> str:
        ##there is also a predefined preprocess tool in the sci-kit library ? should that be used ?
                
        #downloads
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet') 
        nltk.download('stopwords') 
        
        #lowercase
        text = text.lower()
        #print(text)
        
        #change numbers to words
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
        #print(text)
        
        #remove punctuation/special characters ?? or is this useful for e.g romance ?
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        
        #remove whitespace
        text = " ".join(text.split())
        #print(text) #works up until this point
        
        #tokenise and remove stopwords
        word_tokens = word_tokenize(text)
        stopwords = set(stopwords.words("english")) #getting an error here 
        filtered = [word for word in word_tokens if word not in stopwords]
        text = filtered
        
        #stemming (getting the root form of a word) (i don't think this is necessary if i have lemmatisation)
        #stemmer = PorterStemmer()
        #word_tokens = word_tokenize(text)
        #stems = [stemmer.stem(word) for word in word_tokens]
        #text = stems
        
        #lemmatisation
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
        text = lemmas
        #print(text)
    
    def train(self, X: List[str], y: List[str]) -> None:
        X_vectorized = self.vectorizer.fit_transform(X)
        self.model.fit(X_vectorized, y)
        #train each model
    
    def predict(self, text: str) -> str:
        text_vectorized = self.vectorizer.transform([self.preprocess(text)])
        prediction = self.model.predict(text_vectorized)
        return prediction[0]
    
    def evaluate(self, X: List[str], y: List[str]) -> float:
        X_vectorized = self.vectorizer.transform(X)
        predictions = self.model.predict(X_vectorized)
        return accuracy_score(y, predictions)
    #on second thoughts is this method needed? it's not currently part of the plan
    #other thoughts - display summary of key words/etc