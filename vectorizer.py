from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from joblib import dump, load

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def fit_transform(self, text: List[str]):
        return self.vectorizer.fit_transform(text)
    
    def transform(self, text: List[str]):
        return self.vectorizer.transform(text) 
    
    def save(self, filepath: str):
        #saves the trained vectorizer to a file
        dump(self.vectorizer, filepath)
    
    def load(self, filepath: str):
        #loads trained vectorizer from a file
        self.vectorizer = load(filepath)