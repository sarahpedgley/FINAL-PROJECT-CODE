from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

class Vectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.X = None 
    
    def fit_transform(self, text: List[str]):
        self.X = self.vectorizer.fit_transform(text)
        return self.X
    
    def transform(self, text: List[str]):
        return self.vectorizer.transform(text)
    
    def get_X(self):
        if self.X is None:
            raise ValueError("X is not defined. Make sure to call fit_transform first.")
        return self.X