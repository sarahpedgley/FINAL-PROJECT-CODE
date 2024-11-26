from models import EnsembleModel
from vectorizer import Vectorizer
from models import Model
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import os

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
        # preprocess the text (tokenisation etc)
        return text
    
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