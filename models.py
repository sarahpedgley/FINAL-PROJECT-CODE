from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
    
genre_keywords = {
    "Fantasy": ["dragon", "magic", "wizard", "castle", "sorcery", "witch", "curse", "hex", "castle", "royal", "princess", "prince", "palace", "country", "land", "wonder", "ancient", "prophecy", "god", "goddess", "elf", "fight", "sword", "battle", "hero", "quest", "knight", "alchemy", "rune", "beast", "forest", "spell", "enchantment", "fairy", "phoenix", "portal", "artifact"],
    "Sci-fi": ["spaceship", "mars", "martian", "moon", "space", "alien", "robot", "future", "quantum", "equation", "formula", "lunar", "solar", "world", "earth", "space", "tunnel", "fact", "theorem", "AI", "galaxy", "simulation", "technology", "dystopia", "utopia", "android", "cyber", "gravity", "universe", "federation", "clone", "hologram", "drone", "orbital", "colony"],
    "Horror": ["ghost", "haunted", "vampire", "zombie", "fear", "blood", "decapitated", "head", "body", "kill", "terror", "afraid", "limb", "grotesque", "organ", "ooze", "needle", "coffin", "grave", "cemetery", "funeral", "possession", "exorcism", "demon", "shadow", "creepy", "eerie", "dark", "monster", "creation", "nightmare", "evil", "clown", "ritual", "doll", "sacrifice", "apparition"],
    "Thriller": ["murder", "spy", "conspiracy", "detective", "chase", "dark", "quiet", "suspicious", "shadow", "gloom", "night", "race", "body", "heart", "sudden", "knife", "fright", "fog", "secrets", "betrayal", "twist", "pursuit", "alibi", "cover-up", "hostage", "plot", "revenge", "surveillance", "investigation", "danger", "threat", "intense", "criminal", "trap" ],
    "Mystery": ["mystery", "murder","clue", "investigation", "detective", "crime", "scene", "body", "police", "evidence", "robbery", "business", "affair", "puzzle", "knife", "gun", "vengeance", "suspect", "witness", "interrogation", "alibi", "secret", "motive", "scheme", "disappearance", "whodunit", "hidden", "case", "sleuth", "trap", "detective", "testimony", "timeline", "autopsy", "solved", "files", "report", "journal", "lead","forensics", "evidence", "inspector"],
    "Romance": ["love", "marry", "romance", "passion", "kiss", "heart", "husband", "wife", "beauty", "inheritance", "partner", "estate", "house", "home", "hand", "daughter", "family", "sex", "pleasure", "embrace", "affection", "desire", "relationship", "romantic", "intimacy", "dream", "spark", "devotion", "adoration", "disposition", "cherish", "attraction", "wedding", "engagement", "flirtation", "admiration", "obsession", "destiny", "jealousy", "fate", "swoon", "happiness"]
}
        
class Model(ABC):
    @abstractmethod
    def fit(self, X: List[str], y: List[str]) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def score(self, X: List[str], y: List[str]) -> float:
        pass

class NaiveBayesModel(Model):
    def __init__(self):
        self.model = MultinomialNB()
    
    def fit(self, X, y) -> None:
        self.model.fit(X, y)
    
    #def predict(self, X) -> List[str]:
        #return self.model.predict(X)
    
    def predict(self, text):
        probabilities = self.model.predict_proba([text])[0] 
        genre_probabilities = list(zip(self.genre_labels, probabilities))
        genre_probabilities.sort(key=lambda x: x[1], reverse=True)
        return genre_probabilities
    
    def score(self, X, y) -> float:
        return self.model.score(X, y)

class LogisticRegressionModel(Model):
    def __init__(self):
        self.model = LogisticRegression()
    
    def fit(self, X, y) -> None:
        self.model.fit(X, y)
    
    #def predict(self, X) -> List[str]:
        #return self.model.predict(X)
    
    def predict(self, text):
        probabilities = self.model.predict_proba([text])[0] 
        genre_probabilities = list(zip(self.genre_labels, probabilities))
        genre_probabilities.sort(key=lambda x: x[1], reverse=True)
        return genre_probabilities
    
    def score(self, X, y) -> float:
        return self.model.score(X, y)
        
class SVMModel(Model):
    def __init__(self):
        self.model = SVC()
    
    def fit(self, X, y) -> None:
        self.model.fit(X, y)
    
    #def predict(self, X) -> List[str]:
        #return self.model.predict(X)
    
    def predict(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y) -> float:
        return self.model.score(X, y)
    
class EnsembleModel(Model):
    def __init__(self):
        self.naive_bayes = MultinomialNB()
        self.logistic_regression = LogisticRegression()
        self.svm = SVC(probability=True)  # probability estimates for voting

    def fit(self, X, y) -> None:
        # fit all models on same data
        self.naive_bayes.fit(X, y)
        self.logistic_regression.fit(X, y)
        self.svm.fit(X, y)

    def predict(self, X) -> List[List[Tuple[str, int]]]:
        # get predictions from each model
        nb_preds = self.naive_bayes.predict(X)
        lr_preds = self.logistic_regression.predict(X)
        svm_preds = self.svm.predict(X)
        
        # majority vote with frequency counts
        final_predictions = []
        for nb, lr, svm in zip(nb_preds, lr_preds, svm_preds):
            votes = [nb, lr, svm]
            vote_counts = Counter(votes)
            
            # sort genres by the number of votes in descending order
            sorted_votes = sorted(vote_counts.items(), key=lambda item: item[1], reverse=True)
            final_predictions.append(sorted_votes) 
        
        return final_predictions

    def score(self, X, y) -> float:
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
class DictionaryAlgorithm(): 
    def __init__(self, genre_keywords: Dict[str, List[str]]):
        self.genre_keywords = genre_keywords
    
    def count_keywords(self, text: str) -> Dict[str, int]:
        genre_scores = {genre: 0 for genre in self.genre_keywords} 
        for genre, keywords in self.genre_keywords.items():
            for keyword in keywords:
                genre_scores[genre] += text.lower().count(keyword.lower()) 
        return genre_scores
    
    def fit(self, X: List[str], y: List[str]) -> None:
        #not needed for this algorithm
        pass
    
    def predict(self, X) -> List[List[Tuple[str, int]]]:
        predictions = []
        for text in X:
            genre_scores = self.count_keywords(text)  # using raw text
            sorted_genres = sorted(genre_scores.items(), key=lambda item: item[1], reverse=True)
            predictions.append(sorted_genres) 
        return predictions
    
    def score(self, X, y) -> float:
        predictions = self.predict(X)
        return accuracy_score(y, predictions)