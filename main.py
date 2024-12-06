from genre_classifier import GenreClassifier
from models import EnsembleModel, NaiveBayesModel, LogisticRegressionModel, SVMModel
from vectorizer import Vectorizer
import os

def main():
    
    print("Literary genre classifier.")
    
    # initialize components
    vectorizer = Vectorizer()
    model = EnsembleModel()  # can replace with LogisticRegressionModel or SVMModel or naive bayes while developing
    genre_labels = ["fantasy", "sci-fi", "horror", "thriller", "mystery", "romance"]
    
    classifier = GenreClassifier(model, vectorizer, genre_labels)
    
    training_dir = "C:\\Users\\pedgl\\OneDrive\\Documents\\Uni\\Final Year Project\\FINAL PROJECT CODE\\training_data"
    texts, labels = classifier.load_training_data(training_dir)

    if not texts or not labels:
        print("Failed to load training data. Exiting...")
        exit(1)
    
    # train the classifier
    texts = [text if text is not None else "" for text in texts]
    classifier.train(texts, labels)

    filename = input("Please enter the file name/location of the literary sample you would like to classify: ")
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            sample = file.read()  
    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        exit(1)
    
    prediction = classifier.predict(sample.lower())   
    print("Prediction:", prediction)
    
    # evaluate the classifier
    accuracy = classifier.evaluate(texts, labels)
    print("Accuracy:", accuracy)
    #(not sure if this is needed)

if __name__ == "__main__":
    main()