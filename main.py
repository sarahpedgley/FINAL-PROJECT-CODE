from genre_classifier import GenreClassifier
from models import EnsembleModel, NaiveBayesModel, LogisticRegressionModel, SVMModel, DictionaryAlgorithm
from models import genre_keywords
from vectorizer import Vectorizer
import os

def main():
    print("Literary genre classifier.")
    vectorizer = Vectorizer()
    genre_labels = ["Fantasy", "Sci-fi", "Horror", "Thriller", "Mystery", "Romance"]
    dictionary_algorithm = DictionaryAlgorithm(genre_keywords=genre_keywords)
    
    models = {
        "Naive Bayes": NaiveBayesModel(),
        "Logistic Regression": LogisticRegressionModel(),
        "SVM": SVMModel(),
        "Ensemble": EnsembleModel(),
        "Dictionary": dictionary_algorithm
    }
    
    #training_dir = "C:\\Users\\pedgl\\OneDrive\\Documents\\Uni\\Final Year Project\\FINAL PROJECT CODE\\training_data"
    training_dir = os.path.join(os.getcwd(), "training_data")

    # load training data 
    classifier = GenreClassifier(models["Naive Bayes"], vectorizer, genre_labels)
    texts, labels = classifier.load_training_data(training_dir)

    if not texts or not labels:
        print("Training data is empty.")
        exit(1)
    
    # train classifiers
    for model_name, model in models.items():
        try:
            classifier = GenreClassifier(model, vectorizer, genre_labels)
            classifier.train(texts, labels)
        except Exception as e:
            print(f"Error loading training data: {e}")
            exit(1)

    # classify sample text
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

    # loop through each model so we get results for each one individually
    print("\nResults of each model:")
    for model_name, model in models.items():
        try:
            if isinstance(model, DictionaryAlgorithm):
                prediction = model.predict([sample.lower()])[0]
            else:
                classifier = GenreClassifier(model, vectorizer, genre_labels)
                prediction = classifier.predict(sample.lower())
            print(f"\n{model_name} Prediction: {prediction}")
        except Exception as e:
            print(f"Error with model '{model_name}': {e}")

if __name__ == "__main__":
    main()