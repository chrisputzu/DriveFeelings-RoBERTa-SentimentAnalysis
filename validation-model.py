import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def validate_model_fine_tuned(test_file):

    """ Funzione per validare il modello fine tuned DriveFeelings """

    # Caricamento del modello per modello fine-tuned
    tokenizer = RobertaTokenizer.from_pretrained("bibbia/DriveFeelings-Roberta-sentiment-analyzer-for-twitter")
    model = RobertaForSequenceClassification.from_pretrained("bibbia/DriveFeelings-Roberta-sentiment-analyzer-for-twitter")
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Caricamento dei dati per modello fine-tuned
    test_file = pd.read_csv(test_file)[:10000]

    # Calcolo predizioni per modello fine-tuned
    predictions = [pipe(text) for text in test_file['text']]
    extracted_predictions = [result[0]['label'] for result in predictions]

    # Calcolo accuratezza per modello fine-tuned
    true_sentiments = test_file['airline_sentiment']
    accuracy = accuracy_score(true_sentiments, extracted_predictions)
    matrix_c = confusion_matrix(true_sentiments, extracted_predictions)
    print(f"Accuracy su per modello {fine_tuned}: {round(accuracy * 100, 2)}%")
    print(f"Confusion Matrix per modello {fine_tuned}: \n{matrix_c}")

    # Creazione del report di classificazione con RoBERTa modello fine-tuned
    roberta_classification_report = classification_report(true_sentiments, extracted_predictions)
    print(f"RoBERTa Classification Report per {fine_tuned}:")
    print(roberta_classification_report)

    # Grafico Matrice di confusione con mappa di calore per modello fine-tuned
    plt.figure(figsize=(8, 6))
    hmap = sns.heatmap(matrix_c, annot=True, fmt='d', cmap='Blues')
    class_names = ['Negativo', 'Neutro', 'Positivo']
    hmap.set_xticklabels(class_names, rotation=0, fontsize=10)
    hmap.set_yticklabels(class_names, rotation=0, fontsize=10)
    plt.ylabel('Sentiment Vero', fontsize=14)
    plt.xlabel('Sentiment Predetto', fontsize=14)
    plt.title(f'Matrice di confusione per {fine_tuned}', fontsize=16)
    plt.show()

def validate_model_pre_trained(test_file):

    """ Funzione per validare il modello pre-addestrato """

    # Caricamento del modello pre-trained
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = RobertaForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest")
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Caricamento dei dati
    test_file = pd.read_csv(test_file)[:10000]

    # Calcolo predizioni per modello pre-trained
    predictions = [pipe(text) for text in test_file['text']]
    extracted_predictions = [result[0]['label'] for result in predictions]

    # Calcolo accuratezza per modello pre-trained
    true_sentiments = test_file['airline_sentiment']
    accuracy = accuracy_score(true_sentiments, extracted_predictions)
    matrix_c = confusion_matrix(true_sentiments, extracted_predictions)
    print(f"Accuracy su per modello {pre_trained}: {round(accuracy * 100, 2)}%")
    print(f"Confusion Matrix per modello {pre_trained}: \n{matrix_c}")

    # Creazione del report di classificazione con RoBERTa modello pre-trained
    roberta_classification_report = classification_report(true_sentiments, extracted_predictions)
    print(f"RoBERTa Classification Report per {pre_trained}:")
    print(roberta_classification_report)

    # Grafico Matrice di confusione con mappa di calore per modello pre-trained
    plt.figure(figsize=(8, 6))
    hmap = sns.heatmap(matrix_c, annot=True, fmt='d', cmap='Blues')
    class_names = ['Negativo', 'Neutro', 'Positivo']
    hmap.set_xticklabels(class_names, rotation=0, fontsize=10)
    hmap.set_yticklabels(class_names, rotation=0, fontsize=10)
    plt.ylabel('Sentiment Vero', fontsize=14)
    plt.xlabel('Sentiment Predetto', fontsize=14)
    plt.title(f'Matrice di confusione per {pre_trained}', fontsize=16)
    plt.show()

if __name__ == '__main__':

    # File di validazione
    test_file = 'Test_airlines-sentiment.csv'

    # Variabili per stampe debug e grafici
    fine_tuned = 'fine_tuned'
    pre_trained = 'pre_trained'

    # Chiamata di funzione per validare modello fine tuned
    validate_model_fine_tuned(test_file)

    # Chiamata di funzione per validare modello pre-trained
    validate_model_pre_trained(test_file)

