import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from transformers import pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalyzer:

    """Classe per la sentiment analysis con modello RoBERT """

    def __init__(self, input_files):

        self.input_files = input_files

    def analyze_sentiment_roberta(self, texts):

        """Funzione che analizza il sentiment dei testi utilizzando RoBERTa e restituisce etichette."""

        # Pipeline modello per analizzare il sentiment dei testi
        pipe = pipeline("text-classification",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        max_length=512)

        sentiment_roberta = []

        for text in tqdm(texts, desc=f"Calcolo sentiment per file {input_file}", unit="testo"):
            result = pipe([text])
            sentiment_roberta.append(result[0]['label'])

        # Return del calcolo delle etichette di sentiment
        return sentiment_roberta


if __name__ == "__main__":

    # Directory lavoro di input e output
    input_folder = "./tweets_puliti"
    output_folder = "./tweets_sentiment"

    # Oggetto Classe utilizzato per le predizioni
    sentiment_analyzer = SentimentAnalyzer(input_folder)

    # Elaborazione dei file nella directory di input
    for input_file in os.listdir(input_folder):

        if input_file.endswith('.csv'):

            # Nome file semplificato per stampe debug e grafici
            target_name_input = input_file.split('-')[-2]
            input_filepath = os.path.join(input_folder, input_file)

            # Import file csv
            df = pd.read_csv(input_filepath)

            # Conversione della serie 'Content' in una lista di stringhe
            texts = df['Content'].astype(str).tolist()

            # Analisi del sentiment per i testi nel DataFrame
            sentiments = sentiment_analyzer.analyze_sentiment_roberta(texts)

            # Aggiunta delle etichette di sentiment al DataFrame
            df['Sentiment_Roberta'] = sentiments

            # Salvataggio dei risultati in un nuovo file CSV nella cartella di output
            output_filename = f"roberta-sentiment-analysis-{target_name_input}.csv"
            output_filepath = os.path.join(output_folder, output_filename)
            df.to_csv(output_filepath, index=False)
            print(f'File di output {output_filename} salvato con successo in {output_filepath}')

            # Conteggio delle etichette di sentiment
            roberta_counts = df['Sentiment_Roberta'].value_counts()

            # Grafici a barre per visualizzare le distribuzioni del sentiment per tutti i file csv
            plt.figure(figsize=(12, 10))
            sns.barplot(x=roberta_counts.index, y=roberta_counts.values, palette="Set2")
            plt.title(f"Sentiment con RoBERTa per {target_name_input}", fontsize=30)
            plt.ylabel('Frequenza assoluta', fontsize=24)
            plt.xlabel('Sentiment RoBERTa', fontsize=24)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

            # Stampe debug percentuali tweets per etichetta di sentiment per ogni file csv
            totale = float(roberta_counts['positive'] + roberta_counts['neutral'] + roberta_counts['negative'])
            print(f'\nFrequenze relative per le etichette di sentiment di {target_name_input}:')
            print(f"Sentiment Positivi: {round((roberta_counts['positive'] / totale * 100), 2)}%\n"
                  f"Sentiment Neutri: {round((roberta_counts['neutral'] / totale * 100), 2)}%\n"
                  f"Sentiment Negativi: {round((roberta_counts['negative'] / totale * 100), 2)}%\n")






