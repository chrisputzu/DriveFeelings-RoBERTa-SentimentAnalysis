import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import GridSearchCV
import pyLDAvis.lda_model
import warnings
warnings.filterwarnings("ignore")

class TopicModelingAnalyzer:

    """Classe che gestisce la Topic Modeling"""

    def __init__(self, input_folder):

        # Inizializzazione directory di input
        self.input_folder = input_folder

    def get_input_files(self):

        """Funzione che restituisce una lista dei file CSV nella cartella di input."""

        input_files = []
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith(".csv"):
                    input_files.append(os.path.join(root, file))
        return input_files

    def perform_topic_modeling(self, input_filename):

        """Funzione che esegue la topic modeling su ogni file CSV specifico."""

        # Lettura del file CSV
        df = pd.read_csv(input_filename)

        # Gestione dei valori NaN nella colonna 'Content'
        df['Content'].fillna('', inplace=True)  # Sostituisci i valori NaN con stringhe vuote

        # Inizializzazione del vettorizzatore CountVectorizer per le frequenze delle parole (tf)
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tf = tf_vectorizer.fit_transform(df['Content'])

        return tf_vectorizer, tf, tf_vectorizer.transform(df['Content'])

    def display_topics(self, tf_vectorizer, lda, n_words=10):

        """Funzione che genera e restituisce i topic individuati dal modello LDA."""

        feature_names = tf_vectorizer.get_feature_names_out()
        topics = {}

        # Ciclo for per indicizzazione dei topic
        for topic_idx, topic in enumerate(lda.components_):
            top_n_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics[f"Topic {topic_idx + 1}"] = top_n_words

        return topics

    def create_wordcloud(self, input_filename):

        """Funzione che crea e visualizza i grafici word cloud basati sul contenuto del file CSV specifico."""

        # Lettura del file CSV
        df = pd.read_csv(input_filename)

        # Gestione dei valori NaN nella colonna 'Content'
        df['Content'].fillna('', inplace=True)  # Sostituzione dei valori NaN con stringhe vuote

        # Join dei testi
        long_string = ','.join(df['Content'].to_list())

        # Creazione oggetto WordCloud per generare i grafici
        wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
        wordcloud.generate(long_string)

        # Grafici Word Cloud
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud per {file_name}", fontsize=20)
        plt.show()



if __name__ == "__main__":

    # Directory di input
    input_folder = "./tweets_puliti"

    # Directory di output
    output_folder = "pagine_html_LDA"

    # Creazione di un oggetto TopicModelingAnalyzer
    topic_modeling_analyzer = TopicModelingAnalyzer(input_folder)

    # Ciclo for per automatizzare l'esecuzione del programma per ogni file
    input_files = topic_modeling_analyzer.get_input_files()
    for input_file in input_files:

        # Estrazione solo del nome del file CSV dal percorso completo
        file_name = os.path.basename(input_file).split('-')[-2]

        # Stampa di debug per vedere su quale file CSV stiamo lavorando
        print(f"\nTopic Modeling per il file input di '{file_name}' con numero di topics ottimale della GridSearchCV:\n")

        # Esecuzione GridSearchCV per trovare il numero di topic ottimale e LDA
        vectorizer, dtm, tf = topic_modeling_analyzer.perform_topic_modeling(input_file)
        searchParams = {'n_components': [5, 10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
        lda = LatentDirichletAllocation()
        model = GridSearchCV(lda, param_grid=searchParams, verbose=3, n_jobs=-1)
        model.fit(tf)
        best_lda_model = model.best_estimator_
        num_topics = best_lda_model.n_components

        # Stampa debug dei risultati del grid search
        print("\nMiglior Punteggio di verosimiglianza logaritmica (Log Likehoods):", model.best_score_)
        print("Parametri del modello migliore:", model.best_params_)
        print("Perplexity del modello:", best_lda_model.perplexity(tf))
        print(f"Numero di Topics Ottimale: {num_topics}\n")

        # Stampa di debug che mostra i topic e le keyword associate
        topics = topic_modeling_analyzer.display_topics(vectorizer, best_lda_model)
        for topic, words in topics.items():
            print(f"{topic}: {', '.join(words)}")

        # Parametri per grafici
        n_topics = searchParams['n_components']
        log_likelihoods = [round(model.cv_results_['mean_test_score'][index]) for index, gscore in
                           enumerate(model.cv_results_['params'])]

        # Grafici matplotlib dei modelli della GridSearchCV
        plt.figure(figsize=(12, 8))
        for decay in [0.5, 0.7, 0.9]:
            log_likelyhoods = [log_likelihoods[i] for i, gscore in enumerate(model.cv_results_['params']) if
                               gscore['learning_decay'] == decay]
            plt.plot(n_topics, log_likelyhoods, label=f'Learning Decay {decay}')

        plt.title(f"Scelta del modello LDA ottimale per {file_name}", fontsize=20)
        plt.xlabel("Numero di Topics", fontsize = 12)
        plt.ylabel("Punteggi di verosimiglianza logaritmica (Log Likehoods)", fontsize = 12)
        plt.legend(loc='best', fontsize = 12)
        plt.show()

        # Generazione del pannello pyLDAvis
        panel = pyLDAvis.lda_model.prepare(lda_model=best_lda_model, dtm=tf, vectorizer=vectorizer)

        # Creazione pagina HTML del modello LDA
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_html_path = os.path.join(output_folder, f'LDA_panel_{file_name}.html')
        pyLDAvis.save_html(panel, output_html_path)

        # Grafici Wordcloud
        topic_modeling_analyzer.create_wordcloud(input_file)


















