import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:

    """Questa classe è responsabile del pre-processing dei testi presenti nei tweet."""
    def __init__(self):
        """ Inizializza l'oggetto TextPreprocessor."""
        # Download dei dati necessari da NLTK
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Inizializzazione del lemmatizzatore e delle stopwords in lingua inglese di WordNet
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Stopwords aggiuntive non influenti da eliminare
        self.filename_tokens = ["bmw", "renault", "tesla", "new", "like", "one", "know",
                                "u", "musk","na", "yes", "baby", "amp", "car", "time"]

    def preprocess_text(self, text):

        """Esegue il pre-processing del testo del tweet."""

        # Trasformazione di tutti i caratteri in minuscolo
        text = text.lower()

        # Rimozione delle occorrenze di 'http' e 'https' per eliminare i link
        text = text.replace('http', '').replace('https', '')

        # Rimozione delle parole che iniziano con '@' per eliminare i tag degli utenti
        text = ' '.join(word for word in text.split() if not word.startswith('@'))

        # Tokenizzazione del testo in parole
        tokens = word_tokenize(text)

        # Rimozione di punteggiature, caratteri speciali e numeri
        tokens = [word for word in tokens if word.isalpha()]

        # Rimozione delle stopwords in lingua inglese e conversione in minuscolo
        tokens = [word for word in tokens if word.lower() not in self.stop_words]

        # Rimozione delle parole presenti nella lista filename_tokens
        tokens = [word for word in tokens if word.lower() not in self.filename_tokens]

        # Lemmatizzazione delle parole
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Ricostruzione del testo pre-processato
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

class DataPreprocessor:

    """Questa classe è responsabile del pre-processing dei file CSV contenenti tweets."""

    def __init__(self, input_files, output_files, input_folder, output_folder):

        """ Inizializza l'oggetto DataPreprocessor."""

        self.input_files = input_files
        self.output_files = output_files
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.text_preprocessor = TextPreprocessor()

    def preprocess_csv(self, input_filename, output_filename):

        """Esegue il pre-processing di un file CSV di input e salva il risultato in un nuovo file."""

        # Legge il file CSV in un DataFrame
        input_path = os.path.join(self.input_folder, input_filename)
        df = pd.read_csv(input_path)

        # Applicazione del pre-processing alla colonna 'Content' del DataFrame
        df['Content'] = df['Content'].apply(self.text_preprocessor.preprocess_text)

        # Salva il DataFrame pre-processato in un nuovo file CSV
        output_path = os.path.join(self.output_folder, output_filename)
        df.to_csv(output_path, index=False)

    def preprocess_all(self):

        """Esegue il pre-processing per tutti i file di input."""

        for input_file, output_file in zip(self.input_files, self.output_files):
            self.preprocess_csv(input_file, output_file)

if __name__ == "__main__":

    # Files di input sporchi
    input_files = [
        "tweets-BMW-en.csv",
        "tweets-Renault-en.csv",
        "tweets-Tesla-en.csv"
    ]

    # Files di output puliti
    output_files = [
        "tweets_puliti-BMW-en.csv",
        "tweets_puliti-Renault-en.csv",
        "tweets_puliti-Tesla-en.csv"
    ]

    # Directory input
    input_folder = "./tweets_estratti"

    # Directory output
    output_folder = "./tweets_puliti"

    # Creazione di un oggetto DataPreprocessor
    data_preprocessor = DataPreprocessor(input_files, output_files, input_folder, output_folder)

    # Esecuzione del pre-processing per tutti i file di input
    data_preprocessor.preprocess_all()

    # Stampa debug pre-processing eseguito correttamente
    print("Pre-processing completato.")
