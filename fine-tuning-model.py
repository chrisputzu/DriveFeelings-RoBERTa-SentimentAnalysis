import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, RobertaTokenizer, TrainingArguments, Trainer
import torch

class TwitterDataset(torch.utils.data.Dataset):

    """ Classe che rappresenta un dataset personalizzato per l'analisi del sentiment su Twitter """

    def __init__(self, dataframe, tokenizer, max_length):

        # Inizializzazione delle variabili necessarie
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["clean_text"].tolist()  # Estrazione dei testi puliti
        self.labels = dataframe["category"].tolist()  # Estrazione delle categorie
        self.max_length = max_length

    def __len__(self):

        # Ritorna la lunghezza del dataset
        return len(self.text)

    def __getitem__(self, index):

        # Estrazione dei dati per un determinato indice
        text = str(self.text[index])

        # Tokenizzazione e formattazione dei dati di input
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        label = self.labels[index]

        # Restituzione dei dati nel formato atteso per il modello
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class TwitterSentimentAnalysis:

    """ Classe che rappresenta l'analisi del sentiment su Twitter """

    def __init__(self, train_file, max_length=100, num_labels=3):

        # Caricamento dei dati di addestramento
        self.load_training_data(train_file)

        # Preparazione del modello e del tokenizzatore
        self.prepare_model_and_tokenizer(num_labels)

        # Creazione del dataset di addestramento e del dataset di test
        self.dataset_train = TwitterDataset(self.train_data, self.tokenizer, max_length)
        self.dataset_test = TwitterDataset(self.test_data, self.tokenizer, max_length)

        # Configurazione del trainer
        self.setup_trainer()

    def load_training_data(self, train_file):

        """ Funzione che carica i dati da un file CSV e li divide in set di addestramento e test set """

        # Caricamento dei dati da un file CSV e suddivisione in set di addestramento e test
        df = pd.read_csv(train_file)[:30000]

        # Mappatura delle categorie in un formato specifico
        df['category'] = df['category'].map({1: 2, 0: 1, -1: 0})

        # Divisione in set di addestramento e test
        self.train_data, self.test_data = train_test_split(df, test_size=0.33, random_state=13)

    def prepare_model_and_tokenizer(self, num_labels):

        """ Caricamento del modello e del tokenizzatore preaddestrati """

        # Caricamento del modello e del tokenizzatore preaddestrati
        self.MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.MODEL)
        self.model = RobertaForSequenceClassification.from_pretrained(self.MODEL, num_labels=num_labels)

    def setup_trainer(self):

        """ Funzione per configurare ed eseguire il training """

        # Configurazione degli argomenti di addestramento
        self.training_args = TrainingArguments(
            output_dir="./DriveFeelings-Roberta-sentiment-analyzer-for-twitter",
            num_train_epochs=5,
            per_device_train_batch_size=6,
            per_device_eval_batch_size=2,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        # Inizializzazione del trainer per l'addestramento del modello
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,
            tokenizer=self.tokenizer,
        )

        # Avvio dell'addestramento del modello
        self.trainer.train()

        # Pubblicazione del modello su directory cloud HuggingFace.co
        self.trainer.push_to_hub()

if __name__ == '__main__':

    # File di input per il fine-tuning
    input_file = 'Twitter_Data.csv'

    # Chiamata di funzione per l'analisi del sentiment su Twitter
    sentiment_analysis = TwitterSentimentAnalysis(input_file)



