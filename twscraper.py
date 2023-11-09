import asyncio
from twscrape import API
import csv
import sys
import os
from contextlib import aclosing


class TwitterScraper:

    """ Classe per gestire l'estrazione dei tweets da twitter. """

    # Chiamata API alla libreria twscrape
    def __init__(self):
        """ Inizializza l'oggetto TwitterScraper e crea un'istanza di API per l'accesso alle funzionalità di twscrape. """
        self.api = API()

    async def login(self):

        """ Esegue il login a tutti gli account Twitter salvati nella pool.
            Nota: È necessario aggiungere gli account utilizzando il metodo `add_account` prima di eseguire il login."""

        # await self.api.pool.add_account("USERNAME", "USERNAME_PSW", "INDIRIZZO_EMAIL_ACCOUNT_TWITTER", "PSW_INDIRIZZO_EMAIL_ACCOUNT_TWITTER")
        # await self.api.pool.add_account("USERNAME", "USERNAME_PSW", "INDIRIZZO_EMAIL_ACCOUNT_TWITTER", "PSW_INDIRIZZO_EMAIL_ACCOUNT_TWITTER")
        # await self.api.pool.add_account("USERNAME", "USERNAME_PSW", "INDIRIZZO_EMAIL_ACCOUNT_TWITTER", "PSW_INDIRIZZO_EMAIL_ACCOUNT_TWITTER")
        # await self.api.pool.add_account("USERNAME", "USERNAME_PSW", "INDIRIZZO_EMAIL_ACCOUNT_TWITTER", "PSW_INDIRIZZO_EMAIL_ACCOUNT_TWITTER")
        # await self.api.pool.add_account("USERNAME", "USERNAME_PSW", "INDIRIZZO_EMAIL_ACCOUNT_TWITTER", "PSW_INDIRIZZO_EMAIL_ACCOUNT_TWITTER")

        await self.api.pool.login_all()

    async def scrape_tweets(self, query, max_tweets=10000):

        """ Esegue lo scraping dei tweet in base a una query specifica con un limite massimo di tweets. """

        # Lista principale tweets
        tweets_list = []

        # Contatore quantità tweets
        tweet_count = 0

        # Uscita dal programma quando raggiunge quantità tweets richiesto
        async with aclosing(self.api.search(query)) as gen:

            async for tweet in gen:

                # Stampa debug per ogni tweet estratto
                print(tweet.id, tweet.user.username, tweet.rawContent)

                # Tweet appeso in coda in lista tweets
                tweets_list.append([tweet.user.username, tweet.rawContent])

                # Contatore incrementato +1 per ogni tweet appeso in lista
                tweet_count += 1

                # Controllo condizione di uscita numero tweets
                if tweet_count == max_tweets:
                    break

        return tweets_list

    @staticmethod
    def write_tweets_to_csv(tweets_list, filename):

        """ Scrive i dati estratti su un file CSV. """

        with open(filename, 'w', newline='') as f:

            # Creazione oggetto per scrivere ogni tweet
            writer = csv.writer(f)

            # Scrittura etichette colonne username e contenuto tweet
            writer.writerow(["Username", "Content"])

            # Ciclo for per scrivere ogni tweet estratto su CSV
            for tweet in tweets_list:

                # Scrittura tweet
                writer.writerow(tweet)


if __name__ == "__main__":

    """ Gestione del flusso di esecuzione del programma. """

    # Creazione oggetto scraper contenente la classe TwitterScraper
    scraper = TwitterScraper()

    # Chiamata alla funzione login
    asyncio.run(scraper.login())  # Attendere che il login venga completato

    # Lista contenente le Query dei nomi delle case automobilistiche obiettivo seguito dalla lingua
    queries = ["BMW lang:en", "Renault lang:en", "Tesla lang:en"]

    # Cartella di destinazione per i file CSV di output
    output_folder = "tweets_estratti"
    os.makedirs(output_folder, exist_ok=True)

    # Ciclo for attraverso le query
    for query in queries:

        # Chiamata alla funzione di Scrapping scrape_tweets per ogni Query inserita
        tweets_list = asyncio.run(scraper.scrape_tweets(query))

        # Rinominazione filename
        filename = os.path.join(output_folder,f"tweets-{query.replace(' ', '-').replace(':', '').replace('lang', '')}.csv")

        # Stampa debug per salvataggio file csv
        print(f"Salvataggio CSV: {filename}")

        # Chiamata di funzione per salvataggio tweets estratti in 3 file csv, uno per ogni casa automobilistica obiettivo
        scraper.write_tweets_to_csv(tweets_list, filename)

    # Uscita dal programma
    sys.exit()
