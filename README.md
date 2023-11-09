# DriveFeelings: Cosa pensano gli utenti di Twitter su BMW, Renault e Tesla? #

Il progetto analizza i testi dei tweets degli utenti del social network Twitter che contengono le parole chiave 
di tre case automobilistiche scelte casualmente tra le più rilevanti del settore:

1. BMW
2. Renault
3. Tesla

L'obiettivo del progetto è quello di capire cosa pensano gli utenti riguardo queste auto, quale è la più apprezzata
o la meno apprezzata, ma anche gli argomenti a cui fanno riferimento ed infine implementare un classificatore di sentiment
con il modello pre-addestrato di RoBERTa.

-------------------------


### Analisi da svolgere ###

Il progetto si compone di 4 fasi e ad ogni fase corrisponde un branch dedicato:

1. TwitterScraper: Estrazione 10.000 tweets per ogni casa automobilistica, per un Totale 30.000.
2. PreProcessing: Pulitura dei tweets per la predisposizione all'analisi di sentiment e topic.
3. TopicModeling: Per raggruppare i tweets estratti in 5 categorie di argomenti o topic.
4. RoBERTa SentimentAnalysis: sentiment analysis effettuata con modello RoBERTa considerando ogni etichetta (positive/neutral/negative) per 
   ogni casa automobilistica e confrontando l'accuratezza tra i modelli pre-trained e fine-tuned, così da etichettare i dati estratti utilizzando il modello più performante.

-------------------------
