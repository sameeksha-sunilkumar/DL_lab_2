import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('punkt')
corpus = ["Natural Language Processing (NLP) is a field of artificial intelligence (AI) that focuses on enabling computers to understand, interpret, and generate human language. By combining techniques from linguistics, computer science, and machine learning, NLP allows machines to process and analyze large amounts of natural language data. Common applications of NLP include language translation, sentiment analysis, chatbots, speech recognition, and text summarization. NLP works by breaking down sentences into structures that machines can work with, such as tokens, parts of speech, and semantic meaning. As the technology advances, NLP is playing a crucial role in bridging the gap between human communication and computer understanding, making interactions with machines more intuitive and intelligent."]

tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

word = "word"
if word in model.wv:
    print(f"Embedding for '{word}':\n{model.wv[word]}\n")
    
    print(f"Most similar words to '{word}':")
    for similar_word, similarity in model.wv.most_similar(word, topn=5):
        print(f"{similar_word}: {similarity:.4f}")
words = ["natural", "language", "processing"]
print("\nEmbeddings for selected words:")
for w in words:
    if w in model.wv:
        print(f"{w}: {model.wv[w]}")
