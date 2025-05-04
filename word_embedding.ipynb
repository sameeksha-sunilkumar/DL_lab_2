import gensim 
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('punkt')
nltk.download('punkt_tab')
corpus = ["Natural Language Processing (NLP) is a field of artificial intelligence (AI) that focuses on enabling computers to understand, interpret, and generate human language. By combining techniques from linguistics, computer science, and machine learning, NLP allows machines to process and analyze large amounts of natural language data. Common applications of NLP include language translation, sentiment analysis, chatbots, speech recognition, and text summarization. NLP works by breaking down sentences into structures that machines can work with, such as tokens, parts of speech, and semantic meaning. As the technology advances, NLP is playing a crucial role in bridging the gap between human communication and computer understanding, making interactions with machines more intuitive and intelligent."]
tokenized_corpus=[word_tokenize(sentence.lower()) for sentence in corpus]
model = Word2Vec(sentences=tokenized_corpus, vector_size=100,window=5,min_count=1,workers=4)
word="word"
if word in model.wv:
    embedding=model.wv[word]
    print(f"emb for {word} : {embedding}")
    similar_words = model.wv.most_similar("word",topn=5)
    print("most similar words to word")
    for similar_words, similarity in similar_words:
        print(f"{similar_words}: {similarity:.4f}")
words=["natural","language","processing"]
embeddings=np.array([model.wv[word] for word in words])
print("\n embedding for words: ")
for words, embedding in zip(words,embeddings):
    print(f"{word}:{embedding}")
