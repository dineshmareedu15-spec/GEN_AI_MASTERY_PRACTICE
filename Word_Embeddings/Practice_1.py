from gensim.models import Word2Vec

sentences = [["python", "is", "a", "language"],
             ["java", "is", "a", "language"],
             ["python", "is", "easy"]]

m1 = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)
m2 = Word2Vec(sentences, vector_size=10, window=2, min_count=2, workers=1)

print("min_count=1:", list(m1.wv.index_to_key))
print("min_count=2:", list(m2.wv.index_to_key))
