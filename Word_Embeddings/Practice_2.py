from gensim.models import Word2Vec

sentences = [["python", "is", "easy"], ["java", "is", "easy"]]
model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1)

model.save("my_w2v.model")

loaded = Word2Vec.load("my_w2v.model")
print(loaded.wv.most_similar('python', topn=2))
