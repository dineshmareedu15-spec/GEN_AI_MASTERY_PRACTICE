import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

sentences = [["python", "is", "a", "language"], ["java", "is", "a", "language"],
             ["python", "is", "easy"], ["java", "is", "easy"],
             ["python", "is", "powerful"], ["java", "is", "powerful"]]

model = Word2Vec(sentences, vector_size=10, window=2, min_count=1, workers=1, seed=1)

words = list(model.wv.index_to_key)
points = PCA(n_components=2).fit_transform(model.wv[words])

plt.scatter(points[:, 0], points[:, 1])
for w, (x, y) in zip(words, points):
    plt.annotate(w, (x, y))
plt.show()