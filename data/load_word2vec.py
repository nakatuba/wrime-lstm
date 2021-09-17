from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "./entity_vector/entity_vector.model.bin", binary=True
)
model.save_word2vec_format("./japanese_word2vec_vectors.vec")
