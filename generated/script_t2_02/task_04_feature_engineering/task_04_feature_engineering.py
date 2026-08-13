from dependency import *  # noqa: F401,F403


def feature_engineering_4(MAX_NUM_WORDS, embeddings_index, word_index):
    # Prepare the embedding matrix:

    print('Preparing embedding matrix.')

    EMBEDDING_DIM = 100

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))
    n_not_found = 0
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS-2:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i+2] = embedding_vector
        else:
            n_not_found += 1

    embedding_matrix = torch.FloatTensor(embedding_matrix)
    print('Shape of embedding matrix:', embedding_matrix.shape)
    print('Words not found in pre-trained embeddings:', n_not_found)

    return embedding_matrix
