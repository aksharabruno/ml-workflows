from dependency import *  # noqa: F401,F403


def feature_engineering_2(labels, texts):
    # Tokenize the texts using gensim.

    tokens = list()
    for text in texts:
        tokens.append(simple_preprocess(text))

    # Vectorize the text samples into a 2D integer tensor.

    MAX_NUM_WORDS = 10000 # 2 words reserved: 0=pad, 1=oov
    MAX_SEQUENCE_LENGTH = 1000

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=0, no_above=1.0,
                               keep_n=MAX_NUM_WORDS-2)

    word_index = dictionary.token2id
    print('Found %s unique tokens.' % len(word_index))

    data = [dictionary.doc2idx(t) for t in tokens]

    # Truncate and pad sequences.

    data = [i[:MAX_SEQUENCE_LENGTH] for i in data]
    data = np.array([np.pad(i, (MAX_SEQUENCE_LENGTH-len(i), 0),
                            mode='constant', constant_values=-2)
                     for i in data], dtype=int)
    data = data + 2

    print('Shape of data tensor:', data.shape)
    print('Length of label vector:', len(labels))

    return MAX_NUM_WORDS, data, word_index
