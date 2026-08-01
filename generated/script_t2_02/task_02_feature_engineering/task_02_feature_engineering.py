from dependency import *  # noqa: F401,F403


def feature_engineering_2(texts):
    print('Found %s texts.' % len(texts))

    # Tokenize the texts using gensim.

    tokens = list()
    for text in texts:
        tokens.append(simple_preprocess(text))

    # Vectorize the text samples into a 2D integer tensor.

    MAX_NUM_WORDS = 10000 # 2 words reserved: 0=pad, 1=oov
    return MAX_NUM_WORDS, tokens
