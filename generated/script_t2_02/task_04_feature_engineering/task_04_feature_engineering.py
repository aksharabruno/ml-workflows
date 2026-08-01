from dependency import *  # noqa: F401,F403


def feature_engineering_4(MAX_NUM_WORDS, tokens):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=0, no_above=1.0,
                               keep_n=MAX_NUM_WORDS-2)

    word_index = dictionary.token2id
    print('Found %s unique tokens.' % len(word_index))

    data = [dictionary.doc2idx(t) for t in tokens]

    return word_index
