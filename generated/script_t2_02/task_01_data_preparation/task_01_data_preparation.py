from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # ## GloVe word embeddings
    #
    # Let's begin by loading a datafile containing pre-trained word
    # embeddings.  The datafile contains 100-dimensional embeddings for
    # 400,000 English words.

    datapath = os.getenv('DATADIR')
    if datapath is None:
        print("Please set DATADIR environment variable!")
        sys.exit(1)

    glove_dir = os.path.join(datapath, "glove.6B")

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))


    # ## 20 Newsgroups data set
    #
    # Next we'll load the [20 Newsgroups]
    # (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
    # data set.
    #
    # The dataset contains 20000 messages collected from 20 different
    # Usenet newsgroups (1000 messages from each group):
    #
    # | alt.atheism           | soc.religion.christian   | comp.windows.x     | sci.crypt
    # | talk.politics.guns    | comp.sys.ibm.pc.hardware | rec.autos          | sci.electronics
    # | talk.politics.mideast | comp.graphics            | rec.motorcycles    | sci.space
    # | talk.politics.misc    | comp.os.ms-windows.misc  | rec.sport.baseball | sci.med
    # | talk.religion.misc    | comp.sys.mac.hardware    | rec.sport.hockey   | misc.forsale

    text_data_dir = os.path.join(datapath, "20_newsgroup")

    print('Processing text dataset')

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            print('-', name, label_id)
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                    labels.append(label_id)

    return embeddings_index, labels, texts
