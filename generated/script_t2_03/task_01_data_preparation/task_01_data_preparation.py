from dependency import *  # noqa: F401,F403


def data_preparation_1():
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)

    rcParams['figure.figsize'] = 14, 8

    RANDOM_SEED = 42
    LABELS = ["Normal", "Fraud"]

    df = pd.read_csv("data/creditcard.csv")
    count_classes = pd.value_counts(df['Class'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    plt.title("Transaction class distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")

    frauds = df[df.Class == 1]
    normal = df[df.Class == 0]

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.suptitle('Amount per transaction by class')

    bins = 50

    ax1.hist(frauds.Amount, bins = bins)
    ax1.set_title('Fraud')

    ax2.hist(normal.Amount, bins = bins)
    ax2.set_title('Normal')

    plt.xlabel('Amount ($)')
    plt.ylabel('Number of Transactions')
    plt.xlim((0, 20000))
    plt.yscale('log')
    plt.show();

    # prepare data
    data = df.drop(['Time'], axis=1)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)

    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test = X_test.values

    return X_test, X_train, y_test
