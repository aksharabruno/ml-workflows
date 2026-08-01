from dependency import *  # noqa: F401,F403


def data_preparation_1():
    # Load the dataset
    file_path = "MultiClass_CyberBully_Dataset.xlsx"
    df = pd.read_excel(file_path)

    # Text cleaning function
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)  # remove URLs
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # remove special characters
        text = text.lower()  # convert to lowercase
        text = ' '.join([word for word in text.split() if word not in stop_words])  # remove stopwords
        return text

    print("Cleaning text data...")
    df['tweet_text'] = df['tweet_text'].apply(clean_text)

    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['cyberbullying_type'])

    # Use a smaller subset for hyperparameter tuning
    df_subset = df.sample(n=2000, random_state=42)

    # Split the dataset for tuning
    X_subset = df_subset['tweet_text']
    y_subset = df_subset['label']
    X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

    return X_test_subset, X_train_subset, df, label_encoder, y_train_subset
