from dependency import *  # noqa: F401,F403


def data_preparation_3(data, labels):
    # Split the data into a training set and a validation set

    VALIDATION_SET, TEST_SET = 1000, 4000

    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=TEST_SET,
                                                        shuffle=True,
                                                        random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      test_size=VALIDATION_SET,
                                                      shuffle=False)

    print('Shape of training data tensor:', x_train.shape)
    print('Length of training label vector:', len(y_train))
    print('Shape of validation data tensor:', x_val.shape)
    print('Length of validation label vector:', len(y_val))
    print('Shape of test data tensor:', x_test.shape)
    print('Length of test label vector:', len(y_test))

    # Create PyTorch DataLoaders for all data sets:

    BATCH_SIZE = 128

    print('Train: ', end="")
    train_dataset = TensorDataset(torch.LongTensor(x_train),
                                  torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4)
    print(len(train_dataset), 'messages')

    print('Validation: ', end="")
    validation_dataset = TensorDataset(torch.LongTensor(x_val),
                                       torch.LongTensor(y_val))
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=4)
    print(len(validation_dataset), 'messages')

    print('Test: ', end="")
    test_dataset = TensorDataset(torch.LongTensor(x_test),
                                 torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4)
    print(len(test_dataset), 'messages')

    return test_loader, train_loader, validation_loader
