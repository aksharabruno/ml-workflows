from dependency import *  # noqa: F401,F403


def model_generation_5(embedding_matrix, train_loader, validation_loader):
    model = Net(embedding_matrix)
    model = model.to(device)

    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion = nn.CrossEntropyLoss()

    print(model)

    num_epochs = 40

    # Training loop
    start_time = datetime.now()
    for epoch in range(num_epochs):
        train_ret = train(train_loader, model, criterion, optimizer)
        log_measures(train_ret, log, "train", epoch)

        val_ret = test(validation_loader, model, criterion)
        log_measures(val_ret, log, "val", epoch)
        print(f"Epoch {epoch+1}: "
              f"train loss: {train_ret['loss']:.6f} "
              f"train accuracy: {train_ret['accuracy']:.2%}, "
              f"val accuracy: {val_ret['accuracy']:.2%}")

    end_time = datetime.now()
    print('Total training time: {}.'.format(end_time - start_time))

    return criterion, model
