from dependency import *  # noqa: F401,F403


def model_evaluation_8(criterion, end_time, model, start_time, test_loader):
    print('Total training time: {}.'.format(end_time - start_time))

    # Inference
    ret = test(test_loader, model, criterion)
