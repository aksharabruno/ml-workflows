from dependency import *  # noqa: F401,F403


def model_evaluation_6(criterion, model, test_loader):
    # Inference
    ret = test(test_loader, model, criterion)
    print(f"\nTesting: accuracy: {ret['accuracy']:.2%}")


