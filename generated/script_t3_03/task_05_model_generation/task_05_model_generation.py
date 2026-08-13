from dependency import *  # noqa: F401,F403


def model_generation_5(n_class):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),

                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(4),

                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Dropout2d(0.3)
            )

            # ⭐ FIX 1: flat_size must be computed before using it
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                out = self.features(dummy)
                flat_size = out.numel()

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(),
                nn.Linear(flat_size, 500),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(500, n_class)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    modelnew = CNN()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modelnew.parameters(), lr = 0.001, weight_decay=0.0001)

    return loss_fn, modelnew, optimizer
