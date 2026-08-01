from dependency import *  # noqa: F401,F403


def data_preparation_4(classes, dataset_path, datasetz, n_class):
    print(n_class)

    count = Counter([label for _,label in tqdm(datasetz,desc="counting", leave = False)])
    print(count)

    class_distribution = {classes[k]:v for k,v in count.items()}
    print(class_distribution)

    class_distribution = {classes[k]:v for k,v in count.items()}
    print(class_distribution)

    df_class_distribution = pd.Series(class_distribution)

    # Make a bar chart from the function output
    df_class_distribution.plot(kind = "bar")

    # Add axis labels and title
    plt.xlabel("Class Label")
    plt.ylabel("Frequency [count]")
    plt.title("Distribution of Classes in Training Dataset");

    batch_size = 64
    dataset_loader = DataLoader(datasetz, batch_size = batch_size, shuffle= True)

    first_batch = next(iter(dataset_loader))
    print(f"Shape of one batch: {first_batch[0].shape}")

    def get_mean_std(loader):

        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in tqdm(loader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

            mean = channels_sum/num_batches

            std = (channels_squared_sum/num_batches - mean**2)**0.5
            return mean, std

    mean, std = get_mean_std(dataset_loader)
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std}")

    height = 224
    width = 224

    transformNorm = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std= std)
        ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),

        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),

        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    dataset = datasets.ImageFolder(root = dataset_path, transform = train_transform)

    g = torch.Generator()
    g.manual_seed(42)

    train_dataset, val_dataset = random_split(dataset, [0.85, 0.15] , generator=g)

    test_dataset_path = r"C:\Users\Jelil\Desktop\New folder (7)\seg_test\seg_test"
    test_dataset = datasets.ImageFolder(root = test_dataset_path, transform = transformNorm )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    TestLoader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    image , _ = next(iter(train_loader))
    return TestLoader, train_loader, transformNorm, val_loader
