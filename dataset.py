
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision import transforms





class DiffSet(Dataset):
    def __init__(self, data_type, dataset="MNIST", root="./data"):
        """
        Initializes the dataset with the specified transforms and dataset choice.

        Args:
            train (bool): Whether to load the training set (True) or test set (False).
            dataset (str): The name of the dataset to load. Supported: MNIST, Fashion, CIFAR, CelebA.
            root (str): Path to the root directory for downloading datasets.
        """
        # Define transforms for each dataset
        if dataset in ["MNIST", "Fashion"]:
            transform = transforms.Compose([
                transforms.Pad(2),  # Resize to 32x32
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize grayscale
            ])
        elif dataset == "CIFAR":
            transform = transforms.Compose([
                transforms.ToTensor(),  # Already 32x32x3
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize RGB
            ])
        elif dataset == "CelebA":
            transform = transforms.Compose([
                transforms.Resize(64),  # Resize CelebA images to 64x64
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize RGB
            ])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Dataset mapping
        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
            "CelebA": CelebA,
        }

        # Load the specified dataset
        if dataset == "CelebA":
            # CelebA requires split instead of train argument
            self.train_dataset = datasets[dataset](root, split=data_type, download=True, transform=transform)
        else:
            self.train_dataset = datasets[dataset](root, train=data_type, download=True, transform=transform)

        # Store dataset length and metadata
        self.dataset_len = len(self.train_dataset)
        if dataset in ["MNIST", "Fashion"]:
            self.depth = 1  # Grayscale
            self.size = 32
        elif dataset == "CIFAR":
            self.depth = 3  # RGB
            self.size = 32
        elif dataset == "CelebA":
            self.depth = 3  # RGB
            self.size = 64

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        """
        Retrieves a single image from the dataset.

        Args:
            item (int): The index of the item to retrieve.

        Returns:
            img (Tensor): Transformed image tensor.
        """
        img, _ = self.train_dataset[item]
        return img