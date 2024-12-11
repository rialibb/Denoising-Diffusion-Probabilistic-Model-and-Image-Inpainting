
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms




class DiffSet(Dataset):
    def __init__(self, train, dataset="MNIST", root="./data"):
        # Define transforms with padding for MNIST/Fashion and normalization for all datasets
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
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Dataset mapping
        datasets = {
            "MNIST": MNIST,
            "Fashion": FashionMNIST,
            "CIFAR": CIFAR10,
        }

        # Load dataset with transform applied
        self.train_dataset = datasets[dataset](root, download=True, train=train, transform=transform)

        # Store dataset length and metadata
        self.dataset_len = len(self.train_dataset)
        self.depth = 1 if dataset in ["MNIST", "Fashion"] else 3
        self.size = 32

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        # Return transformed data directly from the dataset
        img, _ = self.train_dataset[item]
        return img
