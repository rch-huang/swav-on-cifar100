import argparse
import random
import numpy as np

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.cluster import KMeans


# ---------------------------------------
# 0. Reproducibility: set global seed
# ---------------------------------------
def set_seed(seed: int = 0):
    print(f"Setting seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------
# 1. Load SwAV model
# ---------------------------------------
def load_model():
    print("Loading SwAV model from torch.hub ...")
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    model.eval()
    model.cuda()
    print("Model loaded.")
    return model


# ---------------------------------------
# 2. Extract embedding
# ---------------------------------------
def get_embedding(model, x):
    with torch.no_grad():
        x = x.cuda()
        feats = model(x)
        return feats.cpu()


# ---------------------------------------
# 3. Extract all features and labels
# ---------------------------------------
def extract_features(model, dataset, num_workers=4):
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    features, labels = [], []

    print(f"Extracting features for {len(dataset)} samples...")
    for x, y in tqdm(loader):
        feats = get_embedding(model, x)   # (B, D)
        features.append(feats)
        labels.append(y)

    return torch.cat(features), torch.cat(labels)


# ---------------------------------------
# 4. kNN classifier
# ---------------------------------------
def knn_predict(train_feat, train_lbl, test_feat, test_lbl, k=50):
    print("\nRunning k-NN evaluation...")
    # L2 normalize for cosine similarity
    train_feat = torch.nn.functional.normalize(train_feat, dim=1)
    test_feat = torch.nn.functional.normalize(test_feat, dim=1)

    # (Nt, Ntr)
    sim = torch.mm(test_feat, train_feat.T)
    topk = sim.topk(k, dim=1).indices  # (Nt, k)

    correct = 0
    for i in tqdm(range(test_feat.size(0))):
        votes = train_lbl[topk[i]]
        pred = torch.mode(votes)[0]
        correct += (pred.item() == test_lbl[i].item())

    acc = correct / test_feat.size(0)
    return acc


# ---------------------------------------
# 5. K-means clustering on TEST features
# ---------------------------------------
def kmeans_test(test_feat, test_lbl, num_classes, seed=0):
    print(f"\nRunning K-means on test set ({num_classes} clusters)...")

    X = np.array(test_feat)  # (Nt, D)
    Y = np.array(test_lbl)   # (Nt,)

    kmeans = KMeans(n_clusters=num_classes, n_init=20, random_state=seed)
    pred_clusters = kmeans.fit_predict(X)

    # Map cluster id -> majority label
    mapping = {}
    for c in range(num_classes):
        idxs = np.where(pred_clusters == c)[0]
        if len(idxs) == 0:
            # empty cluster: random label (still deterministic using seed above)
            mapping[c] = np.random.randint(0, num_classes)
        else:
            mapping[c] = np.bincount(Y[idxs]).argmax()

    pred_final = np.array([mapping[p] for p in pred_clusters])
    acc = (pred_final == Y).mean()
    return acc


# ---------------------------------------
# 6. Linear classifier on top of frozen features
# ---------------------------------------
class LinearClassifier(torch.nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def eval_linear(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train_linear_classifier(train_feat,
                            train_lbl,
                            test_feat,
                            test_lbl,
                            num_classes,
                            epochs=50,
                            batch_size=256,
                            lr=0.1,
                            weight_decay=1e-4,
                            seed=0):

    print("\nTraining linear classifier on frozen SwAV features...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_dim = train_feat.size(1)

    # build dataset
    train_ds = TensorDataset(train_feat, train_lbl)
    test_ds = TensorDataset(test_feat, test_lbl)

    # generator for deterministic shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False)

    model = LinearClassifier(in_dim, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        total_loss /= len(train_ds)

        # 可以只在若干 epoch 打印
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            test_acc = eval_linear(model, test_loader, device)
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"loss = {total_loss:.4f}, "
                f"test acc = {test_acc*100:.2f}%"
            )

    final_acc = eval_linear(model, test_loader, device)
    return final_acc


# ---------------------------------------
# 7. Load chosen dataset
# ---------------------------------------
def load_dataset(name):

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])

    if name == 'cifar10':
        print("Loading CIFAR-10...")
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transform, download=True)
        num_classes = 10

    elif name == 'cifar100':
        print("Loading CIFAR-100...")
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, transform=transform, download=True)
        num_classes = 100

    else:
        raise ValueError("Dataset must be cifar10 or cifar100")

    return trainset, testset, num_classes


# ---------------------------------------
# Entry
# ---------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--k', type=int, default=50,
                        help='k for k-NN')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=50,
                        help='epochs for linear classifier')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for linear classifier')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate for linear classifier')

    args = parser.parse_args()

    # 0) seed
    set_seed(args.seed)

    # 1) Load backbone
    model = load_model()

    # 2) Load dataset
    trainset, testset, num_classes = load_dataset(args.dataset)

    # 3) Compute embeddings (frozen)
    train_feat, train_lbl = extract_features(model, trainset)
    test_feat, test_lbl = extract_features(model, testset)

    # 4) k-NN eval
    # acc_knn = knn_predict(train_feat, train_lbl, test_feat, test_lbl, k=args.k)
    # print(f"\n{args.k}-NN accuracy on {args.dataset}: {acc_knn*100:.2f}%")

    # # 5) K-means clustering eval
    # acc_kmeans = kmeans_test(test_feat, test_lbl, num_classes, seed=args.seed)
    # print(f"K-means accuracy on {args.dataset}: {acc_kmeans*100:.2f}%")

    # 6) Linear classifier eval
    acc_linear = train_linear_classifier(
        train_feat, train_lbl,
        test_feat, test_lbl,
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
    print(f"\nLinear classifier accuracy on {args.dataset}: {acc_linear*100:.2f}%")
