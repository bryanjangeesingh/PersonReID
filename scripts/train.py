from torch.utils.data import Dataset, DataLoader
from create_pairs_new import create_pairs
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class SuperFeatureDataset(Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        assert len(positive_pairs) == len(
            negative_pairs
        ), "Positive and negative pairs must be the same length"
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        pos_pair = self.positive_pairs[idx]
        neg_pair = self.negative_pairs[idx]
        return pos_pair, neg_pair


# load the features pkl
features = pickle.load(
    open(
        "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/super_feature_dataset/features_dict.pkl",
        "rb",
    )
)

train_super_features_dir = "/Users/bryanjangeesingh/Documents/PersonReID/Datasets/Market-Pytorch/Market/super_feature_dataset/"

positive_pairs, negative_pairs = create_pairs(features, train_super_features_dir)
positive_pairs = positive_pairs[: len(negative_pairs)]

dataset = SuperFeatureDataset(positive_pairs, negative_pairs)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Output embeddings, not class logits
        )

    def forward(self, x):
        embeddings = self.network(x)
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings


class CLIPLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CLIPLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_pair, neg_pair):
        # Cosine similarity
        pos_similarity = (pos_pair[0] * pos_pair[1]).sum(dim=1)
        neg_similarity = (neg_pair[0] * neg_pair[1]).sum(dim=1)

        # Contrastive loss: maximize pos_similarity and minimize neg_similarity
        loss = torch.relu(-pos_similarity + neg_similarity + self.margin).mean()
        return loss


# Model, Loss, and Optimizer
model = SimpleClassifier().to(device)
loss_function = CLIPLoss(margin=0.5).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, dataloader, loss_function, optimizer, epochs=10, device=device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            # Send data to the device (GPU or CPU)
            pos_pair = (data[0][0].to(device).float(), data[0][1].to(device).float())
            neg_pair = (data[1][0].to(device).float(), data[1][1].to(device).float())

            optimizer.zero_grad()

            # Compute embeddings
            pos_embeddings_1 = model(pos_pair[0])
            pos_embeddings_2 = model(pos_pair[1])
            neg_embeddings_1 = model(neg_pair[0])
            neg_embeddings_2 = model(neg_pair[1])

            # Calculate loss
            loss = loss_function(
                (pos_embeddings_1, pos_embeddings_2),
                (neg_embeddings_1, neg_embeddings_2),
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss}")


# Running the training
train_model(model, dataloader, loss_function, optimizer, device=device)
