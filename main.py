from utils.preprocess import preprocessing
from utils.dataset import IPFlowDataset
from models.gan import Generator, Discriminator, train_gan
from models.ensemble_model import ELModel
from train import train_model
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "data/flowFeatures.csv"

    df = pd.read_csv(path, usecols=lambda x: x not in ['publicIP', 'FlowID', 'SrcPort', 'DstPort','Timestamp','Protocol', 'FlowDuration'])
    X_train_scaled, X_test_scaled, y_train, y_test, pairs_train, pairs_test = preprocessing(df)
    train_dataset = IPFlowDataset(X_train_scaled, y_train, pairs_train)
    test_dataset = IPFlowDataset(X_test_scaled, y_test, pairs_test)

    latent_dim = 100
    feature_dim = train_dataset.sequences.size(-1) * train_dataset.sequences.size(1)  # Total features in a sequence
    generator = Generator(latent_dim, feature_dim).to(device)
    discriminator = Discriminator(feature_dim).to(device)

    # Train GAN
    generator = train_gan(generator, discriminator, train_dataset, device)

    # Generate synthetic APT sequences
    n_synthetic = int(len(train_dataset.sequences) - 2 * sum(train_dataset.labels).item())  # Convert to integer
    print(f"Generating {n_synthetic} synthetic sequences")

    noise = torch.randn(n_synthetic, latent_dim).to(device)
    # labels = torch.ones((n_synthetic, 1)).to(device) #CGAN

    with torch.no_grad():
        # gen_seqs = generator(noise, labels) #CGAN
        gen_seqs = generator(noise) #GAN
        gen_seqs = gen_seqs.view(n_synthetic, train_dataset.sequences.size(1), -1)

    # Add synthetic sequences to training data
    train_dataset.sequences = torch.cat([train_dataset.sequences, gen_seqs.cpu()], dim=0)
    train_dataset.labels = torch.cat([train_dataset.labels, torch.ones(n_synthetic)], dim=0)
    train_dataset.lengths = torch.cat([train_dataset.lengths, torch.full((n_synthetic,), train_dataset.sequences.size(1), dtype=torch.long)], dim=0)

    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    input_size = X_train_scaled.shape[1]
    model = ELModel(input_size).to(device)

    train_model(model, train_loader, test_loader, device)

    # Get model's predict
    def plot_confusion_matrix(model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_flows, batch_labels, batch_lengths in test_loader:
                batch_flows = batch_flows.to(device)
                batch_lengths = batch_lengths.to(device)

                outputs = model(batch_flows, batch_lengths)
                outputs = outputs.view(-1)

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.show()

    plot_confusion_matrix(model, test_loader)