import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, feature_dim), nn.Tanh()
        )

    def forward(self, noise): return self.model(noise)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, features): return self.model(features)

def train_gan(generator, discriminator, train_dataset, device, epochs=100, batch_size=32):
    # Hyperparameters
    latent_dim = 100
    lr = 1e-4
    b1 = 0.5
    b2 = 0.999
    # Optimizer
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    # Loss
    adversarial_loss = nn.BCELoss()

    apt_sequences = []
    apt_lengths = []
    for i in range(len(train_dataset)):
        if train_dataset.labels[i] == 1:
            apt_sequences.append(train_dataset.sequences[i])
            apt_lengths.append(train_dataset.lengths[i])
    apt_sequences = torch.stack(apt_sequences)
    apt_lengths = torch.stack(apt_lengths)

    n_apt = len(apt_sequences)
    print(f"Number of real APT sequences: {n_apt}")

    # Training
    for epoch in range(epochs):
        for i in range(0, len(apt_sequences), batch_size):
            batch_size_i = min(batch_size, n_apt - i)

            # Input
            real_seqs = apt_sequences[i:i+batch_size_i].to(device)

            # Create labels
            valid = torch.ones((batch_size_i, 1)).to(device)
            fake = torch.zeros((batch_size_i, 1)).to(device)

            # Generate a batch of sequences
            noise = torch.randn(batch_size_i, latent_dim).to(device)

            # Generate fake sequences
            gen_seqs = generator(noise)

            # Train Discriminator
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_seqs.view(batch_size_i, -1)), valid)
            fake_loss = adversarial_loss(discriminator(gen_seqs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            g_loss = adversarial_loss(discriminator(gen_seqs), valid)

            g_loss.backward()
            optimizer_G.step()

        if (epoch+1) % 10 == 0:
            print(f"[{epoch+1}/{epochs}] D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    return generator