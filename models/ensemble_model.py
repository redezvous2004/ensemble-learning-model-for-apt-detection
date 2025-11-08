import torch
import torch.nn as nn
from models.switch_transformer import SwitchTransformer;
class ELModel(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ELModel, self).__init__()

        # Feature Extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )

        # Transformer Encoder
        encoder_layer = SwitchTransformer(
            d_model=hidden_size,
            nhead=8,
            num_experts=5,
            hidden_dim=hidden_size,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, lengths):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Feature extraction
        x = self.feature_extractor(x)  # (batch, seq_len, hidden_size)

        # Sort sequences by length for packed sequence
        lengths_cpu = lengths.cpu()
        lengths_sorted, indices = torch.sort(lengths_cpu, descending=True)
        x_sorted = x[indices]

        # Pack sequence for LSTM
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x_sorted, lengths_sorted, batch_first=True
        )

        # BiLSTM
        lstm_out, _ = self.lstm(packed_x)

        # Unpack LSTM output
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )

        # Restore original order
        _, restore_indices = torch.sort(indices)
        lstm_out = lstm_out[restore_indices]

        # Create attention mask for transformer
        # False indicates valid position, True indicates padding
        key_padding_mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(x.device)
        key_padding_mask = key_padding_mask >= lengths.unsqueeze(-1)

        # Transformer encoding
        transformer_out = self.transformer(
            lstm_out,
            src_key_padding_mask=key_padding_mask
        )  # (batch, seq_len, hidden_size)

        # Mask out padding before pooling
        mask = (~key_padding_mask).float().unsqueeze(-1)
        transformer_out = transformer_out * mask

        # Global average pooling over sequence length
        # First transpose to get sequence length at the last dimension
        pooled = transformer_out.sum(dim=1) / lengths.float().unsqueeze(-1)

        # Classification
        return self.classifier(pooled)