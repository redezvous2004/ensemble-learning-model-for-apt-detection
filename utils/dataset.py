import torch
from torch.utils.data import Dataset
import numpy as np

class IPFlowDataset(Dataset):
    def __init__(self, X, y, ip_pairs, max_flows=100):
        self.max_flows = max_flows
        self.sequences = []
        self.labels = []
        self.lengths = []

        # Convert to numpy for faster processing
        X_np = X if isinstance(X, np.ndarray) else X.values
        y_np = y.values if hasattr(y, 'values') else np.array(y)

        # Group by IP pairs
        unique_pairs = set(ip_pairs)
        print(f"Number of unique IP pairs: {len(unique_pairs)}")

        for pair in unique_pairs:
            indices = [i for i, p in enumerate(ip_pairs) if p == pair]

            if len(indices) > 0:
                flows = X_np[indices]
                flow_labels = y_np[indices]

                # Get APT flows
                apt_indices = [i for i, label in enumerate(flow_labels) if label == "APT"]

                selected_indices = []
                if len(apt_indices) > 0.15*max_flows:
                    # Make sure to get all APT flows as possible
                    if len(apt_indices) <= max_flows:
                        selected_indices.extend(apt_indices)

                        # Number of flows left
                        remaining_slots = max_flows - len(selected_indices)

                        # Get surrounding flows of an APT flow
                        surrounding_indices = set()
                        for apt_idx in apt_indices:
                            window = 10
                            start = max(0, apt_idx - window)
                            end = min(len(flows), apt_idx + window + 1)
                            surrounding_indices.update(range(start, end))

                        # Check if there remains slot in frame
                        surrounding_indices = surrounding_indices - set(apt_indices)

                        # Add more surrounding flows
                        if surrounding_indices:
                            n_surrounding = min(remaining_slots, len(surrounding_indices))
                            selected_indices.extend(
                                np.random.choice(list(surrounding_indices),
                                               size=n_surrounding,
                                               replace=False)
                            )

                        # If still not enough, add randomly
                        remaining_slots = max_flows - len(selected_indices)
                        if remaining_slots > 0:
                            remaining_indices = list(set(range(len(flows))) -
                                                   set(selected_indices))
                            if remaining_indices:
                                selected_indices.extend(
                                    np.random.choice(remaining_indices,
                                                   size=remaining_slots,
                                                   replace=False)
                                )
                    else:
                        # If there are to much APT flows, get randomly from APT flows
                        selected_indices = list(np.random.choice(apt_indices,
                                                              size=max_flows,
                                                              replace=False))
                else:
                    # If there is no APT flow, get randomly
                    if len(flows) > max_flows:
                        selected_indices = list(np.random.choice(len(flows),
                                                              size=max_flows,
                                                              replace=False))
                    else:
                        selected_indices = list(range(len(flows)))

                # Sort indices
                selected_indices = sorted(selected_indices)

                if len(selected_indices) > max_flows:
                    selected_indices = selected_indices[:max_flows]

                selected_flows = flows[selected_indices]
                length = len(selected_flows)

                # Padding
                if length < max_flows:
                    padding = np.zeros((max_flows - length, flows.shape[1]))
                    selected_flows = np.vstack([selected_flows, padding])
                elif length > max_flows:
                    selected_flows = selected_flows[:max_flows]
                    length = max_flows

                self.sequences.append(selected_flows)
                self.labels.append(1 if len(apt_indices) >= 0.1*max_flows else 0)
                self.lengths.append(length)

        print(f"Total sequences: {len(self.sequences)}")
        print(f"APT sequences: {sum(self.labels)}")

        # Convert to tensors
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.FloatTensor(self.labels)
        self.lengths = torch.LongTensor(self.lengths)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.labels[idx],
            self.lengths[idx]
        )