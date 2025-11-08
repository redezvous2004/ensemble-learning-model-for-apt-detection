import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
@torch.cuda.amp.autocast()
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_flows, batch_labels, batch_lengths in train_loader:
        batch_flows = batch_flows.to(device)
        batch_labels = batch_labels.to(device)
        batch_lengths = batch_lengths.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(batch_flows, batch_lengths)
            outputs = outputs.view(-1)
            loss = criterion(outputs, batch_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = torch.sigmoid(outputs.detach())
        preds = (preds > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch_flows, batch_labels, batch_lengths in test_loader:
        batch_flows = batch_flows.to(device)
        batch_lengths = batch_lengths.to(device)

        outputs = model(batch_flows, batch_lengths)
        outputs = outputs.view(-1)

        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }

def train_model(model, train_loader, test_loader, device, epochs=30):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01
    )

    scaler = torch.cuda.amp.GradScaler()
    best_f1 = 0
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)

        # Early stopping
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_state)
            break

        # Print metrics
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Train - Loss: {train_metrics["loss"]:.4f}, Acc: {train_metrics["accuracy"]:.4f}, '
              f'Prec: {train_metrics["precision"]:.4f}, Rec: {train_metrics["recall"]:.4f}, '
              f'F1: {train_metrics["f1"]:.4f}')
        print(f'Test  - Acc: {test_metrics["accuracy"]:.4f}, Prec: {test_metrics["precision"]:.4f}, '
              f'Rec: {test_metrics["recall"]:.4f}, F1: {test_metrics["f1"]:.4f}')

    # Final evaluation
    model.eval()
    final_metrics = evaluate(model, test_loader)
    print("\nFinal Test Results:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")

    return model, final_metrics