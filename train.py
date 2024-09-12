import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from args import *  # Make sure to define or import `SGR`, `save_checkpoint`, etc.
from utils import save_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    input_dim, train_loader, val_loader = get_loaders()

    model = SGR(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    num_epochs = 100  # Number of epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_preds = 0.0, 0
        for batch in train_loader:
            X_batch, y_batch = batch['features'], batch['label']
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = criterion(outputs.view(-1), y_batch.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.round(torch.sigmoid(outputs))  # Apply sigmoid for logits
            correct_preds += (preds.view_as(y_batch) == y_batch).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_preds / len(train_loader.dataset)

        # Validation phase
        model.eval()
        running_val_loss, val_correct_preds = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch = batch['features'], batch['label']
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs.view(-1), y_batch.float())
                running_val_loss += loss.item()
                preds = torch.round(torch.sigmoid(outputs))  # Apply sigmoid for logits
                val_correct_preds += (preds.view_as(y_batch) == y_batch).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = val_correct_preds / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        save_checkpoint(epoch, model, optimizer, train_losses, val_losses, train_accuracies, val_accuracies)

        # Debugging: Print a sample of predictions and labels
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                X_sample, y_sample = sample_batch['features'].to(device), sample_batch['label'].to(device)
                outputs = model(X_sample)
                preds = torch.round(torch.sigmoid(outputs))
                print(f"Sample Predictions: {preds.cpu().numpy()}")
                print(f"Sample Labels: {y_sample.cpu().numpy()}")


def training_plots():
    save_path = os.path.join('media', 'training_plot.png')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Plotting Accuracy
    plt.subplot(1, 2, 2) 
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.show()

if __name__ == "__main__":
    train()
    training_plots()
