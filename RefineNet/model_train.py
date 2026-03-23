import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from RefineNet import RefineNet

if __name__ == '__main__':
    transform_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda image: torch.from_numpy(np.array(image, dtype=np.int64))),
    ])

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir, mask_dir, transform_image=None, transform_mask=None):
            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.transform_image = transform_image
            self.transform_mask = transform_mask
            self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                                os.path.isfile(os.path.join(image_dir, f)) and f.endswith(".jpg")]
            self.mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if
                               os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(".png")]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image_path = self.image_files[idx]
            mask_path = self.mask_files[idx]
            image = Image.open(image_path).convert('RGB')
            mask = Image.open(mask_path)

            if self.transform_image:
                image = self.transform_image(image)
            if self.transform_mask:
                mask = self.transform_mask(mask)

            # Convert the mask to a category mask
            mask = mask.squeeze().long()  # Squeeze the mask and convert to long type

            return image, mask

    train_image_paths = "./datasets/train_image"
    train_mask_paths = "./datasets/train_mask"
    val_image_paths = "./datasets/val_image"
    val_mask_paths = "./datasets/val_mask"
    train_dataset = MyDataset(train_image_paths, train_mask_paths, transform_image=transform_image, transform_mask=transform_mask)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = MyDataset(val_image_paths, val_mask_paths, transform_image=transform_image, transform_mask=transform_mask)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = RefineNet(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def train(model, optimizer, criterion, train_loader, val_loader, num_epochs):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        best_val_acc = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            # Train the model on the training set
            train_loss = 0
            train_acc = 0
            for i, (images, labels) in enumerate(train_loader):
                # Convert the labels to a tensor
                labels = labels.squeeze(1).long()

                # Forward pass
                outputs = model(images)
                # print("Label shape:", labels.shape)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate the accuracy
                predicted_mask = torch.argmax(outputs, dim=1)
                correct_pixels = torch.sum(predicted_mask == labels)
                total_pixels = torch.prod(torch.tensor(labels.shape))
                acc = correct_pixels / total_pixels

                # Update the training loss and accuracy
                train_loss += loss.item()
                train_acc += acc.item()

            # Calculate the average training loss and accuracy
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Evaluate the model on the validation set
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    # Convert the labels to a tensor
                    labels = labels.squeeze(1).long()

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Calculate the accuracy
                    predicted_mask = torch.argmax(outputs, dim=1)
                    correct_pixels = torch.sum(predicted_mask == labels)
                    total_pixels = torch.prod(torch.tensor(labels.shape))
                    acc = correct_pixels / total_pixels

                    # Update the validation loss and accuracy
                    val_loss += loss.item()
                    val_acc += acc.item()

            # Calculate the average validation loss and accuracy
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Check if current validation accuracy is better than the best so far
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

                # Save model parameters
                torch.save(model.state_dict(), "./model_save/best_parameters.pth")

            # Print the training and validation loss and accuracy for this epoch
            print(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Plot training and validation set loss and accuracy
        epochs = range(1, num_epochs + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Val Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(epochs, train_accs, label="Train Acc")
        plt.plot(epochs, val_accs, label="Val Acc")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    train(model, optimizer, criterion, train_loader, val_loader, num_epochs=10)