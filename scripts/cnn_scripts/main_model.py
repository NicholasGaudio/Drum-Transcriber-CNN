import os
import random
import torch
from torch.utils.data import DataLoader
from scripts.cnn_scripts.dataset import DrumDataset
from scripts.cnn_scripts.cnn import DrumCNN
from scripts.env import directory

# ------------------------
# 1: Collect audio paths and labels
# ------------------------
random.seed(42)
labels = ["kick", "snare", "closed-hat"]
train_paths, train_labels = [], []
test_paths, test_labels = [], []

for label_index, label in enumerate(labels):
    label_dir = os.path.join(directory, label)
    all_files = [f for f in os.listdir(label_dir) if f.endswith(".mp3")]

    # Separate originals vs augmented
    original_files = [f for f in all_files if "AUGMENTED" not in f]
    augmented_files = [f for f in all_files if "AUGMENTED" in f]

    # Take a few originals for test
    num_test = min(10, len(original_files))
    test_files = random.sample(original_files, num_test)
    test_paths.extend([os.path.join(label_dir, f) for f in test_files])
    test_labels.extend([label_index] * num_test)

    # Remaining originals + all augmented for training
    train_files = [f for f in original_files if f not in test_files] + augmented_files
    train_paths.extend([os.path.join(label_dir, f) for f in train_files])
    train_labels.extend([label_index] * len(train_files))

# ------------------------
# 2: Create datasets & dataloaders
# ------------------------
train_dataset = DrumDataset(train_paths, train_labels)
test_dataset = DrumDataset(test_paths, test_labels)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset info
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ------------------------
# 3: Create model, loss, optimizer
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = DrumCNN().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# 4: Training loop
# ------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for mel_batch, label_batch in train_loader:
        mel_batch, label_batch = mel_batch.to(device), label_batch.to(device)

        optimizer.zero_grad()
        outputs = model(mel_batch)
        batch_loss = loss_fn(outputs, label_batch)
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * mel_batch.size(0)

    epoch_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# ------------------------
# 5: Test loop
# ------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for mel_batch, label_batch in test_loader:
        mel_batch, label_batch = mel_batch.to(device), label_batch.to(device)
        outputs = model(mel_batch)
        _, predicted = torch.max(outputs, 1)
        total += label_batch.size(0)
        correct += (predicted == label_batch).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save model
torch.save(model.state_dict(), "drum_cnn_model.pth")