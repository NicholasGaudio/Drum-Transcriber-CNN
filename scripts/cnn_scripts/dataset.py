import torch
from torch.utils.data import Dataset
from scripts.melspectrograms.data_prep import audio_to_melspectrogram

class DrumDataset(Dataset):
    def __init__(self, audio_paths, labels, transform=None):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform
    
    # Returns the number of items in data set
    def __len__(self):
        return len(self.audio_paths)
    
    # Will retrieve and make melspectrogram for an individual sample
    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]

        mel = audio_to_melspectrogram(path)
        label = torch.tensor(label, dtype=torch.long)

        return mel, label