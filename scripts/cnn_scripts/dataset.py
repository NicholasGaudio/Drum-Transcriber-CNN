import torch
from torch.utils.data import Dataset
import os
from melspectrograms.data_prep import audio_to_melspectrogram

class DrumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_paths = []
        self.labels = []
        self.transform = transform
        self.label_map = {
            'kick': 0,
            'snare': 1,
            'closed-hihat': 2,
        }
        
        # fill arrays with file paths and their labels
        for label in self.label_map:
            folder = os.path.join(data_dir, label)
            for file in os.listdir(folder):
                self.data_paths.append(os.path.join(folder, file))
                self.labels.append(self.label_map[label])

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        label = self.labels[idx]
    
        audio = audio_to_melspectrogram(file_path)
        
        # numpy array -> torch tensor 
        return torch.tensor(audio, dtype=torch.float32), label
