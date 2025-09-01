import torch
from scripts.cnn_scripts.cnn import DrumCNN
from scripts.cnn_scripts.dataset import audio_to_melspectrogram  # your function


sample_path = r"C:\Users\ngaud\Downloads\low-acoustic-snare-sound-a-key-07-X09.mp3" # replace with your file

model = DrumCNN()
model.load_state_dict(torch.load("drum_cnn_model.pth", map_location="cpu"))
model.eval() 

mel = audio_to_melspectrogram(sample_path) 
mel = mel.unsqueeze(0) 

with torch.no_grad():
    outputs = model(mel)
    _, predicted_class = torch.max(outputs, 1)

labels = ["kick", "snare", "closed-hat"]
print(f"Predicted class: {labels[predicted_class.item()]}")
