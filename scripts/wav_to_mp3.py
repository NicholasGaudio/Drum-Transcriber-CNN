import os
import subprocess

data_dir = r"insert path"

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            wav_path = os.path.join(root, file)
            mp3_path = wav_path[:-4] + ".mp3"
            if os.path.exists(mp3_path): #exists already
                continue
            subprocess.run([ 
                r"insert path",
                "-i", wav_path,
                "-ab", "192k",
                "-y",
                mp3_path
            ])
            # remove the .wav file
            os.remove(wav_path)
