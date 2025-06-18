import os
import subprocess
from env import directory, ffmpeg_path

data_dir = directory

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            wav_path = os.path.join(root, file)
            mp3_path = wav_path[:-4] + ".mp3"
            if os.path.exists(mp3_path): #exists already
                print("SKIPPING")
                continue
            print("CONVERTING")
            subprocess.run([ 
                ffmpeg_path,
                "-i", wav_path,
                "-ab", "192k",
                "-y",
                mp3_path
            ])
            # remove the .wav file
            print("DELETING")
            os.remove(wav_path)
