"""
FOR A BETTER IMPLEMENTATION CHECKOUT
https://github.com/kushvinth/shazam/

It takes you through a clean path of how the ingession process and pulling of the data from the database is being done
Thanks For Reading :)
"""


import librosa
import numpy as np
import requests
import yt_dlp
import ffmpeg
import os
import shutil

URL = "https://itunes.apple.com/search"


def download_mp3(NAME: str):
    if NAME:
        params = {
            "term": NAME,
            "media": "music",
            "entity": "song",
            "limit": 1,
            # "js_runtimes": "/usr/local/bin/node" ## Installed Demo with Brew
        }
        response = requests.get(URL, params=params)
        data = response.json()

        # https://stackoverflow.com/questions/73516823/using-yt-dlp-in-a-python-script-how-do-i-download-a-specific-section-of-a-video
        # https://stackoverflow.com/questions/74157935/getting-the-file-name-of-downloaded-video-using-yt-dlp#:~:text=import%20subprocess%20someFileType%20=%20subprocess.getoutput,4214%209

        ytd_params = {
            "format": "bestaudio/best",
            "outtmpl": "./download/%(title)s.%(ext)s",
        }
        with yt_dlp.YoutubeDL(ytd_params) as ydl:  # type: ignore
            for i in data["results"]:
                query = f"{i['artistName']} - {i['trackName']} official audio"
                ydl.download([f"ytsearch1: {query}"])

        for i in os.listdir("./download"):
            if i.startswith("."):  ## FIXs the .DS_Store ERROR
                continue
            file_path = os.path.join("./download", i)
            # print("-" * 30)
            # print(i)
            if i.endswith((".mp3", ".webm", ".m4a", ".aac")):
                print("-" * 30)
                ffmpeg.input(file_path).output(
                    os.path.splitext(file_path)[0] + ".wav", audio_bitrate="192k"
                ).run(overwrite_output=True)
                os.remove("./download/" + i)

        for filename in os.listdir("./download"):
            print(f"File Name with .wav {filename}")
            if filename.endswith(".wav"):
                shutil.move(os.path.join("./download", filename), "./data/")
                print("Moved to data")

def generate_Spectogram(audio_path):
    # wav, sr = librosa.load(audio_path, sr=44100, mono=False)
    wav, sr = librosa.load(audio_path, sr=None, mono=False)

    # If stereo, process both channels
    if wav.ndim == 2:
        wav = np.mean(wav, axis=0)  # ((9,0), (2,3)) -> (4.5, 2.5)

    # Generate spectrogram
    hop_length = 512  # Match your hopSize
    n_fft = 1024  # Match your windowSize

    D = librosa.stft(
        wav, n_fft=n_fft, hop_length=hop_length
    )  ## https://librosa.org/doc/latest/generated/librosa.decompose.decompose.html
    ## we still gotta figure out the diff b/w Librosa's fft and Scipy's fft and numpy's fft
    ## LINK https://stackoverflow.com/questions/56286595/librosas-fft-and-scipys-fft-are-different
    S = np.abs(D)  # Convert to magnitude spectrogram
    return S, sr, hop_length


# spec = generate_Spectogram(
#     "/Users/MacbookPro/LocalStorage/Developer/AI/Shazam/data/Hold On, We're Going Home (feat. Majid Jordan).wav"
# )


def fingerprint(spec, sr, hop):
    bands = [(0, 10), (10, 20), (20, 40), (40, 80), (80, 160), (160, 512)]
    fingerprint = []
    for i, frame in enumerate(spec.T):  ## RAMDOM DOCS SAID TO USE ENUMS
        time = i * hop / sr
        for low, high in bands:
            data = frame[low:high]
            if len(data) == 0:
                continue
            else:
                max_val = data[np.argmax(data)]

                if max_val > np.mean(frame):  ## Only keep if above threshold
                    freq_bin = low + np.argmax(data)
                    freq = freq_bin * sr / (2 * len(frame))
                    fingerprint.append((time, freq))  ## Saving time and freq on the DB

    return fingerprint


def create_hashes(peaks, target_size=5):
    fingerprints = {}

    for i, anchor in enumerate(peaks):
        for j in range(i + 1, min(i + target_size + 1, len(peaks))):
            target = peaks[j]

            anchor_freq = int(anchor[1] / 10) & 0x1FF  # 9 bits
            target_freq = int(target[1] / 10) & 0x1FF  # 9 bits
            delta_time = int((target[0] - anchor[0]) * 1000) & 0x3FFF  # 14 bits

            address = (anchor_freq << 23) | (target_freq << 14) | delta_time
            anchor_time_ms = int(anchor[0] * 1000)

            fingerprints[address] = anchor_time_ms

    return fingerprints


# audio_path = "/Users/MacbookPro/LocalStorage/Developer/AI/Shazam/data/Hold On, We're Going Home (feat. Majid Jordan).wav"

# # 1. Spectrogram
# spec, sr, hop = generate_Spectogram(audio_path)

# # 2. Peak extraction
# peaks = fingerprint(spec, sr, hop)

# # print(f"\nTotal Peaks Found: {len(peaks)}")
# # print("Sample Peaks (time, freq):")
# # for p in peaks[:20]:
# #     print(p)

# # 3. Hash generation
# hashes = create_hashes(peaks)

# # print(f"\nTotal Hashes Generated: {len(hashes)}")
# # print("Sample Hashes (address → anchor_time_ms):")
# # for k, v in list(hashes.items())[:20]:
# #     print(f"{k} → {v}")

