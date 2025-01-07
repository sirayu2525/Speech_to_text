# Description: 長時間音声ファイルの文字起こし処理のサンプルコード
import whisper
import librosa
import numpy as np
import os

def process_long_audio(file_path, model, chunk_length_sec=30):
    # 音声ファイルの読み込み
    audio, sr = librosa.load(file_path, sr=16000)
    
    # チャンクサイズの計算（サンプル数）
    chunk_length = sr * chunk_length_sec
    
    # 音声を分割して処理
    transcriptions = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        
        # 一時ファイルに保存
        temp_file = f"temp_chunk_{i}.wav"
        librosa.output.write_wav(temp_file, chunk, sr)
        
        # Whisperで処理
        result = model.transcribe(temp_file)
        transcriptions.append(result["text"])
        
        # 一時ファイルの削除
        os.remove(temp_file)
    
    # 結果の結合
    full_transcription = " ".join(transcriptions)
    
    return full_transcription

# Whisperモデルのロード
model = whisper.load_model("base")

# 長時間音声ファイルの処理
long_audio_file = "path/to/your/long_audiofile.mp3"
result = process_long_audio(long_audio_file, model)

print("長時間音声の文字起こし結果:")
print(result)
