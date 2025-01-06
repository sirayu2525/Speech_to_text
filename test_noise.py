import librosa
import numpy as np
import soundfile as sf
import whisper
import os

def preprocess_audio(file_path, target_sr=16000):
    # 音声ファイルの読み込み
    audio, sr = librosa.load(file_path, sr=None)
    
    # サンプリングレートの調整
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
    
    # ノイズ除去（簡易的なハイパスフィルタ）
    audio = librosa.effects.high_pass_filter(audio, cutoff=100)
    
    # 音量の正規化
    audio = librosa.util.normalize(audio)
    
    # 処理済み音声の保存
    sf.write("preprocessed_audio.wav", audio, target_sr)
    
    return "preprocessed_audio.wav"

# 音声ファイルの前処理
original_file = os.path.join(os.getcwd(), "audio.mp3")
preprocessed_file = preprocess_audio(original_file)

# Whisperモデルのロード
model = whisper.load_model("base")

# 前処理前の音声ファイルの認識
original_result = model.transcribe(original_file)
print("前処理前の認識結果:", original_result["text"][:100])

# 前処理後の音声ファイルの認識
preprocessed_result = model.transcribe(preprocessed_file)
print("前処理後の認識結果:", preprocessed_result["text"][:100])
