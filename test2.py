import whisper
import time
import os

models = ["tiny", "base", "small", "medium", "large"]
audio_file = os.path.join(os.getcwd(), "audio.mp3")

for model_size in models:
    print(f"モデルサイズ: {model_size}")
    
    # モデルのロード
    model = whisper.load_model(model_size)
    
    # 処理時間の計測開始
    start_time = time.time()
    
    # 音声ファイルの文字起こし
    result = model.transcribe(audio_file)
    
    # 処理時間の計測終了
    end_time = time.time()
    
    print(f"処理時間: {end_time - start_time:.2f}秒")
    print(f"認識結果: {result['text'][:100]}...")  # 最初の100文字のみ表示
    print("---\n")
