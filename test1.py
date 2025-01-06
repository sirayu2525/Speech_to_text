import whisper
import os

# モデルのロード
model = whisper.load_model("tiny")

audio_file = os.path.join(os.getcwd(), "audio.mp3")
# 音声ファイルの文字起こし
result = model.transcribe(audio_file)

# 結果の表示
print(result["text"])

# 詳細な結果の表示
for segment in result["segments"]:
    print(f"開始時間: {segment['start']:.2f}秒")
    print(f"終了時間: {segment['end']:.2f}秒")
    print(f"テキスト: {segment['text']}")
    print("---")
