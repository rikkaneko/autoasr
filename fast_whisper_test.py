from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download, snapshot_download

# https://huggingface.co/Systran
# https://huggingface.co/distil-whisper
model_id = "Systran/faster-whisper-large-v3"
model_path = f"models/{model_id}"
model_path = snapshot_download(repo_id=model_id, local_dir=model_path, revision="main")

model = WhisperModel(model_size_or_path=model_path, device="cpu", compute_type="int8_float32")
segments, info = model.transcribe("./input.wav", beam_size=5, language="zh", condition_on_previous_text=True, multilingual=True)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))