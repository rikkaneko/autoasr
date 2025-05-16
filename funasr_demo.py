from funasr import AutoModel
from huggingface_hub import snapshot_download

chunk_size = 200 #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model_id = "funasr/fsmn-vad"
model_path = f"models/{model_id}"
model_path = snapshot_download(repo_id=model_id, local_dir=model_path, revision="main")
model = AutoModel(model=model_path, disable_update=True, disable_log=True, disable_pbar=True)

import soundfile

wav_file = f"./models/funasr/fsmn-vad/example/vad_example.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
    if len(res[0]["value"]):
        print(res)