from funasr import AutoModel
from huggingface_hub import snapshot_download
import pyaudio
import numpy as np
import wave

# Streaming config
chunk_size = [5, 10, 5]  # 600ms encoder chunks
encoder_chunk_look_back = 5
decoder_chunk_look_back = 1

model_id = "funasr/paraformer-zh-streaming"
model_path = f"models/{model_id}"
model_path = snapshot_download(repo_id=model_id, local_dir=model_path, revision="main")
vad_model_path = "./models/funasr/fsmn-vad"
vad_model_path = snapshot_download(repo_id="funasr/fsmn-vad", local_dir=vad_model_path, revision="main")
model = AutoModel(model=model_path, disable_update=True, disable_log=True, disable_pbar=True)
vad_model = AutoModel(model=vad_model_path, disable_update=True, disable_log=True, disable_pbar=True)

p = pyaudio.PyAudio()
audio_chunk_size=9600
istream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=audio_chunk_size)

ostream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    output=True,
    frames_per_buffer=audio_chunk_size
)

PCM_OUT_PATH = "input"
ASR_OUT_PATH = "output.txt"
pcm_file = open(f"{PCM_OUT_PATH}.pcm", "wb+")
asr_text = ""

cache_vad = {}
is_finished = True

try:
    print("Start recording...")
    print("> ", end="", flush=True)
    
    cache = {}
    while True:
        data = istream.read(audio_chunk_size, exception_on_overflow=False,)
        pcm_file.write(data)
        # Echo user input
        # ostream.write(data)
        
        chunk = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(chunk**2))
        peak = np.max(np.abs(chunk))
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        # Normization
        chunk = chunk.astype(np.float32) / 32768.0
            
        # print(f"audio rms={rms:6.0f} (0-32768), peak={peak:6.0f}", end='\r', flush=True)
        # Detect user's voice activity detection, keep blank frame to model
        vad_res = vad_model.generate(input=chunk, cache=cache_vad)
        has_speech = ("value" in vad_res[0] and len(vad_res[0]["value"]) > 0)
        if not is_finished or has_speech:
            is_finished = False
            # One more time after speech end to finish the last word
            if not has_speech:
                is_finished = True
            res = model.generate(input=chunk, cache=cache, is_final=is_finished, chunk_size=chunk_size,
                                encoder_chunk_look_back=encoder_chunk_look_back,
                                decoder_chunk_look_back=decoder_chunk_look_back,
                                )

            if res[0]['text'].strip():
                print(f"{res[0]['text'].strip()}", end='' , flush=True)
                asr_text += res[0]['text'].strip()
                
        # TODO Hanlde keypress

except KeyboardInterrupt:
    print("\nExited.")
    
finally:
    # Clean up
    istream.stop_stream()
    istream.close()
    p.terminate()
    pcm_file.close()
    print(f"Converting {PCM_OUT_PATH}.pcm to WAV format...")
    with wave.open(f"{PCM_OUT_PATH}.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(open(f"{PCM_OUT_PATH}.pcm", "rb").read())
    print(f"Saved {PCM_OUT_PATH}.wav [Audio Stream]")
    with open(ASR_OUT_PATH, "w+") as f:
        f.write(asr_text)
    print(f"Saved {ASR_OUT_PATH} [ASR Ouput]")