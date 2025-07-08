import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


model_id = "jlvdoorn/whisper-large-v3-atco2-asr-atcosim"


processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

audio_path = "test2.wav"
print(f"Loading and resampling {audio_path}...")
audio, sr = librosa.load(audio_path, sr=16000)

chunk_duration = 10  
chunk_samples = int(chunk_duration * sr)
total_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

print(f"Processing {total_chunks} chunks of {chunk_duration} seconds each...")

full_transcription = []

for i in range(total_chunks):
    start_sample = i * chunk_samples
    end_sample = min((i + 1) * chunk_samples, len(audio))
    chunk = audio[start_sample:end_sample]
    
    print(f"Processing chunk {i+1}/{total_chunks} ({start_sample/sr:.1f}s - {end_sample/sr:.1f}s)...")
    
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    
    chunk_transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    full_transcription.append(chunk_transcription)

print("\nFull Transcription:\n", " ".join(full_transcription))
