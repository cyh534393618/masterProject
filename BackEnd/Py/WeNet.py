import wenet
import soundfile as sf

model = wenet.load_model('english')
# or model = wenet.load_model(model_dir='xxx')
result = model.transcribe('./files/output_file.wav')
print(result['text'])