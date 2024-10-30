import openvino_genai
import librosa
en_example_short="outputyinhun.wav"
en_raw_speech, samplerate = librosa.load(str(en_example_short), sr=16000)
model_path=r"D:\my\pythonProject\whisper-tiny\whisper-tiny"
ov_pipe = openvino_genai.WhisperPipeline(str(model_path), device="GPU")
languages_genai = {
    "japanese": "<|ja|>",
    "dutch": "<|da|>",
    "french": "<|fr|>",
    "spanish": "<|es|>",
    "italian": "<|it|>",
    "portuguese": "<|pt|>",
    "polish": "<|pl|>",
}
genai_result = ov_pipe.generate(en_raw_speech,task="translate", language=languages_genai["japanese"])
print(f"Result: {genai_result}")