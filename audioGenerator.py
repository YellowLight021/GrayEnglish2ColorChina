from TTS.api import TTS

# generate speech by cloning a voice using default settings
class AudioGenerator:
    def __init__(self,model_path_dir):
        self.tts = TTS(model_path_dir, gpu=False)
    def generateFile(self,text,savePath,ref_wav,language="zh"):
        self.tts.tts_to_file(text=text,
                        file_path=savePath,
                        speaker_wav=ref_wav,
                        language=language)
    def generateWave(self,text,language="zh"):
        wav=self.tts.tts(text=text,speaker_wav="data/ref.wav",language=language)
        return wav

class AudioGeneratorBark:
    def __init__(self):
        from bark.generation import load_model, load_codec_model
        from ovPredictor import OVBarkPredictor
        import torch
        torch.manual_seed(42)
        text_encoder = load_model(model_type="text", use_gpu=False, use_small=True, force_reload=False)
        load_codec_model(use_gpu=False)
        tokenizer = text_encoder["tokenizer"]
        text_encoder_path0 = r"D:\my\bark\models\models\text_encoder_small\bark_text_encoder_0.xml"
        text_encoder_path1 = r"D:\my\bark\models\models\text_encoder_small\bark_text_encoder_1.xml"
        fine_model_dir = r"D:\my\bark\models\models\fine_model"
        coarse_encoder_path = r"D:\my\bark\models\models\coarse_small\bark_coarse_encoder.xml"
        self.ttsGenrator = OVBarkPredictor(tokenizer, text_encoder_path0, text_encoder_path1, coarse_encoder_path,
                                      fine_model_dir, "CPU")
    def generateFile(self,text,savePath):
        self.ttsGenrator.saveWave(text,savePath)

if __name__=="__main__":
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    wav = tts.tts(text="Hello world!", speaker_wav="data/ref.wav", language="en")
