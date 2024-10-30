import openvino_genai
import librosa

class AudioRecongize:
    def __init__(self,model_path_dir):
        self.ov_pipe = openvino_genai.WhisperPipeline(model_path_dir, device="GPU")

    def prepare_srt(self,genai_result):
        segment_lines = []
        for idx, segment in enumerate(genai_result.chunks):
            srt_info = {}
            srt_info['start'] = segment.start_ts
            srt_info['end'] = segment.end_ts
            srt_info['text'] = segment.text
            segment_lines.append(srt_info)
        return segment_lines

    def generateFile(self,en_raw_speech):
        genai_result = self.ov_pipe.generate(en_raw_speech, return_timestamps=True)
        srt_lines=self.prepare_srt(genai_result)
        return srt_lines

    def generateWave(self,text,language="zh"):
        wav=self.tts.tts(text=text,speaker_wav="speaker.wav",language=language)
        return wav