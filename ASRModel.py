import openvino_genai
import librosa

def prepare_srt(transcription):
    """
    Format transcription into srt file format
    """
    segment_lines = []
    for idx, segment in enumerate(transcription.chunks):
        srt_info={}
        srt_info['start']=segment.start_ts
        srt_info['end'] = segment.end_ts
        srt_info['text']=segment.text
        segment_lines.append(srt_info)
    return segment_lines

if __name__=="__main__":
    en_example_short="data/ma.wav"
    en_raw_speech, samplerate = librosa.load(str(en_example_short), sr=16000)
    model_path=r"D:\my\pythonProject\whisper-tiny-en\whisper-tiny.en"

    ov_pipe = openvino_genai.WhisperPipeline(str(model_path), device="GPU")
    #这里测试了一下，token不能太长否则效果不好
    genai_result = ov_pipe.generate(en_raw_speech,return_timestamps=True)
    prepare_srt(genai_result)
    # import pdb
    # pdb.set_trace()
    srt_lines = prepare_srt(genai_result)
    print(srt_lines)
    # print(f"Result: {genai_result}")