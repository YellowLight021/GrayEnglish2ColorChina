from audioGenerator import AudioGenerator
from audioRecognize import AudioRecongize
from translateClass import Translation
from videoColorization import ColorizationPipeline
from moviepy.editor import *
import soundfile as sf
from tqdm import tqdm
import librosa
import numpy as np
import argparse

def translate_and_tts(audio_generator,translator,text):
    new_text=translator.from_en_to_ch(text)
    new_sound=audio_generator.generateWave(new_text[0])
    return new_sound
def merge_sound_video(temp_video,temp_china_sount,output_video):
    input_video_path = temp_video  # 视频文件路径
    input_audio_path = temp_china_sount  # 新的音频文件路径
    output_video_path = output_video  # 输出合并后的视频文件路径

    # 加载视频和音频文件
    video_clip = VideoFileClip(input_video_path)
    audio_clip = AudioFileClip(input_audio_path)

    # # 确保音频长度与视频长度相同
    # if audio_clip.duration != video_clip.duration:
    #     print("audio_clip:{}".format(audio_clip.duration))
    #     print("video_clip:{}".format(video_clip.duration))
    #     raise ValueError("The length of the audio must match the length of the video.")

    # 将新的音频合并到视频中
    final_video = video_clip.set_audio(audio_clip)

    # 保存合并后的视频
    final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    # 释放资源
    video_clip.close()
    audio_clip.close()

    print(f"Merged video saved as {output_video_path}")

def main(input_video,output_video,refSound,addColor):
    temp_video='temp/color.mp4'
    temp_sound="temp/sound.wav"
    temp_china_sount="temp/temp_sound.wav"
    input_video="data/magray.mp4"
    output_video="data/macolorchina.mp4"
    if addColor:
        print("start colorization......")
        colorizer = ColorizationPipeline("ddcolor/ddcolor.xml")
        colorizer.colorize_video(input_video, temp_video)
    print("start audio......")
    video_clip = VideoFileClip(input_video)
    desired_duration = video_clip.duration
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(temp_sound)

    en_raw_speech, samplerate = librosa.load(temp_sound, sr=16000)
    audio_recognize=AudioRecongize("whisper-tiny-en/whisper-tiny.en")
    audio_generator=AudioGenerator("tts_models/multilingual/multi-dataset/xtts_v2")
    translator=Translation("entoch_ov_model")
    recognize_result=audio_recognize.generateFile(en_raw_speech)
    processed_audio = en_raw_speech.copy()

    for text_clip in tqdm(recognize_result):
        start_sample = int(text_clip["start"] * samplerate)  # 转换为样本点
        end_sample = int(text_clip["end"]  * samplerate)  # 转换为样本点
        text=text_clip['text']
        processed_segment=translate_and_tts(audio_generator,translator,text)
        stretch_factor =(end_sample-start_sample)/len(processed_segment) # 时间拉伸因子
        processed_segment = librosa.effects.time_stretch(np.array(processed_segment), rate=1/stretch_factor)
        # 将处理后的音频融回原音频
        processed_audio[start_sample:start_sample + len(processed_segment)] = processed_segment

    adjusted_length = len(processed_audio)
    desired_length = int(desired_duration * samplerate)
    if adjusted_length < desired_length:
        # 填充音频
        padding = desired_length - adjusted_length
        processed_audio = np.pad(processed_audio, (0, padding), mode='constant')
    elif adjusted_length > desired_length:
        # 剪裁音频
        processed_audio = processed_audio[:desired_length]

    sf.write(temp_china_sount, processed_audio, samplerate)
    print("start merge the video and china sound")
    merge_sound_video(temp_video,temp_china_sount,output_video)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="transfer a english video to chinese video.")
    parser.add_argument("--input", type=str, help="input video")
    parser.add_argument("--output", type=str, help="input video")
    parser.add_argument("--refSound", type=str, help="reference Sound")
    parser.add_argument("--addColor", type=bool,default=False, help="if add color")

    args = parser.parse_args()
    main(args.input, args.output,args.refSound,args.addColor)