from moviepy.editor import *
import soundfile as sf
from tqdm import tqdm
import librosa
import numpy as np
import argparse
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
if __name__=="__main__":
    temp_video = 'temp/color.mp4'
    temp_sound = "temp/sound.wav"
    temp_china_sount = "temp/temp_sound.wav"
    output_video = "data/macolorchina.mp4"
    merge_sound_video(temp_video, temp_china_sount, output_video)