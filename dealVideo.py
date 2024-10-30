from moviepy.editor import *
import cv2

def convert_to_grayscale(input_path, output_path):
    # 打开输入视频
    # 加载视频
    clip = VideoFileClip(input_path)
    clip = clip.subclip(0, 32)
    # 转换每一帧为黑白
    gray_clip = clip.fl_image(lambda frame: frame.mean(axis=2).astype('uint8'))  # 转换为灰度

    # 将灰度帧转换为三通道
    gray_clip = gray_clip.fl_image(lambda frame: cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

    # 保存输出视频，保留音频
    gray_clip.write_videofile(output_path, audio=True)

def clip_and_get_sound():
    # 1. 加载视频文件
    video_clip = VideoFileClip(r"data/magray.mp4")
    # 裁剪第5秒到第10秒
    # clip_sub = video_clip.subclip(0, 4)
    # 2. 提取音频并保存为 WAV 文件
    audio_clip = video_clip.audio
    # clip_sub.write_videofile(r"D:\my\pythonProject\data\laoyoujiclip.mp4")
    audio_clip.write_audiofile(r"data/ma.wav")

if __name__=="__main__":
    #convert_to_grayscale("data/mayun.mp4","data/magray.mp4")
    clip_and_get_sound()


