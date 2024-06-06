import os
import argparse
import ffmpeg
import math
import copy
import argostranslate.package
import argostranslate.translate
import shutil
from pathlib import Path
from faster_whisper import WhisperModel


# Storing in a file and not in memory to decrease RAM usage
def extract_audio(video_path):
    Path('temp/audio/').mkdir(parents=True, exist_ok=True)
    audio_path = f"temp/audio/{Path(video_path).stem}-audio.wav"

    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_path)
    ffmpeg.run(stream)

    return audio_path

def transcribe(audio_path, model, device, compute_type):
    os.environ['KMP_DUPLICATE_LIB_OK'] = "True"
    model = WhisperModel(model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path)
    return list(segments), info.language

def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time

def download_translation_package(from_language, to_language):
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

def translate_segment(segment, from_language, to_language):
    return segment._replace(text=argostranslate.translate.translate(segment.text, from_language, to_language))

def translate_segments(segments, from_language, to_language):
    if(from_language == to_language):
        return segments
    
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_language and x.to_code == to_language, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())

    translated_segments = [translate_segment(segment, from_language, to_language) for segment in segments]

    return translated_segments

def generate_srt_file(language, segments):
    Path('temp/subtitles/').mkdir(parents=True, exist_ok=True)
    srt_path = f"temp/subtitles/subtitles.{language}.srt"

    stream_srt_file = open(srt_path, 'w', encoding='utf-8')

    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        segment_text = ""
        segment_text += f"{str(index+1)} \n"
        segment_text += f"{segment_start} --> {segment_end} \n"
        segment_text += f"{segment.text} \n"
        segment_text += "\n"
        stream_srt_file.write(segment_text)

    stream_srt_file.close()

    return srt_path

def add_srt_to_video(video_path, output_path, srt_path_list):
    video_stream = ffmpeg.input(video_path)

    subtitle_streams = []

    subtitle_index = 0
    language_metadata = {}
    for language, srt_path in srt_path_list:
        srt_stream = ffmpeg.input(srt_path)
        subtitle_streams.append(srt_stream['s'])

        language_metadata['metadata:s:s:'+str(subtitle_index)] = "language="+language
        subtitle_index = subtitle_index + 1

    input_video = video_stream['v']
    input_audio = video_stream['a']

    output_ffmpeg = ffmpeg.output(
        input_video, input_audio, *subtitle_streams, output_path,
        vcodec='copy', acodec='copy', 
        **language_metadata
    )
    output_ffmpeg = ffmpeg.overwrite_output(output_ffmpeg)
    print(ffmpeg.compile(output_ffmpeg))
    ffmpeg.run(output_ffmpeg)


def make_sub(video_path, output_path, model, device, compute_type, from_language, to_languages):
    if(output_path == None):
        input_path = Path(video_path)
        output_path = Path(input_path.parent).joinpath(input_path.stem + '_with_subtitles.mkv').absolute().as_posix()

    audio_path = extract_audio(video_path)
    segments, detected_language = transcribe(audio_path, model, device, compute_type)

    if(from_language == None):
        from_language = detected_language

    if(to_languages == None):
        to_languages = [from_language]

    en_segments = None
    if(from_language != 'en' and to_languages != ['en'] and [from_language] != to_languages):
        en_segments = translate_segments(segments, from_language, 'en')
    
    srt_path_list = []
    for to_language in to_languages:
        translated_segments = None
        if(from_language != 'en' and to_languages != ['en'] and from_language != to_language):
            translated_segments = translate_segments(copy.deepcopy(en_segments), 'en', to_language)
        else:
            translated_segments = translate_segments(copy.deepcopy(segments), from_language, to_language)

        srt_path = generate_srt_file(to_language, translated_segments)
        srt_path_list.append((to_language, srt_path))

    add_srt_to_video(video_path, output_path, srt_path_list)

    # shutil.rmtree('temp/')

def main():
    parser = argparse.ArgumentParser(
        prog='video-sub-maker',
        description='A python tool to generate and add soft subtitles in any language to video from any language using faster-wisper, ffmpeg and argos-translate.'
    )
    parser.add_argument(
        'video_path',
        help='The path for the video you want to add subtitles'
    )
    parser.add_argument(
        '-o',
        '--output',
        required=False,
        help='The path for the created video with subtitles'
    )
    parser.add_argument(
        '-m',
        '--model',
        default='large-v2',
        required=False,
        help='The model used to transcribe the video (see SYSTRAN/faster-whisper)'
    )
    parser.add_argument(
        '-d',
        '--device',
        default='cuda',
        required=False,
        help='The device used for transcribing (cuda for GPU or cpu for CPU) (see SYSTRAN/faster-whisper)'
    )
    parser.add_argument('-c',
        '--compute_type',
        default='float16',
        required=False,
        help='The compute type used for transcribing (see SYSTRAN/faster-whisper)'
    )
    parser.add_argument(
        '-f',
        '--from_language',
        required=False,
        help='The source language of the video (if not provided, it will be try to automatically detect it)'
    )
    parser.add_argument(
        '-t',
        '--to_languages',
        required=False,
        help='The language of the generated subtitles (if not provided, same as from_language)',
        action='append'
    )
    args = parser.parse_args()

    make_sub(args.video_path, args.output, args.model, args.device, args.compute_type, args.from_language, args.to_languages)

if __name__ == '__main__':
    main()