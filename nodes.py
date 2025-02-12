import torch
import random
import librosa
import zipfile
import torchaudio
import numpy as np
import os,sys
import folder_paths

now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
#output_dir = os.path.join(folder_paths.get_output_directory(),"cosyvoice_dubb")
pretrained_models = os.path.join(now_dir,"pretrained_models")
#from modelscope import snapshot_download

import ffmpeg
from cosyvoice.cli.cosyvoice import CosyVoice2

from modelscope import snapshot_download

sft_spk_list = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
inference_mode_list = ['3s极速复刻', '跨语种复刻', '自然语言控制']

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
prompt_sr, target_sr = 16000, 22050
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

def speed_change(input_audio, speed, sr):
    # 检查输入数据类型和声道数
    if input_audio.dtype != np.int16:
        raise ValueError("输入音频数据类型必须为 np.int16")


    # 转换为字节流
    raw_audio = input_audio.astype(np.int16).tobytes()

    # 设置 ffmpeg 输入流
    input_stream = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ar=str(sr), ac=1)

    # 变速处理
    output_stream = input_stream.filter('atempo', speed)

    # 输出流到管道
    out, _ = (
        output_stream.output('pipe:', format='s16le', acodec='pcm_s16le')
        .run(input=raw_audio, capture_stdout=True, capture_stderr=True)
    )

    # 将管道输出解码为 NumPy 数组
    processed_audio = np.frombuffer(out, np.int16)

    return processed_audio

class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "CosyVoice2"

    def encode(self,text):
        return (text, )

from time import time as ttime
class CosyVoice2Node:
    def __init__(self):
        self.model_dir = None
        self.cosyvoice = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "tts_text":("TEXT",),
                "speed":("FLOAT",{
                    "default": 1.0
                }),
                "inference_mode":(inference_mode_list,{
                    "default": "3s极速复刻"
                }),
                "text_frontend": ("BOOLEAN", {"default": False, "tooltip": "Enable text frontend or not"}),
                "seed":("INT",{
                    "default": 42
                })
            },
            "optional":{
                "prompt_text":("TEXT",),
                "prompt_wav": ("AUDIO",),
                "instruct_text":("TEXT",),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "CosyVoice2"

    def generate(self,tts_text,speed,inference_mode,text_frontend,seed,
                 prompt_text=None,prompt_wav=None,instruct_text=None):
        t0 = ttime()
        # if inference_mode == '自然语言控制':
        #     model_dir = os.path.join(pretrained_models,"CosyVoice-300M-Instruct")
        #     snapshot_download(model_id="iic/CosyVoice-300M-Instruct",local_dir=model_dir)
        #     assert instruct_text is not None, "in 自然语言控制 mode, instruct_text can't be none"
        # if inference_mode in ["跨语种复刻",'3s极速复刻']:
        #     model_dir = os.path.join(pretrained_models,"CosyVoice-300M")
        #     snapshot_download(model_id="iic/CosyVoice-300M",local_dir=model_dir)
        #     assert prompt_wav is not None, "in 跨语种复刻 or 3s极速复刻 mode, prompt_wav can't be none"
        #     if inference_mode == "3s极速复刻":
        #         assert len(prompt_text) > 0, "prompt文本为空，您是否忘记输入prompt文本？"
        # if inference_mode == "预训练音色":
        #     model_dir = os.path.join(pretrained_models,"CosyVoice-300M-SFT")
        #     snapshot_download(model_id="iic/CosyVoice-300M-SFT",local_dir=model_dir)

        model_dir = os.path.join(pretrained_models,"CosyVoice2-0.5B")
        if self.cosyvoice is None:
            self.cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

        # if self.model_dir != model_dir:
        #     self.model_dir = model_dir
        #     self.cosyvoice = CosyVoice2(model_dir)
        
        if prompt_wav:
            waveform = prompt_wav['waveform'].squeeze(0)
            source_sr = prompt_wav['sample_rate']
            speech = waveform.mean(dim=0,keepdim=True)
            if source_sr != prompt_sr:
                speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        if inference_mode == '3s极速复刻':
            print('get zero_shot inference request')
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=False, text_frontend=text_frontend)
        elif inference_mode == '跨语种复刻':
            print('get cross_lingual inference request')
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            set_all_random_seed(seed)
            output = self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False, text_frontend=text_frontend)
        else:
            print('get instruct inference request')
            set_all_random_seed(seed)
            print(self.model_dir)
            prompt_speech_16k = postprocess(speech)
            output = self.cosyvoice.inference_instruct2(tts_text, instruct_text,prompt_speech_16k, stream=False, text_frontend=text_frontend)
        output_list = []
        for out_dict in output:
            output_numpy = out_dict['tts_speech'].squeeze(0).numpy() * 32768 
            output_numpy = output_numpy.astype(np.int16)
            if speed > 1.0 or speed < 1.0:
                output_numpy = speed_change(output_numpy,speed,target_sr)
            output_list.append(torch.Tensor(output_numpy/32768).unsqueeze(0))
        t1 = ttime()
        print("cost time \t %.3f" % (t1-t0))
        audio = {"waveform": torch.cat(output_list,dim=1).unsqueeze(0),"sample_rate":target_sr}
        return (audio,)

class CosyVoice2ModelDownload:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "download"
    CATEGORY = "CosyVoice2"
    OUTPUT_NODE = True
    def download(self,text):
        print("check and download CosyVoice2-0.5B")
        model_dir = os.path.join(pretrained_models,"CosyVoice2-0.5B")
        snapshot_download(model_id="iic/CosyVoice2-0.5B",local_dir=model_dir)
        print("check and download CosyVoice-ttsfrd")        
        model_dir = os.path.join(pretrained_models,"CosyVoice-ttsfrd")        
        snapshot_download('iic/CosyVoice-ttsfrd', local_dir=model_dir)
        print("download done")
        return (text, )
    
# class LoadSRT:
#     @classmethod
#     def INPUT_TYPES(s):
#         files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["srt", "txt"]]
#         return {"required":
#                     {"srt": (sorted(files),)},
#                 }

#     CATEGORY = "AIFSH_CosyVoice"

#     RETURN_TYPES = ("SRT",)
#     FUNCTION = "load_srt"

#     def load_srt(self, srt):
#         srt_path = folder_paths.get_annotated_filepath(srt)
#         return (srt_path,)
