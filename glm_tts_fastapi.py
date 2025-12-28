import os
import logging
import torch
import torchaudio
import numpy as np
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, AsyncGenerator
import tempfile
import shutil
from contextlib import asynccontextmanager
import time
import gc
import fastapi_cdn_host
from enum import Enum
import io
import wave
import struct

# --- 全局常量 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LLM_SEQ_INP_LEN = 750
REFERENCE_VOICES_DIR = os.path.join(CURRENT_DIR, "reference_voices")

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 模型缓存系统 ---
MODEL_CACHE = {
    "loaded": False,
    "sample_rate": None,
    "components": None,
    "use_phoneme": None
}

# --- 全局变量 ---
frontend = None
text_frontend = None
speech_tokenizer = None
llm = None
flow = None
temp_dirs = []


# --- 导入必要模块 ---
from cosyvoice.cli.frontend import TTSFrontEnd, SpeechTokenizer, TextFrontEnd
from utils import file_utils, seed_util
from utils import tts_model_util, yaml_util
from transformers import AutoTokenizer, LlamaForCausalLM
from llm.glmtts import GLMTTS
from utils.audio import mel_spectrogram
from functools import partial

try :
    from semantic_text_splitter import SemanticTextSplitter

    # 创建分割器实例
    splitter = SemanticTextSplitter()

except :
    from semantic_text_splitter import SplitterConfig, CachedSemanticTextSplitter
    # 基本使用
    # config = SplitterConfig(min_length=30, max_length=60)
    # splitter = SemanticTextSplitter(config)
    # segments = splitter.split_text(your_text)

    # 高级配置
    advanced_config = SplitterConfig(
        min_length=50,
        max_length=100,
        chunk_strategy="recursive",
        use_advanced_segmentation=True,
        debug_mode=False
    )

    # 带缓存的分割器（提高重复分割性能）
    cached_splitter = CachedSemanticTextSplitter(advanced_config, cache_size=500)
    splitter = cached_splitter


# --- 音色类型枚举 ---
class VoiceType(str, Enum):
    BUILTIN = "builtin"  # 使用内置参考音色
    UPLOAD = "upload"    # 上传自定义参考音色

# --- 音色选择模型 ---
class VoiceSelection(BaseModel):
    type: VoiceType = Field(VoiceType.BUILTIN, description="音色类型：builtin(内置)或upload(上传)")
    builtin_voice_id: Optional[str] = Field(None, description="内置音色ID，当type为builtin时必需")

    @validator('builtin_voice_id')
    def validate_builtin_voice_id(cls, v, values):
        if 'type' in values and values['type'] == VoiceType.BUILTIN and v is None:
            raise ValueError("使用内置音色时，builtin_voice_id参数必需")
        return v

# --- 核心模型函数 ---
def get_special_token_ids(tokenize_fn):
    """获取特殊token ID"""
    _special_token_ids = {
        "ats": "<|audio_0|>",
        "ate": "<|audio_32767|>",
        "boa": "<|begin_of_audio|>",
        "eoa": "<|user|>",
        "pad": "",
    }

    special_token_ids = {}
    # 处理空字符串编码
    empty_tokens = tokenize_fn("")
    if len(empty_tokens) == 0:
        logger.warning("Empty string encoded to empty list, trying alternative approaches")
        empty_tokens = tokenize_fn(" ")
        if len(empty_tokens) == 0:
            empty_tokens = tokenize_fn("<|pad|>")
            if len(empty_tokens) == 0:
                empty_tokens = tokenize_fn("a")
                if len(empty_tokens) == 0:
                    raise ValueError("Tokenizer returned empty list for all test inputs")
    
    endoftext_id = empty_tokens[0]
    logger.info(f"Endoftext ID determined as: {endoftext_id}")

    for k, v in _special_token_ids.items():
        __ids = tokenize_fn(v)
        logger.info(f"Token '{k}' ({v}) encoded to: {__ids}")
        
        if k == "pad" and len(__ids) == 0:
            logger.warning(f"Token 'pad' encoded to empty list, using endoftext_id {endoftext_id} as fallback")
            special_token_ids[k] = endoftext_id
            continue
            
        # 确保token长度为1
        if len(__ids) != 1:
            logger.warning(f"Token '{k}' ({v}) encoded to multiple tokens: {__ids}, using first token")
            if len(__ids) > 0:
                __ids = [__ids[0]]
            else:
                logger.warning(f"Token '{k}' ({v}) encoded to empty list, using fallback value")
                special_token_ids[k] = endoftext_id + len(special_token_ids) + 100
                continue
        
        token_id = __ids[0]
        if token_id < endoftext_id:
            logger.warning(f"Token '{k}' ({v}) ID {token_id} is smaller than endoftext ID {endoftext_id}, but proceeding anyway")

        special_token_ids[k] = token_id

    logger.info(f"Final special token IDs: {special_token_ids}")
    return special_token_ids

def _assert_shape_and_get_len(token):
    """验证token形状并获取长度"""
    assert token.ndim == 2 and token.shape[0] == 1
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32).to(token.device)
    return token_len

def load_frontends(speech_tokenizer, sample_rate=24000, use_phoneme=False, frontend_dir="frontend"):
    """加载前端模型"""
    if sample_rate == 32000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=640, n_fft=2560, 
                               num_mels=80, win_size=2560, fmin=0, fmax=8000, center=False)
        print("Configured for 32kHz frontend.")
    elif sample_rate == 24000:
        feat_extractor = partial(mel_spectrogram, sampling_rate=sample_rate, hop_size=480, n_fft=1920, 
                               num_mels=80, win_size=1920, fmin=0, fmax=8000, center=False)
        print("Configured for 24kHz frontend.")
    else:
        raise ValueError(f"Unsupported sampling_rate: {sample_rate}")

    glm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join('ckpt', 'vq32k-phoneme-tokenizer'), trust_remote_code=True
    )
    tokenize_fn = lambda text: glm_tokenizer.encode(text)

    frontend = TTSFrontEnd(
        tokenize_fn,
        speech_tokenizer,
        feat_extractor,
        os.path.join(frontend_dir, "campplus.onnx"),
        os.path.join(frontend_dir, "spk2info.pt"),
        DEVICE,
    )
    text_frontend = TextFrontEnd(use_phoneme)
    return frontend, text_frontend

def local_llm_forward(llm, prompt_text_token, tts_text_token, prompt_speech_token, 
                     beam_size=1, sampling=25, sample_method="ras"):
    """单次LLM推理"""
    prompt_text_token_len = _assert_shape_and_get_len(prompt_text_token)
    tts_text_token_len = _assert_shape_and_get_len(tts_text_token)
    prompt_speech_token_len = _assert_shape_and_get_len(prompt_speech_token)

    tts_speech_token = llm.inference(
        text=tts_text_token,
        text_len=tts_text_token_len,
        prompt_text=prompt_text_token,
        prompt_text_len=prompt_text_token_len,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        beam_size=beam_size,
        sampling=sampling,
        sample_method=sample_method,
        spk=None,
    )
    return tts_speech_token[0].tolist()

def local_flow_forward(flow, token_list, prompt_speech_tokens, speech_feat, embedding):
    """单次Flow推理"""
    wav, full_mel = flow.token2wav_with_cache(
        token_list,
        prompt_token=prompt_speech_tokens,
        prompt_feat=speech_feat,
        embedding=embedding,
    )
    return wav.detach().cpu(), full_mel

def get_cached_prompt(cache, synth_text_token, device=DEVICE):
    """从缓存构建提示tokens"""
    cache_text = cache["cache_text"]
    cache_text_token = cache["cache_text_token"]
    cache_speech_token = cache["cache_speech_token"]

    def __len_cache_text_token():
        return sum(map(lambda x: x.shape[1], cache_text_token))

    def __len_cache_speech_token():
        return sum(map(len, cache_speech_token))

    text_len = __len_cache_text_token()
    ta_ratio = __len_cache_speech_token() / (text_len if text_len > 0 else 1.0)
    __len_synth_text_token = synth_text_token.shape[1]
    __len_synth_audi_token_estim = int(ta_ratio * __len_synth_text_token)

    # 修剪缓存如果太长
    while __len_cache_speech_token() + __len_synth_audi_token_estim > MAX_LLM_SEQ_INP_LEN:
        if len(cache_speech_token) <= 1:
            break
        cache_text.pop(1)
        cache_text_token.pop(1)
        cache_speech_token.pop(1)

    # 构建文本提示
    prompt_text_token_from_cache = []
    for a_token in cache_text_token:
        prompt_text_token_from_cache.extend(a_token.squeeze().tolist())

    prompt_text_token = torch.tensor([prompt_text_token_from_cache]).to(device)

    # 构建语音提示
    speech_tokens = []
    for a_cache_speech_token in cache_speech_token:
        speech_tokens.extend(a_cache_speech_token)

    llm_speech_token = torch.tensor([speech_tokens], dtype=torch.int32).to(device)
    return prompt_text_token, llm_speech_token

async def generate_streaming(
    frontend, text_frontend, llm, flow, text_info, cache, device, embedding, 
    seed=0, sample_method="ras", flow_prompt_token=None, speech_feat=None,
    use_phoneme=False, chunk_size=1024
):
    """生成长语音，支持流式输出"""
    full_mels = []
    output_token_list = []
    uttid = text_info[0]
    syn_text = text_info[1]
    text_tn_dict = {
        "uttid": uttid,
        "syn_text": syn_text,
        "syn_text_tn": [],
        "syn_text_phoneme": [],
    }
    # short_text_list = text_frontend.split_by_len(syn_text)
    # print(f"before:{short_text_list}")
    short_text_list = splitter.gyc_split_text(syn_text)
    print(f"my_func:{short_text_list}")

    for segment_idx, tts_text in enumerate(short_text_list):
        seed_util.set_seed(seed + segment_idx)
        tts_text_tn = text_frontend.text_normalize(tts_text)
        print(f"real text : {tts_text_tn}")
        text_tn_dict["syn_text_tn"].append(tts_text_tn)
        if use_phoneme:
            tts_text_tn = text_frontend.g2p_infer(tts_text_tn)
            text_tn_dict["syn_text_phoneme"].append(tts_text_tn)
        tts_text_token = frontend._extract_text_token(tts_text_tn)

        cache_text = cache["cache_text"]
        cache_text_token = cache["cache_text_token"]
        cache_speech_token = cache["cache_speech_token"]


        # 确定提示
        if cache["use_cache"] and len(cache_text_token) > 1:
            prompt_text_token, prompt_speech_token = get_cached_prompt(cache, tts_text_token, device)
        else:
            prompt_text_token = cache_text_token[0].to(device)
            prompt_speech_token = torch.tensor([cache_speech_token[0]], dtype=torch.int32).to(device)
            logger.debug(f"[generate_streaming] Using initial prompt (empty cache history) for segment {segment_idx}")

        # LLM推理
        token_list_res = local_llm_forward(
            llm=llm,
            prompt_text_token=prompt_text_token,
            tts_text_token=tts_text_token,
            prompt_speech_token=prompt_speech_token,
            sample_method=sample_method
        )
        output_token_list.extend(token_list_res)

        # Flow推理
        output, full_mel = local_flow_forward(
            flow=flow,
            token_list=token_list_res,
            prompt_speech_tokens=flow_prompt_token,
            speech_feat=speech_feat,
            embedding=embedding
        )

        # 更新缓存
        if cache is not None:
            cache_text.append(tts_text_tn)
            cache_text_token.append(tts_text_token)
            cache_speech_token.append(token_list_res)

        if full_mel is not None:
            full_mels.append(full_mel)
        
        # 音频后处理
        audio_data = output.squeeze().cpu().numpy()
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767.0).astype(np.int16)
        
        if audio_int16.ndim == 1:
            audio_int16 = audio_int16.reshape(1, -1)
        
        # 转换为字节数据
        audio_bytes = audio_int16.tobytes()
        
        # 按块分割音频数据
        for i in range(0, len(audio_bytes), chunk_size * 2):  # *2 because int16 is 2 bytes
            chunk = audio_bytes[i:i + chunk_size * 2]
            if chunk:
                yield {
                    "segment_idx": segment_idx,
                    "audio_chunk": chunk,
                    "is_last_segment": False,
                    "total_segments": len(short_text_list)
                }

    # 发送结束信号
    yield {
        "segment_idx": len(short_text_list) - 1,
        "audio_chunk": b"",
        "is_last_segment": True,
        "total_segments": len(short_text_list),
        "text_info": text_tn_dict
    }

# --- 模型管理 ---
def clear_model_cache():
    """清除模型缓存和释放GPU内存"""
    global MODEL_CACHE, frontend, text_frontend, speech_tokenizer, llm, flow
    
    logger.info("Clearing model cache and freeing GPU memory...")
    
    # 清除全局变量
    frontend = None
    text_frontend = None
    speech_tokenizer = None
    llm = None
    flow = None
    
    # 清除缓存
    if MODEL_CACHE["components"]:
        del MODEL_CACHE["components"]
    
    MODEL_CACHE["loaded"] = False
    MODEL_CACHE["sample_rate"] = None
    MODEL_CACHE["components"] = None
    
    # 强制垃圾回收
    gc.collect()
    
    # 清除CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")
    
    logger.info("Model cache cleared successfully")
    return {"status": "success", "message": "Model cache cleared successfully"}

def get_models(use_phoneme=False, sample_rate=24000):
    """懒加载模型。如果sample_rate或use_phoneme改变则重新加载。"""
    global frontend, text_frontend, speech_tokenizer, llm, flow
    
    # 检查是否已加载相同配置
    if (MODEL_CACHE["loaded"] and MODEL_CACHE["sample_rate"] == sample_rate and 
        MODEL_CACHE["use_phoneme"] == use_phoneme):
        logger.info("Using cached models")
        return MODEL_CACHE["components"]
    
    logger.info(f"Loading models with sample_rate={sample_rate}, use_phoneme={use_phoneme}...")
    
    # 清理旧模型
    if MODEL_CACHE["components"]:
        logger.info("Clearing previous model cache...")
        clear_model_cache()
    
    # 加载语音分词器
    speech_tokenizer_path = os.path.join("ckpt", "speech_tokenizer")
    _model, _feature_extractor = yaml_util.load_speech_tokenizer(speech_tokenizer_path)
    speech_tokenizer = SpeechTokenizer(_model, _feature_extractor)

    # 加载前端
    frontend, text_frontend = load_frontends(speech_tokenizer, sample_rate=sample_rate, use_phoneme=use_phoneme)

    # 加载LLM
    llama_path = os.path.join("ckpt", "llm")
    llm = GLMTTS(llama_cfg_path=os.path.join(llama_path, "config.json"), mode="PRETRAIN")
    llm.llama = LlamaForCausalLM.from_pretrained(llama_path, dtype=torch.float32).to(DEVICE)
    llm.llama_embedding = llm.llama.model.embed_tokens

    special_token_ids = get_special_token_ids(frontend.tokenize_fn)
    llm.set_runtime_vars(special_token_ids=special_token_ids)

    # 加载Flow
    flow_ckpt = os.path.join("ckpt", "flow", "flow.pt")
    flow_config = os.path.join("ckpt", "flow", "config.yaml")
    flow = yaml_util.load_flow_model(flow_ckpt, flow_config, DEVICE)

    token2wav = tts_model_util.Token2Wav(flow, sample_rate=sample_rate, device=DEVICE)
    components = (frontend, text_frontend, speech_tokenizer, llm, token2wav)
    
    # 更新缓存
    MODEL_CACHE["components"] = components
    MODEL_CACHE["sample_rate"] = sample_rate
    MODEL_CACHE["use_phoneme"] = use_phoneme
    MODEL_CACHE["loaded"] = True
    
    logger.info("Models loaded successfully!")
    return components

# --- 音色管理 ---
def get_audio_info(file_path):
    """兼容不同版本的torchaudio获取音频信息"""
    try:
        print(f"file_path:{file_path}")
        
        # 方法1: 直接使用 torchaudio.info (最新版本)
        if hasattr(torchaudio, 'info'):
            try:
                return torchaudio.info(file_path)
            except Exception as e:
                logger.warning(f"torchaudio.info failed: {e}")
        
        # 方法2: 检查具体的后端
        if hasattr(torchaudio, 'backend'):
            # 检查可用的后端
            backends = dir(torchaudio.backend)
            if 'sox_backend' in backends:
                try:
                    return torchaudio.backend.sox_backend.info(file_path)
                except Exception as e:
                    logger.warning(f"sox_backend failed: {e}")
            if 'sox_io_backend' in backends:
                try:
                    return torchaudio.backend.sox_io_backend.info(file_path)
                except Exception as e:
                    logger.warning(f"sox_io_backend failed: {e}")
        
        # 方法3: 使用 soundfile 作为备选
        try:
            import soundfile as sf
            info = sf.info(file_path)
            class AudioInfo:
                def __init__(self):
                    self.sample_rate = info.samplerate
                    self.num_channels = info.channels
                    self.num_frames = int(info.frames)
                    self.bits_per_sample = info.subtype.split('_')[-1] if '_' in info.subtype else 16
            
            return AudioInfo()
        except ImportError:
            logger.warning("soundfile not available")
        except Exception as e:
            logger.warning(f"soundfile failed: {e}")
        
        # 方法4: 最终回退
        logger.error(f"无法获取音频信息，使用默认值")
        class Info:
            def __init__(self):
                self.sample_rate = 24000
                self.num_channels = 1
                self.num_frames = 24000
                self.bits_per_sample = 16
        
        return Info()
        
    except Exception as e:
        logger.error(f"获取音频信息时出错: {e}")
        class Info:
            def __init__(self):
                self.sample_rate = 24000
                self.num_channels = 1
                self.num_frames = 24000
                self.bits_per_sample = 16
        
        return Info()

def get_available_voices():
    """获取所有可用的内置参考音色"""
    # 定义支持的音频文件扩展名列表
    # 你可以根据需要添加更多格式，如 '.aac', '.wma', '.opus' 等
    SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    def scan_directory_for_voices(directory):
        """内部辅助函数：扫描目录并返回音色列表"""
        found_voices = []
        # 检查目录是否存在
        if not os.path.exists(directory):
            return found_voices
            
        for file_name in os.listdir(directory):
            # 检查文件扩展名是否在支持列表中（不区分大小写）
            if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                voice_id = os.path.splitext(file_name)[0]
                file_path = os.path.join(directory, file_name)
                
                try:
                    # 注意：确保你的 get_audio_info 函数支持以下扩展名对应的解码库
                    audio_info = get_audio_info(file_path)
                    duration = audio_info.num_frames / audio_info.sample_rate
                    found_voices.append({
                        "id": voice_id,
                        "name": voice_id.replace("_", " ").title(),
                        "file_path": file_path,
                        "sample_rate": audio_info.sample_rate,
                        "num_channels": audio_info.num_channels,
                        "duration": duration
                    })
                except Exception as e:
                    logger.warning(f"Error reading voice file {file_name}: {e}")
        return found_voices

    # 确保参考音色目录存在
    if not os.path.exists(REFERENCE_VOICES_DIR):
        os.makedirs(REFERENCE_VOICES_DIR, exist_ok=True)
        logger.info(f"Reference voices directory created at {REFERENCE_VOICES_DIR}")
    
    # 第一次扫描：查找现有音色
    voices = scan_directory_for_voices(REFERENCE_VOICES_DIR)
    
    # 如果没有找到任何内置音色，创建一个默认音色
    if not voices:
        logger.warning("No reference voices found. Creating default voice.")
        create_default_voice()
        
        # 重新获取音色列表（复用扫描逻辑）
        voices = scan_directory_for_voices(REFERENCE_VOICES_DIR)
    
    return voices

def create_default_voice():
    """创建默认参考音色"""
    default_voice_path = os.path.join(REFERENCE_VOICES_DIR, "default.wav")
    
    # 检查是否已经存在
    if os.path.exists(default_voice_path):
        logger.info(f"Default voice already exists at {default_voice_path}")
        return
    
    # 创建一个简单的正弦波
    sample_rate = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    
    # 保存为WAV文件
    try:
        torchaudio.save(default_voice_path, audio, sample_rate)
        logger.info(f"Default reference voice created at {default_voice_path}")
    except Exception as e:
        logger.error(f"Failed to create default voice: {e}")

def get_voice_embeddings(frontend, audio_path, sample_rate):
    """获取音色嵌入和特征"""
    try:
        speech_feat = frontend._extract_speech_feat(audio_path, sample_rate=sample_rate)
        embedding = frontend._extract_spk_embedding(audio_path)
        return speech_feat, embedding
    except Exception as e:
        logger.error(f"Error extracting voice embeddings from {audio_path}: {e}")
        raise

# --- WAV文件头生成器（修复版本）---
def create_streaming_wav_header(sample_rate, num_channels=1, bits_per_sample=16):
    """
    创建用于流式传输的WAV文件头
    
    对于流式传输，我们使用0xFFFFFFFF（4294967295）作为占位符，
    表示文件大小未知。这是WAV格式的标准做法。
    
    Args:
        sample_rate: 采样率（24000或32000）
        num_channels: 通道数（默认1，单声道）
        bits_per_sample: 每样本位数（默认16）
    
    Returns:
        bytes: WAV文件头
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    
    header = bytearray()
    
    # RIFF header (12 bytes)
    header.extend(b'RIFF')
    # ChunkSize: 使用0xFFFFFFFF表示未知大小（标准流式WAV做法）
    header.extend(struct.pack('<I', 0xFFFFFFFF))  # 4294967295
    header.extend(b'WAVE')
    
    # fmt subchunk (24 bytes)
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Subchunk1Size (16 for PCM)
    header.extend(struct.pack('<H', 1))   # AudioFormat (1 = PCM)
    header.extend(struct.pack('<H', num_channels))  # NumChannels
    header.extend(struct.pack('<I', sample_rate))   # SampleRate
    header.extend(struct.pack('<I', byte_rate))     # ByteRate
    header.extend(struct.pack('<H', block_align))   # BlockAlign
    header.extend(struct.pack('<H', bits_per_sample))  # BitsPerSample
    
    # data subchunk header (8 bytes)
    header.extend(b'data')
    # Subchunk2Size: 同样使用0xFFFFFFFF表示未知大小
    header.extend(struct.pack('<I', 0xFFFFFFFF))
    
    return bytes(header)

def create_complete_wav_header(sample_rate, num_channels, bits_per_sample, data_size):
    """
    创建完整WAV文件的文件头（用于非流式传输）
    
    Args:
        sample_rate: 采样率
        num_channels: 通道数
        bits_per_sample: 每样本位数
        data_size: 音频数据大小（字节）
    
    Returns:
        bytes: 完整的WAV文件头
    """
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    
    header = bytearray()
    
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', data_size + 36))  # 36 = 4 + 24 + 8 (fmt + data headers)
    header.extend(b'WAVE')
    
    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Subchunk1Size
    header.extend(struct.pack('<H', 1))   # AudioFormat (PCM)
    header.extend(struct.pack('<H', num_channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', byte_rate))
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', bits_per_sample))
    
    # data subchunk
    header.extend(b'data')
    header.extend(struct.pack('<I', data_size))
    
    return bytes(header)

# --- 资源清理函数 ---
def cleanup_temp_dir(dir_path: str):
    """安全清理临时目录"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"临时目录已清理: {dir_path}")
            if dir_path in temp_dirs:
                temp_dirs.remove(dir_path)
    except Exception as e:
        logger.error(f"清理临时目录失败 {dir_path}: {e}")

def cleanup_temp_file(file_path: str):
    """安全清理临时文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"临时文件已清理: {file_path}")
    except Exception as e:
        logger.error(f"清理临时文件失败 {file_path}: {e}")

# --- FastAPI 应用 ---
class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本内容")
    uttid: Optional[str] = Field("default", description="语音ID，用于标识生成的语音")
    use_cache: Optional[bool] = Field(True, description="是否使用KV缓存，长文本生成时推荐开启")
    use_phoneme: Optional[bool] = Field(False, description="是否使用音素处理")
    seed: Optional[int] = Field(42, description="随机种子")
    sample_method: Optional[str] = Field("ras", description="采样方法")
    sample_rate: Optional[int] = Field(24000, description="采样率，支持24000或32000")

    @validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v not in [24000, 32000]:
            raise ValueError("采样率必须是24000或32000")
        return v

class TTSResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    text_info: Optional[Dict[str, Any]] = None

class VoiceListResponse(BaseModel):
    voices: List[Dict[str, Any]]
    success: bool
    message: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting up FastAPI application...")
    
    # 确保参考音色目录存在
    if not os.path.exists(REFERENCE_VOICES_DIR):
        os.makedirs(REFERENCE_VOICES_DIR, exist_ok=True)
        logger.info(f"Reference voices directory created at {REFERENCE_VOICES_DIR}")
    
    # 预加载默认模型
    try:
        logger.info("Pre-loading default models (sample_rate=24000, use_phoneme=False)...")
        get_models(use_phoneme=False, sample_rate=24000)
        logger.info("Default models pre-loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to pre-load default models: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # 关闭时清理资源
    logger.info("Shutting down FastAPI application...")
    
    # 1. 清理模型缓存
    clear_model_cache()
    
    # 2. 清理所有创建的临时目录
    global temp_dirs
    for temp_dir in list(temp_dirs):  # 使用list避免修改迭代中的列表
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Temporary directory cleaned up: {temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning temporary directory {temp_dir}: {e}")
    
    temp_dirs = []
    logger.info("All cleanup completed")

app = FastAPI(
    title="GLM-TTS API",
    description="FastAPI wrapper for GLM-TTS text-to-speech synthesis with streaming support",
    version="1.1.0",
    lifespan=lifespan
)

fastapi_cdn_host.patch_docs(app)

@app.get("/")
async def root():
    """根端点，返回API信息"""
    return {
        "name": "GLM-TTS API",
        "version": "1.1.0",
        "description": "Text-to-Speech synthesis using GLM-TTS model with streaming support",
        "supported_sample_rates": [24000, 32000],
        "endpoints": {
            "voices": "/voices - 获取可用的内置参考音色列表",
            "tts/generate": "/tts/generate - 生成语音并返回文件",
            "tts/stream": "/tts/stream - 生成语音并流式返回（真正的边生成边播放）",
            "memory/clear": "/memory/clear - 清除模型缓存和释放GPU内存",
            "health": "/health - 检查API健康状态"
        }
    }

@app.get("/voices", response_model=VoiceListResponse)
async def list_voices():
    """获取所有可用的内置参考音色列表"""
    try:
        voices = get_available_voices()
        return VoiceListResponse(
            voices=voices,
            success=True,
            message=f"找到 {len(voices)} 个可用参考音色"
        )
    except Exception as e:
        logger.error(f"获取可用音色列表失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "gpu_memory": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else "N/A",
        "models_loaded": MODEL_CACHE["loaded"],
        "current_sample_rate": MODEL_CACHE["sample_rate"],
        "use_phoneme": MODEL_CACHE["use_phoneme"],
        "available_voices_count": len(get_available_voices()),
        "timestamp": time.time()
    }

@app.post("/memory/clear")
async def clear_memory_endpoint():
    """清除模型缓存和GPU内存"""
    result = clear_model_cache()
    return JSONResponse(content=result)

def prepare_voice_context(
    frontend,
    voice_type: VoiceType,
    builtin_voice_id: Optional[str],
    prompt_text: str,
    uploaded_audio_path: Optional[str],
    sample_rate: int,
    use_phoneme: bool
):
    """准备音色上下文，统一处理内置和上传音色"""
    # 处理音色选择
    if voice_type == VoiceType.BUILTIN:
        if not builtin_voice_id:
            raise ValueError("使用内置音色时，builtin_voice_id参数必需")
        
        # 获取内置音色
        available_voices = get_available_voices()
        voice_dict = {voice["id"]: voice for voice in available_voices}
        
        if builtin_voice_id not in voice_dict:
            raise ValueError(f"内置音色ID '{builtin_voice_id}' 不存在。可用音色: {list(voice_dict.keys())}")
        
        prompt_audio_path = voice_dict[builtin_voice_id]["file_path"]
    elif voice_type == VoiceType.UPLOAD:
        if not uploaded_audio_path or not os.path.exists(uploaded_audio_path):
            raise ValueError("使用上传音色时，必须提供有效的音频文件")
        prompt_audio_path = uploaded_audio_path
    else:
        raise ValueError("不支持的音色类型")
    
    # 处理文本
    text_frontend = TextFrontEnd(use_phoneme)
    prompt_text = text_frontend.text_normalize(prompt_text)
    
    # 提取提示文本token
    prompt_text_token = frontend._extract_text_token(prompt_text + " ")
    
    # 提取提示语音token
    prompt_speech_token = frontend._extract_speech_token([prompt_audio_path])
    
    # 获取音色嵌入和特征
    speech_feat, embedding = get_voice_embeddings(frontend, prompt_audio_path, sample_rate)
    
    cache_speech_token = [prompt_speech_token.squeeze().tolist()]
    flow_prompt_token = torch.tensor(cache_speech_token, dtype=torch.int32).to(DEVICE)
    
    # 初始化缓存
    cache = {
        "cache_text": [prompt_text],
        "cache_text_token": [prompt_text_token],
        "cache_speech_token": cache_speech_token,
        "use_cache": True,
    }
    
    return {
        "prompt_audio_path": prompt_audio_path,
        "prompt_text": prompt_text,
        "speech_feat": speech_feat,
        "embedding": embedding,
        "flow_prompt_token": flow_prompt_token,
        "cache": cache,
        "voice_type": voice_type,
        "builtin_voice_id": builtin_voice_id
    }

async def tts_stream_generator(
    text: str,
    uttid: str,
    voice_type: VoiceType,
    builtin_voice_id: Optional[str],
    prompt_text: str,
    sample_rate: int,
    use_phoneme: bool,
    seed: int,
    sample_method: str,
    chunk_size: int = 1024,
    uploaded_audio_path: Optional[str] = None
) -> AsyncGenerator[bytes, None]:
    """流式生成TTS音频的生成器"""
    try:
        logger.info(f"开始流式TTS生成，uttid={uttid}, voice_type={voice_type}")
        
        # 加载模型
        frontend, text_frontend, speech_tokenizer, llm, flow = get_models(
            use_phoneme=use_phoneme,
            sample_rate=sample_rate
        )
        
        # 准备音色上下文
        voice_context = prepare_voice_context(
            frontend,
            voice_type,
            builtin_voice_id,
            prompt_text,
            uploaded_audio_path,
            sample_rate,
            use_phoneme
        )
        
        # 创建流式WAV文件头（修复了整数溢出问题）
        wav_header = create_streaming_wav_header(sample_rate=sample_rate)
        yield wav_header
        
        logger.info("WAV文件头已发送，开始流式音频生成...")
        
        # 生成流式音频
        segment_count = 0
        total_audio_bytes = 0
        
        async for chunk_info in generate_streaming(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=[uttid, text],
            cache=voice_context["cache"],
            device=DEVICE,
            embedding=voice_context["embedding"],
            seed=seed,
            sample_method=sample_method,
            flow_prompt_token=voice_context["flow_prompt_token"],
            speech_feat=voice_context["speech_feat"],
            use_phoneme=use_phoneme,
            chunk_size=chunk_size
        ):
            if chunk_info["audio_chunk"]:
                yield chunk_info["audio_chunk"]
                segment_count += 1
                total_audio_bytes += len(chunk_info["audio_chunk"])
                logger.debug(f"已发送音频块: {len(chunk_info['audio_chunk'])} 字节, 累计: {total_audio_bytes} 字节")
            
            if chunk_info["is_last_segment"]:
                logger.info(f"流式生成完成，共 {segment_count} 个音频块，总大小 {total_audio_bytes} 字节")
                if "text_info" in chunk_info:
                    logger.debug(f"文本处理信息: {json.dumps(chunk_info['text_info'], ensure_ascii=False)}")
        
    except Exception as e:
        logger.error(f"流式TTS生成失败: {e}")
        import traceback
        traceback.print_exc()
        # 在出现错误时，发送一个空的音频块作为结束信号
        yield b""
        raise

@app.post("/tts/generate", response_class=FileResponse)
async def tts_generate(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="要合成的文本内容"),
    uttid: str = Form("default", description="语音ID，用于标识生成的语音"),
    voice_type: VoiceType = Form(VoiceType.BUILTIN, description="音色类型：builtin(内置)或upload(上传)"),
    builtin_voice_id: Optional[str] = Form(None, description="内置音色ID，当type为builtin时必需"),
    prompt_text: str = Form("你好，我是语音合成助手。", description="参考音频对应的文本"),
    prompt_audio: Optional[UploadFile] = File(None, description="上传的参考音频文件"),
    use_cache: bool = Form(True, description="是否使用KV缓存，长文本生成时推荐开启"),
    use_phoneme: bool = Form(False, description="是否使用音素处理"),
    seed: int = Form(42, description="随机种子"),
    sample_method: str = Form("ras", description="采样方法"),
    sample_rate: int = Form(24000, description="采样率，支持24000或32000")
):
    """
    生成语音并返回文件
    
    此接口支持两种音色选择方式：
    1. 使用内置音色：设置 voice_type=builtin 并提供 builtin_voice_id
    2. 上传自定义音色：设置 voice_type=upload 并上传 prompt_audio 文件
    
    音频将作为文件附件返回。
    """
    # 验证采样率
    if sample_rate not in [24000, 32000]:
        raise HTTPException(status_code=400, detail="不支持的采样率。请使用24000或32000。")
    
    # 验证音色选择
    if voice_type == VoiceType.BUILTIN and not builtin_voice_id:
        raise HTTPException(status_code=400, detail="使用内置音色时，builtin_voice_id参数必需")
    
    if voice_type == VoiceType.UPLOAD and not prompt_audio:
        raise HTTPException(status_code=400, detail="使用上传音色时，必须提供prompt_audio文件")
    
    # 创建临时目录管理
    temp_cleanup_dirs = []
    temp_cleanup_files = []
    
    try:
        # 处理上传的音频文件
        uploaded_audio_path = None
        upload_temp_dir = None
        if voice_type == VoiceType.UPLOAD and prompt_audio:
            upload_temp_dir = tempfile.mkdtemp()
            temp_dirs.append(upload_temp_dir)
            temp_cleanup_dirs.append(upload_temp_dir)
            
            uploaded_audio_path = os.path.join(upload_temp_dir, f"prompt_{uttid}.wav")
            with open(uploaded_audio_path, "wb") as f:
                content = await prompt_audio.read()
                f.write(content)
            temp_cleanup_files.append(uploaded_audio_path)
        
        # 创建临时目录保存输出
        output_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(output_temp_dir)
        temp_cleanup_dirs.append(output_temp_dir)
        
        wav_path = os.path.join(output_temp_dir, f"{uttid}.wav")
        temp_cleanup_files.append(wav_path)
        
        # 加载模型
        frontend, text_frontend, speech_tokenizer, llm, flow = get_models(
            use_phoneme=use_phoneme,
            sample_rate=sample_rate
        )
        
        # 准备音色上下文
        voice_context = prepare_voice_context(
            frontend,
            voice_type,
            builtin_voice_id,
            prompt_text,
            uploaded_audio_path,
            sample_rate,
            use_phoneme
        )
        
        # 生成完整音频
        outputs = []
        text_info = None
        
        async for chunk_info in generate_streaming(
            frontend=frontend,
            text_frontend=text_frontend,
            llm=llm,
            flow=flow,
            text_info=[uttid, text],
            cache=voice_context["cache"],
            device=DEVICE,
            embedding=voice_context["embedding"],
            seed=seed,
            sample_method=sample_method,
            flow_prompt_token=voice_context["flow_prompt_token"],
            speech_feat=voice_context["speech_feat"],
            use_phoneme=use_phoneme,
            chunk_size=8192  # 较大的块大小用于完整文件生成
        ):
            if chunk_info["audio_chunk"]:
                outputs.append(chunk_info["audio_chunk"])
            if chunk_info["is_last_segment"] and "text_info" in chunk_info:
                text_info = chunk_info["text_info"]
        
        # 合并所有音频块
        complete_audio_data = b''.join(outputs)
        
        # 创建完整WAV文件（包含正确的文件头和大小信息）
        wav_header = create_complete_wav_header(
            sample_rate=sample_rate,
            num_channels=1,
            bits_per_sample=16,
            data_size=len(complete_audio_data)
        )
        
        # 保存完整WAV文件
        with open(wav_path, 'wb') as f:
            f.write(wav_header)
            f.write(complete_audio_data)
        
        # 从清理列表中移除这个文件，因为它需要在响应后保留
        if wav_path in temp_cleanup_files:
            temp_cleanup_files.remove(wav_path)
        
        logger.info(f"完整音频文件已保存到: {wav_path}, 大小: {len(complete_audio_data)} 字节")
        
        # 准备响应头
        headers = {
            "X-Success": "true",
            "X-Message": "Audio generated successfully",
            "X-Sample-Rate": str(sample_rate),
            "X-Uttid": uttid,
            "X-Voice-Type": voice_type.value
        }
        
        if voice_type == VoiceType.BUILTIN:
            headers["X-Builtin-Voice-Id"] = builtin_voice_id
        
        # 使用后台任务清理临时文件和目录
        for file_path in temp_cleanup_files:
            background_tasks.add_task(cleanup_temp_file, file_path)
        
        for dir_path in temp_cleanup_dirs:
            if dir_path != output_temp_dir:  # 不要立即清理输出目录
                background_tasks.add_task(cleanup_temp_dir, dir_path)
        
        # 重要：在后台任务中安排清理输出目录，但在文件响应完成后
        background_tasks.add_task(cleanup_temp_file, wav_path)
        background_tasks.add_task(cleanup_temp_dir, output_temp_dir)
        
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename=f"{uttid}.wav",
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"处理TTS请求失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 发生异常时，安排清理所有临时资源
        for file_path in temp_cleanup_files:
            background_tasks.add_task(cleanup_temp_file, file_path)
        
        for dir_path in temp_cleanup_dirs:
            background_tasks.add_task(cleanup_temp_dir, dir_path)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/stream")
async def tts_stream(
    request: Request,
    text: str = Form(..., description="要合成的文本内容"),
    uttid: str = Form("default", description="语音ID，用于标识生成的语音"),
    voice_type: VoiceType = Form(VoiceType.BUILTIN, description="音色类型：builtin(内置)或upload(上传)"),
    builtin_voice_id: Optional[str] = Form(None, description="内置音色ID，当type为builtin时必需"),
    prompt_text: str = Form("你好，我是语音合成助手。", description="参考音频对应的文本"),
    prompt_audio: Optional[UploadFile] = File(None, description="上传的参考音频文件"),
    use_phoneme: bool = Form(False, description="是否使用音素处理"),
    seed: int = Form(42, description="随机种子"),
    sample_method: str = Form("ras", description="采样方法"),
    sample_rate: int = Form(24000, description="采样率，支持24000或32000"),
    chunk_size: int = Form(1024, description="音频块大小（字节），范围128-8192")
):
    """
    生成语音并流式返回（真正的边生成边播放）
    
    此接口支持两种音色选择方式：
    1. 使用内置音色：设置 voice_type=builtin 并提供 builtin_voice_id
    2. 上传自定义音色：设置 voice_type=upload 并上传 prompt_audio 文件
    
    音频将以流式方式逐步返回，支持边生成边播放。
    
    参数按逻辑分组：
    - 基础文本信息：text, uttid
    - 音色配置（放在一起）：
        * voice_type: 音色类型
        * builtin_voice_id: 内置音色ID（当voice_type=builtin时必需）
        * prompt_text: 参考音频对应的文本
        * prompt_audio: 上传的参考音频文件（当voice_type=upload时必需）
    - 生成参数：use_phoneme, seed, sample_method, sample_rate
    - 流式控制：chunk_size
    
    注意：chunk_size建议值：
    - 网络延迟敏感：128-512
    - 平衡延迟和吞吐：1024-4096  
    - 最大吞吐：8192
    """
    # 验证采样率
    if sample_rate not in [24000, 32000]:
        raise HTTPException(status_code=400, detail="不支持的采样率。请使用24000或32000。")
    
    # 验证音色选择
    if voice_type == VoiceType.BUILTIN and not builtin_voice_id:
        raise HTTPException(status_code=400, detail="使用内置音色时，builtin_voice_id参数必需")
    
    if voice_type == VoiceType.UPLOAD and not prompt_audio:
        raise HTTPException(status_code=400, detail="使用上传音色时，必须提供prompt_audio文件")
    
    # 验证chunk_size
    if not (128 <= chunk_size <= 8192):
        raise HTTPException(status_code=400, detail="chunk_size必须在128-8192范围内")
    
    # 处理上传的音频文件
    uploaded_audio_path = None
    temp_dir = None
    if voice_type == VoiceType.UPLOAD and prompt_audio:
        temp_dir = tempfile.mkdtemp()
        global temp_dirs
        temp_dirs.append(temp_dir)
        
        uploaded_audio_path = os.path.join(temp_dir, f"prompt_{uttid}.wav")
        try:
            content = await prompt_audio.read()
            with open(uploaded_audio_path, "wb") as f:
                f.write(content)
            logger.info(f"上传的参考音频已保存到: {uploaded_audio_path}, 大小: {len(content)} 字节")
        except Exception as e:
            logger.error(f"保存上传音频失败: {e}")
            raise HTTPException(status_code=500, detail=f"保存上传音频失败: {str(e)}")
    
    # 创建流式响应
    async def audio_stream():
        try:
            async for chunk in tts_stream_generator(
                text=text,
                uttid=uttid,
                voice_type=voice_type,
                builtin_voice_id=builtin_voice_id,
                prompt_text=prompt_text,
                sample_rate=sample_rate,
                use_phoneme=use_phoneme,
                seed=seed,
                sample_method=sample_method,
                chunk_size=chunk_size,
                uploaded_audio_path=uploaded_audio_path
            ):
                yield chunk
        except Exception as e:
            logger.error(f"流式生成过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 发送错误信息作为音频数据（静音片段）
            error_audio = b'\x00' * 44100  # 1秒静音
            yield error_audio
            raise
        finally:
            # 清理临时文件
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    if temp_dir in temp_dirs:
                        temp_dirs.remove(temp_dir)
                    logger.info(f"上传音频临时目录已清理: {temp_dir}")
                except Exception as e:
                    logger.error(f"清理临时目录失败 {temp_dir}: {e}")
    
    # 准备响应头
    headers = {
        "X-Uttid": uttid,
        "X-Voice-Type": voice_type.value,
        "X-Sample-Rate": str(sample_rate),
        "X-Chunk-Size": str(chunk_size),
        "X-Content-Type-Options": "nosniff",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive"
    }
    
    if voice_type == VoiceType.BUILTIN:
        headers["X-Builtin-Voice-Id"] = builtin_voice_id
    
    # 客户端信息日志
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"流式TTS请求 - 客户端: {client_ip}, UA: {user_agent}")
    logger.info(f"流式TTS参数 - uttid:{uttid}, voice_type:{voice_type}, sample_rate:{sample_rate}, chunk_size:{chunk_size}")
    logger.info(f"文本长度: {len(text)} 字符, 预览: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    
    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav",
        headers=headers,
        status_code=200
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "glm_tts_fastapi:app",
        host="0.0.0.0",
        port=8809,
        reload=False,
        workers=1
    )