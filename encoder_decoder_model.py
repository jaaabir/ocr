import torch 
from transformers import (
    VisionEncoderDecoderModel, AutoImageProcessor, T5ForConditionalGeneration, 
    AutoModel, AutoModelForCausalLM, AutoTokenizer, 
    PerceptionLMForConditionalGeneration, PerceptionLMProcessor, MBartForCausalLM,
    AutoProcessor, AutoModelForImageTextToText, PreTrainedTokenizerFast )
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, trainers

from accelerate import load_checkpoint_and_dispatch
import os 
from peft import LoraConfig, get_peft_model, TaskType

import torch.nn.functional as F

def resize_position_embeddings(model, old_size=224, new_size=768, patch_size=16):
    """
    Resize position embeddings for DiT model to handle larger images.
    
    Args:
        model: The DiT model
        old_size: Original image size (224)
        new_size: Target image size (768)
        patch_size: Patch size (16)
    """
    # Calculate number of patches
    old_num_patches = (old_size // patch_size) ** 2  # 196
    new_num_patches = (new_size // patch_size) ** 2  # 2304
    
    print(f"Old patches: {old_num_patches}, New patches: {new_num_patches}")
    
    # Get current position embeddings [1, 197, 1024] (196 patches + 1 CLS token)
    old_pos_embed = model.embeddings.position_embeddings.data
    
    # Separate CLS token embedding and patch embeddings
    cls_pos_embed = old_pos_embed[:, 0:1, :]  # [1, 1, 1024]
    patch_pos_embed = old_pos_embed[:, 1:, :]  # [1, 196, 1024]
    
    # Reshape patch embeddings to 2D grid
    old_grid_size = int(old_num_patches ** 0.5)  # 14
    new_grid_size = int(new_num_patches ** 0.5)  # 48
    embed_dim = patch_pos_embed.shape[-1]  # 1024
    
    # Reshape to [1, 1024, 14, 14]
    patch_pos_embed = patch_pos_embed.reshape(1, old_grid_size, old_grid_size, embed_dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    
    # Interpolate to new size [1, 1024, 48, 48]
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        size=(new_grid_size, new_grid_size),
        mode='bicubic',
        align_corners=False
    )
    
    # Reshape back to [1, 2304, 1024]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
    patch_pos_embed = patch_pos_embed.reshape(1, new_num_patches, embed_dim)
    
    # Concatenate CLS token and patch embeddings
    new_pos_embed = torch.cat([cls_pos_embed, patch_pos_embed], dim=1)  # [1, 2305, 1024]
    
    print(f"Original position embeddings shape: {old_pos_embed.shape}")
    print(f"New position embeddings shape: {new_pos_embed.shape}")
    
    # Create new embedding layer with correct size
    new_embeddings = torch.nn.Parameter(new_pos_embed)
    model.embeddings.position_embeddings = new_embeddings
    
    return model


def load_pretrained_iprocessor_tokenizer(pre_trained_ckpt_path):
    processor = AutoImageProcessor.from_pretrained(pre_trained_ckpt_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_ckpt_path, use_fast=True)
    return processor, tokenizer


def load_pretrained_enc_dec_model(pre_trained_ckpt_path, 
                                  base_encoder_model="microsoft/dit-base-finetuned-rvlcdip", 
                                  base_decoder_model="facebook/bart-base",
                                  pre_trained_model_weights_tied=True,
                                  lora_applied=True,
                                  new_tokens = [],
                                  **pre_trained_lora_configs_kwargs,
                                  ):
    
    if base_encoder_model and base_decoder_model:
        enc = AutoModel.from_pretrained(base_encoder_model)
        dec = AutoModelForCausalLM.from_pretrained(base_decoder_model)
        if lora_applied:
            dec = add_lora_to_decoder(dec, enc_dec_model=False, **pre_trained_lora_configs_kwargs)
        m = VisionEncoderDecoderModel(encoder=enc, decoder=dec)  
        sft = 'model.safetensors'
        pmb = 'pytorch_model.bin'
        if sft in os.listdir(pre_trained_ckpt_path):
            st_path = os.path.join(pre_trained_ckpt_path, sft)
            st = load_file(st_path)
        else:
            st_path = os.path.join(pre_trained_ckpt_path, pmb)
            st = torch.load(st_path, map_location='cpu')
        
        if "decoder.base_model.model.lm_head.weight" not in st.keys() or pre_trained_model_weights_tied:
            st["decoder.base_model.model.lm_head.weight"] = st['decoder.base_model.model.model.decoder.embed_tokens.weight']
        m.load_state_dict(st, strict=True)
    else:
        m = VisionEncoderDecoderModel.from_pretrained(pre_trained_ckpt_path)

    files = os.listdir(pre_trained_ckpt_path)
    if 'tokenizer_config.json' in files and 'tokenizer.json' in files:
        print('loading pre-trained tokenizer')
        text_tokenizer = AutoTokenizer.from_pretrained(pre_trained_ckpt_path, use_fast=True)
    else:
        text_tokenizer = AutoTokenizer.from_pretrained(base_decoder_model, use_fast=True)

    if len(new_tokens) > 0:
        text_tokenizer.add_tokens(new_tokens)

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    m.config.decoder_start_token_id = text_tokenizer.bos_token_id
    m.config.pad_token_id = text_tokenizer.pad_token_id
    m.config.eos_token_id = text_tokenizer.eos_token_id
    tokenizer_total_leng = len(text_tokenizer)
    
    if hasattr(m.config, 'vocab_size') and m.config.vocab_size != tokenizer_total_leng:
        print('config vocab size is not equal to tokenizer leng. changing the configs ...')
        m.config.vocab_size = tokenizer_total_leng
    if m.config.decoder.vocab_size != tokenizer_total_leng:
        print('decoder config vocab size is not equal to tokenizer leng. changing the configs ...')
        m.config.decoder.vocab_size = tokenizer_total_leng
        m.decoder.resize_token_embeddings(tokenizer_total_leng)

    m.generation_config.bos_token_id = text_tokenizer.bos_token_id
    m.generation_config.eos_token_id = text_tokenizer.eos_token_id
    m.generation_config.pad_token_id = text_tokenizer.pad_token_id
    m.generation_config.decoder_start_token_id = text_tokenizer.bos_token_id

    print('Loaded the pre-trained model successfully...')
    return m 

def add_lora_to_decoder(encoder_decoder_model, enc_dec_model=True, r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], bias="none", modules_to_save=None, pre_trained_ckpt_path = None):
    """
    Wrap the decoder inside VisionEncoderDecoderModel with LoRA adapters.
    Fixed version that properly handles the decoder architecture.
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  
        r=r,                           
        lora_alpha=alpha,             
        lora_dropout=dropout,
        target_modules=target_modules, 
        bias=bias,
        modules_to_save=modules_to_save,          
    )
    
    if enc_dec_model:
        encoder_decoder_model.decoder = get_peft_model(
        encoder_decoder_model.decoder, lora_config
        )
        print("LoRA applied to decoder")
    else:
        encoder_decoder_model = get_peft_model(
        encoder_decoder_model, lora_config
        )
        print("LoRA applied to the model")
    return encoder_decoder_model


def add_lora_to_encoder_decoder(encoder_decoder_model, r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], bias="none"):
    """
    Apply LoRA to the entire encoder-decoder model instead of just the decoder.
    This might work better for your use case.
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
    )
    
    # Apply LoRA to the entire model
    encoder_decoder_model = get_peft_model(encoder_decoder_model, lora_config)
    
    print("LoRA applied to encoder-decoder model:")
    encoder_decoder_model.print_trainable_parameters()
    
    return encoder_decoder_model

def print_model_layer_sizes(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} parameters | {param.shape}")

def init_dit_dbart_models(encoder_model="microsoft/dit-large-finetuned-rvlcdip", 
                          decoder_model="naver-clova-ix/donut-base", 
                          load_model=True):
    
    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True)
    text_tokenizer.add_tokens(['Ã', 'Ê', 'Â'])

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    if load_model:
        dit = AutoModel.from_pretrained(encoder_model)
        dbart = VisionEncoderDecoderModel.from_pretrained(decoder_model).decoder
        dbart.resize_token_embeddings(len(text_tokenizer))
        dbart.config.vocab_size = len(text_tokenizer)
        model = VisionEncoderDecoderModel(encoder=dit, decoder=dbart)
        model.config.decoder_start_token_id = text_tokenizer.bos_token_id
        model.config.pad_token_id = text_tokenizer.pad_token_id
        model.config.eos_token_id = text_tokenizer.eos_token_id
        model.config.vocab_size = len(text_tokenizer)
        
        model.generation_config.bos_token_id = text_tokenizer.bos_token_id
        model.generation_config.eos_token_id = text_tokenizer.eos_token_id
        model.generation_config.pad_token_id = text_tokenizer.pad_token_id
        model.generation_config.decoder_start_token_id = text_tokenizer.bos_token_id

        return image_processor, text_tokenizer, model
    return image_processor, text_tokenizer


def init_dit_mbert_models_fixed(
    encoder_model="microsoft/dit-base-finetuned-rvlcdip",
    decoder_model="jhu-clsp/ettin-decoder-32m",
):
    """
    Fixed version of DiT-Modern Bert Decoder initialization with proper cross-attention setup.
    """

    dit = AutoModel.from_pretrained(encoder_model)
    mbert = AutoModelForCausalLM.from_pretrained(decoder_model)
    encoder_decoder_model = VisionEncoderDecoderModel(encoder = dit, decoder= mbert)

    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True)

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    encoder_decoder_model.config.decoder_start_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.pad_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
    
    encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
    encoder_decoder_model.decoder.resize_token_embeddings(len(text_tokenizer))

    encoder_decoder_model.generation_config.bos_token_id = text_tokenizer.bos_token_id
    encoder_decoder_model.generation_config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.generation_config.pad_token_id = text_tokenizer.pad_token_id

    encoder_decoder_model.config.tie_word_embeddings = True

    return image_processor, text_tokenizer, encoder_decoder_model


def init_dit_bart_models_fixed(
    encoder_model="microsoft/dit-base-finetuned-rvlcdip",
    decoder_model="facebook/bart-base",
    load_model=True
):
    """
    Fixed version of DiT-BART initialization with proper cross-attention setup.
    """
    
    if load_model:   
        encoder_decoder_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=encoder_model,
            decoder_pretrained_model_name_or_path=decoder_model,
            decoder_forced_bos_token_id=None
            )
    

    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True)

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    if load_model:
        encoder_decoder_model.config.decoder_start_token_id = text_tokenizer.bos_token_id
        encoder_decoder_model.config.pad_token_id = text_tokenizer.pad_token_id
        encoder_decoder_model.config.eos_token_id = text_tokenizer.eos_token_id
        encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
        encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
        encoder_decoder_model.decoder.resize_token_embeddings(len(text_tokenizer))
        encoder_decoder_model.generation_config.bos_token_id = text_tokenizer.bos_token_id
        encoder_decoder_model.generation_config.eos_token_id = text_tokenizer.eos_token_id
        encoder_decoder_model.generation_config.pad_token_id = text_tokenizer.pad_token_id

        return image_processor, text_tokenizer, encoder_decoder_model
    return image_processor, text_tokenizer


def init_dit_t5_models_fixed(
    encoder_model="microsoft/dit-base-finetuned-rvlcdip",
    decoder_model="google-t5/t5-base",
):
    """
    Fixed version of DiT-T5 initialization with proper cross-attention setup.
    """
    dit = AutoModel.from_pretrained(encoder_model)
    t5 = T5ForConditionalGeneration.from_pretrained(decoder_model)
    encoder_decoder_model = VisionEncoderDecoderModel(
        encoder=dit, decoder=t5
    )
    t5.config.is_encoder_decoder = False
    t5.config.is_decoder = True
    t5.config.add_cross_attention = True

    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True)

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    encoder_decoder_model.config.decoder_start_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.pad_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
    
    encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
    encoder_decoder_model.decoder.resize_token_embeddings(len(text_tokenizer))

    encoder_decoder_model.generation_config.bos_token_id = text_tokenizer.bos_token_id
    encoder_decoder_model.generation_config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.generation_config.pad_token_id = text_tokenizer.pad_token_id

    encoder_decoder_model.config.tie_word_embeddings = True

    return image_processor, text_tokenizer, encoder_decoder_model

def init_dit_qwen_models(encoder_model="microsoft/dit-base-finetuned-rvlcdip",
                         decoder_model="Qwen/Qwen2.5-VL-7B-Instruct", 
                         use_fast_tokenizer=True):
    # Load DiT encoder
    encoder = AutoModel.from_pretrained(encoder_model)

    # Load Qwen decoder
    decoder = AutoModel.from_pretrained(decoder_model)

    # Make sure decoder config allows cross-attention
    decoder.config.is_decoder = True
    decoder.config.is_encoder_decoder = False
    decoder.config.add_cross_attention = True

    # Combine encoder + decoder
    encoder_decoder_model = VisionEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder,
    )

    # Image processor for DiT
    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)

    # Tokenizer for Qwen
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=use_fast_tokenizer)

    # Ensure tokenizer has a pad token
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    return image_processor, text_tokenizer, encoder_decoder_model


def init_perception_lm_model(model_name="facebook/Perception-LM-1B"):
    """
    Initialize Perception-LM for multimodal OCR or vision-to-text tasks.

    Returns:
        processor: For preparing images and text (includes image processor and tokenizer)
        model: Pretrained Perception-LM model
    """

    # Load the combined processor (image processor + tokenizer)
    processor = PerceptionLMProcessor.from_pretrained(model_name)

    # Load the Perception-LM model for conditional generation
    model = PerceptionLMForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.cuda()

    return processor, model

def init_perception_lm_model_efficient(model_name="facebook/Perception-LM-1B"):
    """
    Initialize Perception-LM for multimodal OCR or vision-to-text tasks.
    """

    local_dir = os.path.join('models', model_name.replace('/', "-"), 
                             "models--facebook--Perception-LM-1B", 
                             "snapshots", 
                             "2b1a854663b80d6c8b9a10e4b229be97c7f6be1f")

    model = PerceptionLMForConditionalGeneration.from_pretrained(model_name)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=local_dir,
        device_map="auto",
        offload_folder="offload",
        dtype=torch.float16,
        offload_state_dict=True
    )

    # Tie weights to avoid warnings
    model.tie_weights()

    processor = PerceptionLMProcessor.from_pretrained(model_name)
    return processor, model

def init_deepseek_vl2_model(model_name="deepseek-community/deepseek-vl-1.3b-base"):
    """
    Initialize DeepSeek-VL 2 for multimodal OCR or image-to-text tasks.

    Returns:
        processor: For preparing images and text (handles vision+language)
        model: Pretrained DeepSeek-VL 2 model
    """

    # Load the multimodal processor (handles both image and text)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

    # Load the decoder-only vision-language model
    model = AutoModelForImageTextToText.from_pretrained(model_name, trust_remote_code=True)
    
    if torch.cuda.is_available():
        model = model.cuda()

    return processor, model


# Add these configurations to your model
def configure_model(encoder_decoder_model, text_tokenizer, max_length = 512, inference = False):
    """Configure T5-based model for document OCR"""
    
    # T5-specific token configurations
    encoder_decoder_model.config.decoder_start_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.pad_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.eos_token_id = text_tokenizer.eos_token_id
    
    encoder_decoder_model.config.use_cache = inference
    encoder_decoder_model.config.is_encoder_decoder = True
    encoder_decoder_model.config.use_cross_attention = True
    
    # Generation parameters
    encoder_decoder_model.config.max_length = max_length
    encoder_decoder_model.config.early_stopping = True
    encoder_decoder_model.config.no_repeat_ngram_size = 0  # Prevent repetition
    encoder_decoder_model.config.length_penalty = 1.0     # Encourage longer sequences
    
    
    return encoder_decoder_model


import os 

def get_local_model_dir(model_name="facebook/Perception-LM-1B", base_dir="models"):
    """
    Ensure the pretrained model is downloaded locally and return the local directory path.
    """
    local_dir = os.path.join(base_dir, model_name.replace("/", "-"))
    if not os.path.exists(local_dir):
        # download the model into local_dir
        PerceptionLMForConditionalGeneration.from_pretrained(model_name, cache_dir=local_dir)
    return True


def create_byte_level_tokenizer(vocab_size=32000, special_tokens=None):
    """
    Create a byte-level BPE tokenizer for OCR tasks.
    This handles all possible bytes, making it robust for OCR output.
    """
    if special_tokens is None:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    # Create byte-level BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Use byte-level pre-tokenizer (handles all Unicode properly)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Use byte-level decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post processor for special tokens
    tokenizer.post_processor = processors.BertProcessing(
        sep=("</s>", tokenizer.token_to_id("</s>")),
        cls=("<s>", tokenizer.token_to_id("<s>"))
    )
    
    return tokenizer

def train_byte_level_tokenizer_on_data(texts, vocab_size=32000):
    """
    Train a byte-level tokenizer on your OCR texts.
    """
    # Create tokenizer
    tokenizer = create_byte_level_tokenizer()
    
    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        min_frequency=2
    )
    
    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer)
    
    # Wrap in HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    # Set special tokens
    hf_tokenizer.bos_token = "<s>"
    hf_tokenizer.eos_token = "</s>"
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.unk_token = "<unk>"
    hf_tokenizer.mask_token = "<mask>"
    
    return hf_tokenizer

def init_dit_modernbert_models(
    encoder_model="microsoft/dit-base-finetuned-rvlcdip",
    decoder_model="answerdotai/ModernBERT-base",
    use_byte_level_tokenizer=False,
    custom_tokenizer=None
):
    """
    Initialize DiT + ModernBERT-decoder model.
    
    Args:
        encoder_model: DiT model path
        decoder_model: ModernBERT model path  
        use_byte_level_tokenizer: Whether to use custom byte-level tokenizer
        custom_tokenizer: Pre-trained custom tokenizer to use
    """
    print("Loading DiT + ModernBERT-decoder...")
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(encoder_model)
    
    # Handle tokenizer choice
    if custom_tokenizer is not None:
        text_tokenizer = custom_tokenizer
        print("Using provided custom tokenizer")
    elif use_byte_level_tokenizer:
        # You would train this on your OCR data first
        print("Note: You need to train byte-level tokenizer first!")
        print("Use train_byte_level_tokenizer_on_data() function")
        text_tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    else:
        # Use ModernBERT's BPE tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(decoder_model)
        print(f"Using ModernBERT BPE tokenizer, vocab size: {text_tokenizer.vocab_size}")
    
    
    encoder_decoder_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encoder_model,
        decoder_pretrained_model_name_or_path=decoder_model,
    )
    
    # Configure tokenizer
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    
    # Configure model
    encoder_decoder_model.config.decoder_start_token_id = text_tokenizer.bos_token_id
    encoder_decoder_model.config.pad_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
    
    # Generation parameters optimized for OCR
    encoder_decoder_model.config.max_length = 1024  # Longer for OCR
    encoder_decoder_model.config.min_length = 5
    encoder_decoder_model.config.no_repeat_ngram_size = 3
    encoder_decoder_model.config.early_stopping = True
    encoder_decoder_model.config.length_penalty = 1.2
    encoder_decoder_model.config.num_beams = 4
    encoder_decoder_model.config.repetition_penalty = 1.3
    
    # Resize token embeddings if using custom tokenizer
    encoder_decoder_model.decoder.resize_token_embeddings(len(text_tokenizer))
    
    print(f"Model initialized successfully!")
    print(f"Tokenizer type: {type(text_tokenizer.tokenizer).__name__ if hasattr(text_tokenizer, 'tokenizer') else 'Unknown'}")
    print(f"Vocab size: {text_tokenizer.vocab_size}")
    
    return image_processor, text_tokenizer, encoder_decoder_model




if __name__ == '__main__':
    # processor, model = init_deepseek_vl2_model()
    # get_local_model_dir()
    init_perception_lm_model_efficient()