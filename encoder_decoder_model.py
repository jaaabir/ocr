import torch 
from transformers import (
    VisionEncoderDecoderModel, AutoImageProcessor, T5Tokenizer, 
    AutoModel, T5ForConditionalGeneration, AutoTokenizer, 
    PerceptionLMForConditionalGeneration, PerceptionLMProcessor,
    AutoProcessor, AutoModelForImageTextToText, PreTrainedTokenizerFast )

from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, trainers

from accelerate import load_checkpoint_and_dispatch

from peft import LoraConfig, get_peft_model, TaskType

def add_lora_to_decoder(encoder_decoder_model, r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], bias="none", modules_to_save=None):
    """
    Wrap the decoder inside VisionEncoderDecoderModel with LoRA adapters.
    Fixed version that properly handles the decoder architecture.
    """
    # Use CAUSAL_LM task type since we're only applying LoRA to the decoder
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # Changed from SEQ_2_SEQ_LM
        r=r,                           # rank
        lora_alpha=alpha,              # scaling
        lora_dropout=dropout,
        target_modules=target_modules, # Include more attention modules
        bias=bias,
        # Add these parameters for better compatibility
        modules_to_save=modules_to_save,          # Don't save additional modules
    )
    
    # Apply LoRA **only** to the decoder
    encoder_decoder_model.decoder = get_peft_model(
        encoder_decoder_model.decoder, lora_config
    )
    
    print("LoRA applied to decoder:")
    encoder_decoder_model.decoder.print_trainable_parameters()
    
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


def init_dit_gpt_models(encoder_model="microsoft/dit-base-finetuned-rvlcdip", decoder_model="google/t5-v1_1-base"):

    # Initialize the VisionEncoderDecoderModel
    encoder_decoder_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encoder_model,
        decoder_pretrained_model_name_or_path=decoder_model,
    )

    # Initialize the image processor for the DiT encoder
    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)

    # Load the fast tokenizer for GPT-2
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True)

    # GPT-2 does not have pad_token by default, so we need to set it
    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    return image_processor, text_tokenizer, encoder_decoder_model


def init_dit_bart_models_fixed(
    encoder_model="microsoft/dit-base-finetuned-rvlcdip",
    decoder_model="facebook/bart-base",
):
    """
    Fixed version of DiT-BART initialization with proper cross-attention setup.
    """
    encoder_decoder_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_pretrained_model_name_or_path=encoder_model,
        decoder_pretrained_model_name_or_path=decoder_model,
        decoder_forced_bos_token_id=None,  
    )

    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)
    text_tokenizer = AutoTokenizer.from_pretrained(decoder_model, use_fast=True)

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

    encoder_decoder_model.config.decoder_start_token_id = text_tokenizer.bos_token_id
    encoder_decoder_model.config.pad_token_id = text_tokenizer.pad_token_id
    encoder_decoder_model.config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.config.vocab_size = text_tokenizer.vocab_size
    
    encoder_decoder_model.config.use_cache = False  
    encoder_decoder_model.config.is_encoder_decoder = True
    
    encoder_decoder_model.config.max_length = 512
    encoder_decoder_model.config.min_length = 5
    encoder_decoder_model.config.no_repeat_ngram_size = 2
    encoder_decoder_model.config.early_stopping = True
    encoder_decoder_model.config.length_penalty = 1.5
    encoder_decoder_model.config.num_beams = 3
    encoder_decoder_model.config.repetition_penalty = 1.3
    encoder_decoder_model.config.do_sample = False  
    
    if hasattr(encoder_decoder_model.decoder.config, 'tie_word_embeddings'):
        encoder_decoder_model.decoder.config.tie_word_embeddings = True
    
    encoder_decoder_model.decoder.config.vocab_size = text_tokenizer.vocab_size
    encoder_decoder_model.decoder.resize_token_embeddings(len(text_tokenizer))

    encoder_decoder_model.generation_config.bos_token_id = text_tokenizer.bos_token_id
    encoder_decoder_model.generation_config.eos_token_id = text_tokenizer.eos_token_id
    encoder_decoder_model.generation_config.pad_token_id = text_tokenizer.pad_token_id

    return image_processor, text_tokenizer, encoder_decoder_model


def init_dit_t5_models(encoder_model="microsoft/dit-base-finetuned-rvlcdip", decoder_model="google/t5-v1_1-base"):

    encoder = AutoModel.from_pretrained(encoder_model)

    full_t5 = T5ForConditionalGeneration.from_pretrained(decoder_model)
    decoder = full_t5.get_decoder()   # get only the decoder part
    decoder.config.is_decoder = True
    decoder.config.is_encoder_decoder = False
    decoder.config.add_cross_attention = True


    # Initialize the VisionEncoderDecoderModel
    encoder_decoder_model = VisionEncoderDecoderModel(
        encoder=encoder,
        decoder=decoder,

    )

    # Initialize the image processor for the DiT encoder
    image_processor = AutoImageProcessor.from_pretrained(encoder_model, use_fast=True)

    text_tokenizer = T5Tokenizer.from_pretrained(decoder_model)

    if text_tokenizer.pad_token is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token

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