"""
Complete Vision-Text Model Attention Visualization & Logit Lens Explorer

This comprehensive app lets you:

1. Visualize cross-attention (decoder tokens → image regions)
2. Apply logit lens to see intermediate predictions  
3. Analyze attention patterns to verify learning
4. Compare attention across layers
5. Interactive exploration of model internals

Author: Built for deep model interpretability
"""

import streamlit as st
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Any
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
from glob import glob
import traceback

# Try importing custom modules with error handling
try:
    from encoder_decoder_model import load_pretrained_enc_dec_model, init_dit_bart_models_fixed, load_pretrained_iprocessor_tokenizer
    from func_utils.pydataloader import SynthDogDataset
    CUSTOM_IMPORTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Custom modules not available: {e}")
    CUSTOM_IMPORTS_AVAILABLE = False

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

LORA_CONFIG = {
    'r': 32,
    'alpha': 64,
    'dropout': 0.35,
    'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
    'modules_to_save': None
}

MAX_TOKEN_SIZE = 512
SAMPLE_SIZE = 30
MAX_POSITIONS_TO_DISPLAY = 50
DEFAULT_MODEL_CHECKPOINT = os.path.join('saved_models', 'checkpoint-89992')
DEFAULT_DATA_ROOT = os.path.join('synthdog', 'outputs')

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.model = None
        st.session_state.image_processor = None
        st.session_state.tokenizer = None
        st.session_state.dataset = None
        st.session_state.current_image = None
        st.session_state.current_pixel_values = None
        st.session_state.generated_tokens = None
        st.session_state.attentions = None
        st.session_state.hidden_states = None
        st.session_state.token_ids = None
        st.session_state.decoded_tokens = None
        st.session_state.generated_text = None

def load_models_and_data():
    """Load model, processors, and dataset (only once per session)."""
    if not CUSTOM_IMPORTS_AVAILABLE:
        st.error("Cannot load models - custom modules not available")
        return False
        
    if st.session_state.initialized:
        return True
        
    try:
        with st.spinner('🔄 Loading model, tokenizer, and dataset... This may take a minute.'):
            # Load model
            st.session_state.model = load_pretrained_enc_dec_model(
                DEFAULT_MODEL_CHECKPOINT, **LORA_CONFIG
            )
            
            # Load processors
            image_processor, tokenizer = init_dit_bart_models_fixed(load_model=False)
            st.session_state.image_processor = image_processor
            st.session_state.tokenizer = tokenizer
            
            # Load dataset
            image_paths, json_paths = get_synth_images_json_path(
                data_root=DEFAULT_DATA_ROOT, split='train'
            )
            
            st.session_state.dataset = SynthDogDataset(
                image_path=image_paths,
                output_jsons_path=json_paths,
                image_feature_extractor=image_processor,
                text_tokenizer=tokenizer,
                max_token_size=MAX_TOKEN_SIZE,
                sample_size=SAMPLE_SIZE,
                read_images_from_supabase=False,
                split='train'
            )
            
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.error(f"Failed to load model/data: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return False

# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_synth_images_json_path(data_root: str, split: str = 'train') -> Tuple[List[str], List[str]]:
    """Get paths to dataset images and metadata."""
    image_pattern = os.path.join(data_root, '*', split, '*.jpg')
    json_pattern = os.path.join(data_root, '*', split, 'metadata.jsonl')
    return glob(image_pattern), glob(json_pattern)

def load_image_from_dataset(dataset: Any, idx: int) -> Tuple[torch.Tensor, Image.Image]:
    """Load and preprocess image from dataset."""
    sample = dataset[idx]
    
    # Extract image
    if isinstance(sample, dict):
        if 'pixel_values' in sample:
            pixel_values = sample['pixel_values']
            if not isinstance(pixel_values, torch.Tensor):
                pixel_values = torch.tensor(pixel_values)
            pixel_values = pixel_values.unsqueeze(0) if pixel_values.dim() == 3 else pixel_values
            
            # Try to get PIL image for display
            if 'image' in sample:
                pil_img = sample['image']
                if not isinstance(pil_img, Image.Image):
                    pil_img = Image.fromarray(np.array(pil_img))
            else:
                pil_img = None
                
            return pixel_values, pil_img
            
    raise ValueError("Could not extract image from dataset sample")

# ============================================================================
# MODEL INFERENCE WITH ATTENTION CAPTURE
# ============================================================================

def generate_with_attention(model: torch.nn.Module,
                          pixel_values: torch.Tensor,
                          max_length: int = 50,
                          return_hidden_states: bool = True) -> Dict[str, Any]:
    """
    Generate text from image and capture all attention weights and hidden states.
    
    Returns:
        Dictionary with:
        - token_ids: Generated token IDs
        - attentions: Cross-attention weights (decoder → encoder)
        - hidden_states: Decoder hidden states per layer
        - decoded_text: Human-readable text
    """
    device = next(model.parameters()).device
    pixel_values = pixel_values.to(device)
    
    with torch.no_grad():
        # Generate with attention outputs
        outputs = model.generate(
            pixel_values,
            max_length=max_length,
            output_attentions=True,
            output_hidden_states=return_hidden_states,
            return_dict_in_generate=True,
            output_scores=True
        )

        print(dir(outputs))
        
        result = {
            'token_ids': outputs.sequences[0].cpu(),  # [seq_len]
            'attentions': None,
            'hidden_states': None,
            'cross_attentions': None,
            'decoded_text': None
        }
        
        # Process attentions
        # outputs.decoder_attentions is a tuple of tuples: (layer_attn_step1, layer_attn_step2, ...)
        # Each step has (layer0, layer1, ..., layerN)

        print(outputs.encoder_attentions)
        if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions is not None:
            result['cross_attentions'] = process_cross_attentions(outputs.cross_attentions)
            
        if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            result['hidden_states'] = process_hidden_states(outputs.decoder_hidden_states)
            
        # Decode text
        tokenizer = st.session_state.tokenizer
        result['decoded_text'] = tokenizer.decode(result['token_ids'], skip_special_tokens=True)
        result['decoded_tokens'] = [tokenizer.decode([tid]) for tid in result['token_ids']]
        
    return result

# --- PATCH START (only diff excerpt) ---
# Replace the old process_cross_attentions function with the improved version below


def process_cross_attentions(cross_attentions: Tuple) -> Optional[Dict[str, torch.Tensor]]:
    """Safely process cross-attention outputs returned by generate().

    VisionEncoderDecoderModel.generate may return `cross_attentions` that is:
    1. `None` (feature not supported by the model), or
    2. A tuple where each timestep or individual layer is `None` (e.g. when the
       model is configured without `output_cross_attentions=True`).

    This helper sanitizes the structure and returns `None` when no valid tensor
    is found so the rest of the app can gracefully skip the visualisation step.
    """
    # Case 1 – Entire object is falsy → nothing to process
    if not cross_attentions:
        return None

    # Helper to find the first non-None tensor to infer dimensions
    first_valid = None
    for step in cross_attentions:
        if step is None:
            continue
        for layer_attn in step:
            if layer_attn is not None:
                first_valid = layer_attn
                break
        if first_valid is not None:
            break

    # If still None → the model did not compute cross-attention at all
    if first_valid is None:
        return None

    # Infer sizes from the first valid tensor [batch, heads, tgt_len(=1), enc_len]
    num_heads = first_valid.shape[1]
    encoder_seq_len = first_valid.shape[-1]

    # Determine sequence & layer dimensions (may vary across steps)
    num_tokens = len(cross_attentions)
    # Some steps may be None, find a representative layer count
    for step in cross_attentions:
        if step is not None:
            num_layers = len(step)
            break
    else:
        # Fallback
        num_layers = 1

    # Pre-allocate tensor filled with zeros → safer for missing values
    per_layer = torch.zeros(num_layers, num_tokens, num_heads, encoder_seq_len)

    # Populate tensor where data is available
    for tok_idx, step in enumerate(cross_attentions):
        if step is None:
            continue
        for layer_idx, layer_attn in enumerate(step):
            if layer_attn is None:
                continue
            per_layer[layer_idx, tok_idx] = layer_attn[0, :, 0, :].cpu()

    # Aggregate
    per_layer_averaged_heads = per_layer.mean(dim=2)         # [layers, tokens, enc_seq]
    averaged = per_layer_averaged_heads.mean(dim=0)          # [tokens, enc_seq]
    result = {
        'per_layer': per_layer,
        'per_layer_averaged_heads': per_layer_averaged_heads,
        'averaged': averaged,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'encoder_seq_len': encoder_seq_len
    }
    print(result)
    return result

def process_hidden_states(hidden_states: Tuple) -> Dict[str, torch.Tensor]:
    """Process hidden states from generation."""
    if not hidden_states:
        return None
        
    # hidden_states is tuple of length (num_tokens)
    # Each element is tuple of length (num_layers + 1) for embeddings + each layer
    num_tokens = len(hidden_states)
    num_layers = len(hidden_states[0]) - 1  # Exclude embedding layer
    hidden_dim = hidden_states[0][0].shape[-1]
    
    # Stack all hidden states [num_layers, num_tokens, hidden_dim]
    processed = torch.zeros(num_layers, num_tokens, hidden_dim)
    
    for token_idx in range(num_tokens):
        for layer_idx in range(num_layers):
            # Skip embedding layer (index 0), use layer outputs (indices 1+)
            processed[layer_idx, token_idx] = hidden_states[token_idx][layer_idx + 1][0, -1, :].cpu()
    
    return {
        'per_layer': processed,  # [num_layers, num_tokens, hidden_dim]
        'num_layers': num_layers,
        'hidden_dim': hidden_dim
    }

# ============================================================================
# ATTENTION VISUALIZATION UTILITIES
# ============================================================================

def reshape_attention_to_image(attention_weights: torch.Tensor,
                             image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Reshape 1D attention weights to 2D image grid.
    
    Args:
        attention_weights: [encoder_seq_len] attention values
        image_shape: (height, width) of original image
        
    Returns:
        2D attention map [H, W]
    """
    seq_len = attention_weights.shape[0]
    
    # For ViT-like encoders, seq_len = num_patches + 1 (CLS token)
    # Assume square patches
    has_cls = True  # Most vision transformers have CLS token
    if has_cls:
        # Remove CLS token attention (usually first token)
        attn = attention_weights[1:].numpy()
        num_patches = len(attn)
    else:
        attn = attention_weights.numpy()
        num_patches = len(attn)
    
    # Calculate patch grid size
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        # Handle non-square case
        st.warning(f"Non-square patch grid: {num_patches} patches")
        grid_size = int(np.ceil(np.sqrt(num_patches)))
        # Pad with zeros
        padded = np.zeros(grid_size * grid_size)
        padded[:num_patches] = attn
        attn = padded
    
    # Reshape to 2D grid
    attn_map = attn.reshape(grid_size, grid_size)
    return attn_map

def create_attention_heatmap(image: Image.Image,
                           attention_map: np.ndarray,
                           alpha: float = 0.6,
                           cmap: str = 'jet') -> Image.Image:
    """
    Overlay attention heatmap on original image.
    
    Args:
        image: PIL Image
        attention_map: 2D numpy array [H, W]
        alpha: Transparency of heatmap overlay
        cmap: Matplotlib colormap name
        
    Returns:
        PIL Image with attention overlay
    """
    # Normalize attention
    attn_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Resize attention map to match image size
    attn_resized = Image.fromarray((attn_norm * 255).astype(np.uint8))
    attn_resized = attn_resized.resize(image.size, Image.BILINEAR)
    
    # Apply colormap
    attn_colored = plt.get_cmap(cmap)(np.array(attn_resized) / 255.0)
    attn_colored = (attn_colored[:, :, :3] * 255).astype(np.uint8)
    attn_colored = Image.fromarray(attn_colored)
    
    # Blend with original image
    blended = Image.blend(image.convert('RGB'), attn_colored, alpha=alpha)
    
    return blended

def plot_attention_grid(image: Image.Image,
                       attention_maps: List[np.ndarray],
                       tokens: List[str],
                       max_cols: int = 5) -> plt.Figure:
    """
    Create grid of attention maps for multiple tokens.
    
    Args:
        image: Original image
        attention_maps: List of 2D attention maps
        tokens: List of token strings
        max_cols: Maximum columns in grid
        
    Returns:
        Matplotlib figure
    """
    num_tokens = len(tokens)
    num_cols = min(max_cols, num_tokens)
    num_rows = (num_tokens + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    
    if num_rows == 1:
        axes = [axes] if num_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx in range(num_tokens):
        ax = axes[idx]
        
        # Create overlay
        overlay = create_attention_heatmap(image, attention_maps[idx], alpha=0.5)
        ax.imshow(overlay)
        ax.set_title(f"Token {idx}: '{tokens[idx]}'", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_tokens, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def plot_attention_comparison(attention_maps: List[np.ndarray],
                            layer_names: List[str],
                            token_idx: int,
                            token_str: str) -> plt.Figure:
    """
    Compare attention across different layers for a single token.
    
    Args:
        attention_maps: List of 2D attention maps (one per layer)
        layer_names: List of layer names
        token_idx: Token position
        token_str: Token string
        
    Returns:
        Matplotlib figure
    """
    num_layers = len(attention_maps)
    num_cols = min(4, num_layers)
    num_rows = (num_layers + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    
    if num_rows == 1:
        axes = [axes] if num_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, (attn_map, layer_name) in enumerate(zip(attention_maps, layer_names)):
        ax = axes[idx]
        im = ax.imshow(attn_map, cmap='viridis', aspect='auto')
        ax.set_title(f"{layer_name}", fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"Layer Comparison - Token {token_idx}: '{token_str}'", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig

# ============================================================================
# LOGIT LENS UTILITIES
# ============================================================================

def find_lm_head(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Find the language model head in the model."""
    candidates = [
        getattr(model, 'lm_head', None),
        getattr(getattr(model, 'decoder', None), 'lm_head', None),
    ]
    
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None

def apply_logit_lens(hidden_states: torch.Tensor,
                    lm_head: torch.nn.Module,
                    tokenizer: Any,
                    top_k: int = 5) -> List[List[Tuple[str, float]]]:
    """
    Apply logit lens to hidden states to see predicted tokens.
    
    Args:
        hidden_states: [num_tokens, hidden_dim]
        lm_head: Language model head
        tokenizer: Tokenizer for decoding
        top_k: Number of top predictions
        
    Returns:
        List of (token, probability) tuples per position
    """
    # Project through LM head
    weight = lm_head.weight.detach().cpu()
    bias = getattr(lm_head, 'bias', None)
    
    logits = torch.matmul(hidden_states, weight.t())
    if bias is not None:
        logits = logits + bias.detach().cpu()
    
    probs = F.softmax(logits, dim=-1)
    
    # Get top-k per position
    topk = torch.topk(probs, k=top_k, dim=-1)
    values = topk.values.numpy()
    indices = topk.indices.numpy()
    
    results = []
    for pos in range(len(hidden_states)):
        pos_results = []
        for rank in range(top_k):
            token_id = int(indices[pos, rank])
            token_str = tokenizer.decode([token_id])
            prob = float(values[pos, rank])
            pos_results.append((token_str, prob))
        results.append(pos_results)
    
    return results

# ============================================================================
# ANALYSIS & METRICS
# ============================================================================

def compute_attention_statistics(attention_weights: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about attention patterns.
    
    Args:
        attention_weights: [num_tokens, encoder_seq_len]
        
    Returns:
        Dictionary of metrics
    """
    attn = attention_weights.numpy()
    
    # Entropy (lower = more focused)
    epsilon = 1e-10
    entropy = -np.sum(attn * np.log(attn + epsilon), axis=1)
    
    # Max attention value per token
    max_attn = np.max(attn, axis=1)
    
    # Effective attention span (number of patches with >5% attention)
    effective_span = np.sum(attn > 0.05, axis=1)
    
    return {
        'mean_entropy': float(np.mean(entropy)),
        'std_entropy': float(np.std(entropy)),
        'mean_max_attention': float(np.mean(max_attn)),
        'mean_effective_span': float(np.mean(effective_span)),
        'max_max_attention': float(np.max(max_attn)),
        'min_max_attention': float(np.min(max_attn))
    }

def assess_learning_quality(stats: Dict[str, float]) -> Dict[str, Any]:
    """
    Assess if the model shows signs of good learning based on attention.
    
    Returns:
        Dictionary with assessment results
    """
    assessment = {
        'focus_score': 0.0,  # 0-100
        'confidence_score': 0.0,  # 0-100
        'overall_grade': 'F',
        'observations': []
    }
    
    # Focus: Low entropy is good (focused attention)
    # Typical entropy range: 1.0 (focused) to 5.0 (diffuse)
    entropy = stats['mean_entropy']
    if entropy < 2.0:
        focus_score = 100
        assessment['observations'].append("✅ Excellent focus - attention is sharp and concentrated")
    elif entropy < 3.0:
        focus_score = 75
        assessment['observations'].append("✓ Good focus - attention is reasonably concentrated")
    elif entropy < 4.0:
        focus_score = 50
        assessment['observations'].append("⚠ Moderate focus - attention somewhat scattered")
    else:
        focus_score = 25
        assessment['observations'].append("❌ Poor focus - attention is very diffuse")
    
    # Confidence: High max attention is good
    max_attn = stats['mean_max_attention']
    if max_attn > 0.3:
        confidence_score = 100
        assessment['observations'].append("✅ High confidence - model strongly attends to key regions")
    elif max_attn > 0.2:
        confidence_score = 75
        assessment['observations'].append("✓ Good confidence - clear attention peaks")
    elif max_attn > 0.1:
        confidence_score = 50
        assessment['observations'].append("⚠ Moderate confidence - weak attention peaks")
    else:
        confidence_score = 25
        assessment['observations'].append("❌ Low confidence - no clear attention focus")
    
    assessment['focus_score'] = focus_score
    assessment['confidence_score'] = confidence_score
    
    # Overall grade
    overall = (focus_score + confidence_score) / 2
    if overall >= 90:
        assessment['overall_grade'] = 'A+'
    elif overall >= 80:
        assessment['overall_grade'] = 'A'
    elif overall >= 70:
        assessment['overall_grade'] = 'B'
    elif overall >= 60:
        assessment['overall_grade'] = 'C'
    elif overall >= 50:
        assessment['overall_grade'] = 'D'
    else:
        assessment['overall_grade'] = 'F'
    
    return assessment

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        layout='wide',
        page_title='🔍 Model Attention & Logit Lens Explorer',
        initial_sidebar_state='expanded'
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title('🔍 Complete Model Interpretability Dashboard')
    st.markdown("""
    **Explore your Vision-Text model's internals:**
    - 👁️ See what image regions the model attends to
    - 🧠 View intermediate predictions with logit lens
    - 📊 Analyze attention patterns to verify learning
    - 🔬 Compare across layers and tokens
    """)
    
    # Load models
    if not load_models_and_data():
        st.error("Failed to initialize. Please check your model and data paths.")
        return
    
    st.success("✅ Model, tokenizer, and dataset loaded successfully!")
    
    # ========================================================================
    # SIDEBAR: Image Selection & Global Controls
    # ========================================================================
    
    with st.sidebar:
        st.header('🖼️ Image Selection')
        
        dataset = st.session_state.dataset
        if dataset is None:
            st.error("Dataset not loaded")
            return
        
        image_idx = st.number_input(
            'Dataset Index',
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1,
            help='Select an image from the dataset'
        )
        
        # Load and preview image
        if st.button('Load Image', type='primary', use_container_width=True):
            try:
                pixel_values, pil_img = load_image_from_dataset(dataset, image_idx)
                st.session_state.current_image = pil_img
                st.session_state.current_pixel_values = pixel_values
                st.success(f"Loaded image #{image_idx}")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        
        # Show preview
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, caption=f"Image #{image_idx}", use_column_width=True)
        
        st.divider()
        
        # Generation controls
        st.header('⚙️ Generation Settings')
        max_length = st.slider('Max tokens to generate', 10, 512, 50)
        
        if st.button('🚀 Generate & Capture Attention', type='primary', use_container_width=True):
            if st.session_state.current_pixel_values is None:
                st.error("Please load an image first!")
            else:
                with st.spinner('Generating and capturing attention...'):
                    try:
                        model = st.session_state.model
                        result = generate_with_attention(
                            model,
                            st.session_state.current_pixel_values,
                            max_length=max_length,
                            return_hidden_states=True
                        )
                        
                        st.session_state.token_ids = result['token_ids']
                        st.session_state.decoded_tokens = result['decoded_tokens']
                        st.session_state.attentions = result['cross_attentions']
                        st.session_state.hidden_states = result['hidden_states']
                        st.session_state.generated_text = result['decoded_text']
                        
                        st.success("✅ Generation complete!")
                        st.write(f"**Generated text:** {result['decoded_text']}")
                        
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.exception(e)
    
    # ========================================================================
    # MAIN CONTENT: Tabbed Interface
    # ========================================================================
    
    if st.session_state.attentions is None:
        st.info("👈 Load an image and click 'Generate & Capture Attention' to begin exploration!")
        return
    
    tabs = st.tabs([
        "📊 Overview",
        "🔥 Attention Heatmaps", 
        "🎯 Token-by-Token",
        "🏗️ Layer Comparison",
        "🎬 Animation",
        "🧠 Logit Lens",
        "📈 Analysis & Metrics"
    ])
    
    attentions = st.session_state.attentions
    tokens = st.session_state.decoded_tokens
    image = st.session_state.current_image
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    
    with tabs[0]:
        st.header('📊 Generation Overview')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Tokens Generated', len(tokens))
            st.metric('Number of Layers', attentions['num_layers'])
        
        with col2:
            st.metric('Attention Heads', attentions['num_heads'])
            st.metric('Encoder Sequence Length', attentions['encoder_seq_len'])
        
        with col3:
            stats = compute_attention_statistics(attentions['averaged'])
            st.metric('Mean Attention Entropy', f"{stats['mean_entropy']:.2f}")
            st.metric('Mean Max Attention', f"{stats['mean_max_attention']:.3f}")
        
        st.divider()
        
        st.subheader('Generated Text')
        st.markdown(f"**Full text:** `{st.session_state.generated_text}`")
        
        st.subheader('Token Breakdown')
        token_df = pd.DataFrame({
            'Position': range(len(tokens)),
            'Token': tokens,
            'Token ID': st.session_state.token_ids.numpy()
        })
        st.dataframe(token_df, use_container_width=True, height=300)
    
    # ========================================================================
    # TAB 2: ATTENTION HEATMAPS
    # ========================================================================
    
    with tabs[1]:
        st.header('🔥 Attention Heatmaps Overlaid on Image')
        st.markdown("See which image regions each token attends to")
        
        # Layer selection
        layer_idx = st.selectbox(
            'Select Layer',
            range(attentions['num_layers']),
            format_func=lambda x: f"Layer {x}"
        )
        
        alpha = st.slider('Heatmap Transparency', 0.0, 1.0, 0.6, 0.1)
        cmap_choice = st.selectbox('Colormap', ['jet', 'hot', 'viridis', 'plasma', 'inferno'])
        
        # Get attention for selected layer
        layer_attentions = attentions['per_layer_averaged_heads'][layer_idx]  # [num_tokens, enc_seq]
        
        # Create grid
        st.subheader(f'Attention Grid (Layer {layer_idx})')
        num_display = st.slider('Number of tokens to display', 1, min(20, len(tokens)), min(10, len(tokens)))
        
        attention_maps = []
        for token_idx in range(num_display):
            attn_1d = layer_attentions[token_idx]
            attn_2d = reshape_attention_to_image(attn_1d, image.size)
            attention_maps.append(attn_2d)
        
        fig = plot_attention_grid(
            image,
            attention_maps,
            tokens[:num_display],
            max_cols=5
        )
        
        st.pyplot(fig)
        plt.close()
    
    # ========================================================================
    # TAB 3: TOKEN-BY-TOKEN EXPLORER
    # ========================================================================
    
    with tabs[2]:
        st.header('🎯 Token-by-Token Attention Explorer')
        st.markdown("Explore individual token attention in detail")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            token_idx = st.selectbox(
                'Select Token',
                range(len(tokens)),
                format_func=lambda x: f"{x}: '{tokens[x]}'"
            )
            
            layer_idx = st.selectbox(
                'Layer',
                range(attentions['num_layers']),
                format_func=lambda x: f"Layer {x}",
                key='token_layer'
            )
            
            st.markdown(f"### Token {token_idx}")
            st.markdown(f"**Text:** `{tokens[token_idx]}`")
            st.markdown(f"**ID:** `{st.session_state.token_ids[token_idx].item()}`")
        
        with col1:
            # Get attention for this token
            attn_1d = attentions['per_layer_averaged_heads'][layer_idx][token_idx]
            attn_2d = reshape_attention_to_image(attn_1d, image.size)
            
            # Create overlay
            overlay = create_attention_heatmap(image, attn_2d, alpha=0.6)
            st.image(overlay, caption=f"Attention for token '{tokens[token_idx]}' (Layer {layer_idx})", use_column_width=True)
        
        # Attention statistics for this token
        st.subheader('Attention Distribution')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Heatmap
        im1 = ax1.imshow(attn_2d, cmap='hot', aspect='auto')
        ax1.set_title('Attention Map (2D)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)
        
        # Distribution
        attn_flat = attn_1d.numpy()
        ax2.hist(attn_flat, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_title('Attention Weight Distribution')
        ax2.set_xlabel('Attention Weight')
        ax2.set_ylabel('Frequency')
        ax2.axvline(attn_flat.mean(), color='r', linestyle='--', label=f'Mean: {attn_flat.mean():.3f}')
        ax2.axvline(attn_flat.max(), color='g', linestyle='--', label=f'Max: {attn_flat.max():.3f}')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # ========================================================================
    # TAB 4: LAYER COMPARISON
    # ========================================================================
    
    with tabs[3]:
        st.header('🏗️ Cross-Layer Attention Comparison')
        st.markdown("See how attention evolves across layers")
        
        token_idx = st.selectbox(
            'Select Token to Compare',
            range(len(tokens)),
            format_func=lambda x: f"{x}: '{tokens[x]}'",
            key='compare_token'
        )
        
        st.markdown(f"### Comparing token {token_idx}: `{tokens[token_idx]}`")
        
        # Get attention maps for all layers
        attention_maps = []
        layer_names = []
        for layer_idx in range(attentions['num_layers']):
            attn_1d = attentions['per_layer_averaged_heads'][layer_idx][token_idx]
            attn_2d = reshape_attention_to_image(attn_1d, image.size)
            attention_maps.append(attn_2d)
            layer_names.append(f"Layer {layer_idx}")
        
        fig = plot_attention_comparison(
            attention_maps,
            layer_names,
            token_idx,
            tokens[token_idx]
        )
        
        st.pyplot(fig)
        plt.close()
        
        st.subheader('Layer-wise Attention Statistics')
        
        # Compute stats per layer
        layer_stats = []
        for layer_idx in range(attentions['num_layers']):
            attn_1d = attentions['per_layer_averaged_heads'][layer_idx][token_idx]
            layer_stats.append({
                'Layer': layer_idx,
                'Max Attention': f"{attn_1d.max().item():.4f}",
                'Mean Attention': f"{attn_1d.mean().item():.4f}",
                'Std Attention': f"{attn_1d.std().item():.4f}",
                'Entropy': f"{-torch.sum(attn_1d * torch.log(attn_1d + 1e-10)).item():.2f}"
            })
        
        stats_df = pd.DataFrame(layer_stats)
        st.dataframe(stats_df, use_container_width=True)
    
    # ========================================================================
    # TAB 5: ANIMATION
    # ========================================================================
    
    with tabs[4]:
        st.header('🎬 Attention Animation Through Tokens')
        st.markdown("Watch how attention shifts as the model generates each token")
        
        layer_idx = st.selectbox(
            'Select Layer for Animation',
            range(attentions['num_layers']),
            format_func=lambda x: f"Layer {x}",
            key='anim_layer'
        )
        
        token_slider = st.slider(
            'Token Position',
            0,
            len(tokens) - 1,
            0,
            help='Slide to see attention at each generation step'
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Get attention for current token
            attn_1d = attentions['per_layer_averaged_heads'][layer_idx][token_slider]
            attn_2d = reshape_attention_to_image(attn_1d, image.size)
            overlay = create_attention_heatmap(image, attn_2d, alpha=0.6)
            st.image(overlay, caption=f"Token {token_slider}: '{tokens[token_slider]}'", use_column_width=True)
        
        with col2:
            st.markdown(f"### Token {token_slider}")
            st.markdown(f"**Text:** `{tokens[token_slider]}`")
            st.markdown(f"**Layer:** {layer_idx}")
            st.markdown(f"**Max Attn:** {attn_1d.max().item():.4f}")
            st.markdown(f"**Mean Attn:** {attn_1d.mean().item():.4f}")
            
            # Show context (previous tokens)
            if token_slider > 0:
                st.markdown("**Previous tokens:**")
                context = " ".join(tokens[max(0, token_slider-5):token_slider])
                st.markdown(f"`{context}`")
    
    # ========================================================================
    # TAB 6: LOGIT LENS
    # ========================================================================
    
    with tabs[5]:
        st.header('🧠 Logit Lens: Intermediate Predictions')
        st.markdown("See what tokens the model predicts at each layer")
        
        if st.session_state.hidden_states is None:
            st.warning("Hidden states not captured. Try regenerating with attention capture.")
        else:
            hidden_states = st.session_state.hidden_states
            lm_head = find_lm_head(st.session_state.model)
            
            if lm_head is None:
                st.error("Could not find LM head in model")
            else:
                layer_idx = st.selectbox(
                    'Select Layer for Logit Lens',
                    range(hidden_states['num_layers']),
                    format_func=lambda x: f"Layer {x}",
                    key='logit_layer'
                )
                
                top_k = st.slider('Top-K predictions', 1, 10, 5, key='logit_k')
                
                # Apply logit lens
                layer_hidden = hidden_states['per_layer'][layer_idx]  # [num_tokens, hidden_dim]
                predictions = apply_logit_lens(
                    layer_hidden,
                    lm_head,
                    st.session_state.tokenizer,
                    top_k=top_k
                )
                
                st.subheader(f'Predictions at Layer {layer_idx}')
                
                # Create comparison table
                table_data = []
                for pos in range(len(tokens)):
                    actual = tokens[pos]
                    predicted = predictions[pos]
                    pred_str = " | ".join([f"{tok.strip()} ({prob:.2f})" for tok, prob in predicted[:3]])
                    
                    # Check if actual token is in top-k
                    pred_tokens = [tok.strip() for tok, _ in predicted]
                    is_correct = actual.strip() in pred_tokens
                    
                    table_data.append({
                        'Pos': pos,
                        'Actual Token': actual,
                        'Top-3 Predictions': pred_str,
                        'Correct?': '✅' if is_correct else '❌'
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, height=400)
                
                # Accuracy metric
                correct = sum(1 for row in table_data if row['Correct?'] == '✅')
                accuracy = (correct / len(table_data)) * 100
                st.metric(f'Top-{top_k} Accuracy at Layer {layer_idx}', f"{accuracy:.1f}%")
    
    # ========================================================================
    # TAB 7: ANALYSIS & METRICS
    # ========================================================================
    
    with tabs[6]:
        st.header('📈 Attention Analysis & Learning Quality Assessment')
        
        # Compute overall statistics
        stats = compute_attention_statistics(attentions['averaged'])
        assessment = assess_learning_quality(stats)
        
        # Overall grade
        st.subheader('🎓 Learning Quality Grade')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric('Overall Grade', assessment['overall_grade'])
        
        with col2:
            st.metric('Focus Score', f"{assessment['focus_score']:.0f}/100")
        
        with col3:
            st.metric('Confidence Score', f"{assessment['confidence_score']:.0f}/100")
        
        # Observations
        st.subheader('📋 Observations')
        for obs in assessment['observations']:
            st.markdown(f"- {obs}")
        
        st.divider()
        
        # Detailed statistics
        st.subheader('📊 Detailed Attention Statistics')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribution Metrics:**")
            st.metric('Mean Entropy', f"{stats['mean_entropy']:.3f}")
            st.metric('Std Entropy', f"{stats['std_entropy']:.3f}")
            st.caption("Lower entropy = more focused attention")
        
        with col2:
            st.markdown("**Attention Strength:**")
            st.metric('Mean Max Attention', f"{stats['mean_max_attention']:.3f}")
            st.metric('Peak Max Attention', f"{stats['max_max_attention']:.3f}")
            st.caption("Higher values = stronger attention peaks")
        
        # Attention evolution plot
        st.subheader('📈 Attention Evolution Over Tokens')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Max attention per token
        ax1 = axes[0, 0]
        max_attns = [attentions['averaged'][i].max().item() for i in range(len(tokens))]
        ax1.plot(max_attns, marker='o', linewidth=2)
        ax1.set_title('Max Attention per Token')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Max Attention Weight')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Entropy per token
        ax2 = axes[0, 1]
        entropies = []
        for i in range(len(tokens)):
            attn = attentions['averaged'][i].numpy()
            ent = -np.sum(attn * np.log(attn + 1e-10))
            entropies.append(ent)
        ax2.plot(entropies, marker='s', color='orange', linewidth=2)
        ax2.set_title('Attention Entropy per Token')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Entropy')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Effective attention span
        ax3 = axes[1, 0]
        spans = [np.sum(attentions['averaged'][i].numpy() > 0.05) for i in range(len(tokens))]
        ax3.bar(range(len(spans)), spans, color='green', alpha=0.7)
        ax3.set_title('Effective Attention Span (>5% threshold)')
        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Number of Patches')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Attention heatmap over time
        ax4 = axes[1, 1]
        attn_matrix = attentions['averaged'].numpy()
        im = ax4.imshow(attn_matrix, aspect='auto', cmap='hot')
        ax4.set_title('Attention Heatmap (All Tokens)')
        ax4.set_xlabel('Encoder Position')
        ax4.set_ylabel('Token Position')
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Layer-wise comparison
        st.subheader('🏗️ Layer-wise Learning Quality')
        
        layer_assessments = []
        for layer_idx in range(attentions['num_layers']):
            layer_attn = attentions['per_layer_averaged_heads'][layer_idx]
            layer_stats = compute_attention_statistics(layer_attn)
            layer_assess = assess_learning_quality(layer_stats)
            layer_assessments.append({
                'Layer': layer_idx,
                'Grade': layer_assess['overall_grade'],
                'Focus': f"{layer_assess['focus_score']:.0f}",
                'Confidence': f"{layer_assess['confidence_score']:.0f}",
                'Entropy': f"{layer_stats['mean_entropy']:.2f}",
                'Max Attn': f"{layer_stats['mean_max_attention']:.3f}"
            })
        
        layer_df = pd.DataFrame(layer_assessments)
        st.dataframe(layer_df, use_container_width=True)
        
        # Insights
        st.subheader('💡 Key Insights')
        
        best_layer = layer_df.loc[layer_df['Focus'].astype(float).idxmax()]
        st.markdown(f"**Best performing layer:** Layer {int(best_layer['Layer'])} (Grade: {best_layer['Grade']})")
        
        if stats['mean_entropy'] < 3.0:
            st.success("✅ Model shows good attention focus overall")
        else:
            st.warning("⚠️ Model attention is somewhat diffuse - may benefit from more training")
        
        if stats['mean_max_attention'] > 0.2:
            st.success("✅ Model shows strong attention peaks - confident predictions")
        else:
            st.warning("⚠️ Weak attention peaks - model may be uncertain")

if __name__ == '__main__':
    main()