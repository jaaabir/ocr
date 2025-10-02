import torch 
import evaluate 
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

def improved_collate_fn(batch, text_tokenizer, max_length=512):
    """
    Collate function with padding, truncation, and attention_mask handling.
    """
    pixel_values = []
    labels = []
    attn_masks = []
    
    for item in batch:
        # --- Pixel values ---
        if 'pixel_values' in item:
            pixel_values.append(item['pixel_values'])
        else:
            raise ValueError("Missing 'pixel_values' in batch item")
            
        # --- Labels ---
        if 'labels' in item:
            label = item['labels']
            if len(label.shape) == 0:  # single token
                label = label.unsqueeze(0)
            
            # Truncate if too long
            if label.shape[0] > max_length:
                label = label[:max_length-1]  # leave space for EOS
                eos_tensor = torch.tensor([text_tokenizer.eos_token_id], dtype=torch.long)
                label = torch.cat([label, eos_tensor])
            
            labels.append(label)
            # print(label)
        else:
            raise ValueError("Missing 'labels' in batch item")

        # --- Attention masks ---
        if 'attention_mask' in item and item['attention_mask'] is not None:
            attn_mask = item['attention_mask']
            if attn_mask.shape[0] > max_length:
                attn_mask = attn_mask[:max_length]
            attn_masks.append(attn_mask)
            # print(attn_mask)
        else:
            # if not provided, make a mask of ones
            attn_masks.append(torch.ones_like(label, dtype=torch.long))

        # print(f'Max token size: {max_length}')
        
    
    # --- Stack pixel values ---
    pixel_values = torch.stack(pixel_values)
    
    # --- Pad labels ---
    labels = pad_sequence(labels, batch_first=True, padding_value=text_tokenizer.pad_token_id)
    
    # --- Pad attention masks ---
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    
    # --- Replace pad tokens in labels with -100 (ignore index in loss) ---
    labels = labels.clone()
    labels[labels == text_tokenizer.pad_token_id] = -100
    
    res =  {
        'pixel_values': pixel_values,
        'labels': labels,
        'decoder_attention_mask': attn_masks
    }

    return res


def get_immediate_repetition_ratio(decoded_preds):
    repetitive_count = 0
    for pred in decoded_preds:
        words = pred.split()
        if len(words) > 3:
            # Check for immediate repetitions
            immediate_reps = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
            if immediate_reps > len(words) * 0.3:  # More than 30% repetition
                repetitive_count += 1
    
    return repetitive_count / len(decoded_preds)

from collections import Counter

def pred_intersect_labels(preds, labels):
    if not preds:
        return 0.0

    total_score = 0.0
    n = len(preds)

    for pred, label in zip(preds, labels):
        pred_set = set(pred.split())
        label_set = set(label.split())

        if not pred_set:
            continue

        overlap = len(pred_set & label_set)
        score = overlap / len(pred_set)
        total_score += score

    return total_score / n


bleu = evaluate.load("bleu")

def compute_bleu(decoded_preds, decoded_labels):
    try:
        bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
        bleu_value = bleu_score["bleu"]
    except:
        bleu_value = 0.0
    return bleu_value

def compute_metrics_ocr(eval_pred, tokenizer):
    """
    Compute OCR-specific metrics.
    """
    predictions, labels = eval_pred
    
    # Decode predictions and labels

    print('Evaluating ...')

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up texts
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    print(decoded_labels[0])
    print(decoded_preds[0])
    
    # Calculate BLEU score
    bleu_value = compute_bleu(decoded_preds, decoded_labels)
    pred_words_in_labels = pred_intersect_labels(decoded_preds, decoded_labels)
    
    
    metrics =  {
        "bleu": bleu_value,
        "pred_intersect_labels": pred_words_in_labels
    }

    return metrics

def test_model_before_training(model, image_processor, text_tokenizer, sample_image):
    """
    Test the model with a sample image before training.
    """
    print("Testing model before training...")
    model.eval()
    device = next(model.parameters()).device 
    with torch.no_grad():
        # Process image
        inputs = image_processor(sample_image, return_tensors="pt")
        
        # Generate with the same parameters as model config
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_length=model.config.max_length or 512,
            min_length=model.config.min_length or 5,
            num_beams=model.config.num_beams or 3,
            no_repeat_ngram_size=model.config.no_repeat_ngram_size or 2,
            early_stopping=model.config.early_stopping if hasattr(model.config, 'early_stopping') else True,
            pad_token_id=text_tokenizer.pad_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
            repetition_penalty=model.config.repetition_penalty or 1.3,
            length_penalty=model.config.length_penalty or 1.5
        )
        
        generated_text = text_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated text (before training): '{generated_text}'")
        
        # Check if it's repetitive
        words = generated_text.split()
        if len(words) > 3:
            unique_words = len(set(words))
            repetition_ratio = 1 - (unique_words / len(words))
            print(f"Repetition ratio: {repetition_ratio:.2f}")
            if repetition_ratio > 0.5:
                print("WARNING: High repetition detected!")
            else:
                print("Generation looks good - low repetition!")
        
        return generated_text
    
# Better training arguments
def setup_dit_bart_training(train_dataset, val_dataset, training_args=None, run_name="model_run", 
                            model=None, text_tokenizer=None, callbacks=[], max_length = 512):
    """
    Complete setup for DiT-BART training.
    """
    # Initialize model
    
    
    
    # Create collate function
    def collate_fn(batch):
        return improved_collate_fn(batch, text_tokenizer, max_length = max_length)
    
    # Create compute metrics function
    def compute_metrics(eval_pred):
        return compute_metrics_ocr(eval_pred, text_tokenizer)
    
    # Training arguments
    if training_args is None:
        print('Training args not provided, using defaults.')
        training_args = Seq2SeqTrainingArguments(
            output_dir="./dit_bart_outputs_fixed",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,  # Effective batch size = 16
            learning_rate=1e-5,  # Lower learning rate
            num_train_epochs=5,
            warmup_steps=10,
            logging_steps=5,
            save_steps=50,
            eval_steps=50,
            logging_strategy="steps",
            save_total_limit=3,
            fp16=True,
            max_grad_norm=0.5,
            weight_decay=0.01,
            dataloader_pin_memory=False,
            predict_with_generate=True,
            generation_max_length=512,
            generation_num_beams=3,
            report_to=["wandb"],  
            run_name=run_name,
        )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=text_tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    return trainer

import torch.nn as nn


def print_trainable_prams(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ Trainable: {name}")
        else:
            print(f"⛔ Frozen: {name}")

def unfreeze_all_params(model, unfreeze_encoder=True, unfreeze_decoder=True, skip_encoder=False, skip_decoder=False):
    if not skip_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True if unfreeze_encoder else False
    if not skip_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = True if unfreeze_decoder else False
    return model

def freeze_encoder_unfreeze_decoder(model, 
                                    train_decoder_embeddings=True,
                                    train_decoder_cross_attn=True, 
                                    train_decoder_ff=True, 
                                    train_decoder_self_attn=True, 
                                    train_lm_head=True,
                                    applied_lora=False,
                                    skip_encoder=False,
                                    skip_decoder=False,
                                    verbose=True):
    """
    Freeze the encoder and selectively unfreeze parts of the decoder.
    
    Args:
        model: VisionEncoderDecoderModel
        train_decoder_cross_attn (bool): Unfreeze cross-attention layers
        train_decoder_ff (bool): Unfreeze feedforward layers
        train_decoder_self_attn (bool): Unfreeze self-attention layers
        train_lm_head (bool): Unfreeze LM head (output projection)
    """

    # Freeze all encoder parameters
    model = unfreeze_all_params(model, unfreeze_encoder=False, unfreeze_decoder=False, skip_encoder=skip_encoder, skip_decoder=skip_decoder)

    if train_decoder_embeddings:
        for param in model.decoder.base_model.model.model.decoder.embed_tokens.parameters() if applied_lora else model.decoder.model.decoder.embed_tokens.parameters():
            param.requires_grad = True
        for param in model.decoder.base_model.model.model.decoder.embed_positions.parameters() if applied_lora else model.decoder.model.decoder.embed_positions.parameters():
            param.requires_grad = True

    # Now selectively unfreeze decoder parts
    for layer in model.decoder.base_model.model.model.decoder.layers if applied_lora else model.decoder.model.decoder.layers:
        # Cross-attention
        if train_decoder_cross_attn:
            for param in layer.encoder_attn.parameters():
                param.requires_grad = True

        # Self-attention
        if train_decoder_self_attn:
            for param in layer.self_attn.parameters():
                param.requires_grad = True

        # Feed-forward
        if train_decoder_ff:
            for param in layer.fc1.parameters():
                param.requires_grad = True
            for param in layer.fc2.parameters():
                param.requires_grad = True

    # LM head (final linear projection to vocab)
    if train_lm_head and hasattr(model.decoder, "lm_head"):
        for param in model.decoder.lm_head.parameters():
            param.requires_grad = True

    if verbose:
        print("Encoder frozen. Decoder partially trainable:")
        print_trainable_prams(model)
    return model

def unfreeze_last_n_encoder(model, 
                            unfreeze_last_n_layer_block=2,
                            unfreeze_attention_layers=False,
                            skip_encoder=False,
                            skip_decoder=False,
                            verbose=True):
    """
    Freeze the decoder and selectively unfreeze the last N layers of the encoder (DiT).

    Args:
        model: VisionEncoderDecoderModel
        unfreeze_last_n_layers (int): How many of the final encoder blocks to unfreeze
        train_encoder_embeddings (bool): Whether to also unfreeze encoder embeddings (patch + pos)
        skip_encoder (bool): If True, skip freezing encoder
        skip_decoder (bool): If True, skip freezing decoder
        verbose (bool): Print trainable params
    """

    # Step 1: Freeze everything
    model = unfreeze_all_params(model, 
                                unfreeze_encoder=False, 
                                unfreeze_decoder=False, 
                                skip_encoder=skip_encoder, 
                                skip_decoder=skip_decoder)

    
    if unfreeze_attention_layers:
        for layer in model.encoder.encoder.layer:
            for param in layer.attention.parameters():
                param.requires_grad = True

    if unfreeze_last_n_layer_block > 0:
        for layer in model.encoder.encoder.layer[-unfreeze_last_n_layer_block:]:
            for param in layer.parameters():
                param.requires_grad = True

    if verbose:
        print(f"Encoder partially trainable (last {unfreeze_last_n_layer_block} layers")
        print_trainable_prams(model)

    return model

def debug_model_inputs(trainer, model):
    sample_batch = next(iter(trainer.get_train_dataloader()))
    print("Sample batch keys:", sample_batch.keys())
    for key, value in sample_batch.items():
        print(f"{key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    
    # Try a forward pass to see what happens
    model.eval()
    model.to('cuda')
    print("Model device:", next(model.parameters()).device)
    with torch.no_grad():
        try:
            outputs = model(**sample_batch)
            print("Forward pass successful!")
        except Exception as e:
            print(f"Forward pass failed: {e}")
    model.train()

def train_dit_bart(train_dataset, val_dataset, model = None, training_args=None, sample_image=None, save_model = False, save_path = "./dit_bart_final", run_name = 'dit_bart_adam'):
    """
    Main function to train DiT-BART model.
    """
    print("Starting DiT-BART training setup...")
    
    # Setup training
    trainer, model, image_processor, text_tokenizer = setup_dit_bart_training(
        train_dataset, val_dataset, training_args=training_args, loaded_model = model, run_name = run_name
    )
    
    # Test model before training (optional)
    if sample_image is not None:
        test_model_before_training(model, image_processor, text_tokenizer, sample_image)
    
    # Start training
    print("Starting training...")
    history = trainer.train()
    
    # Save the final model
    if save_model:
        trainer.save_model(save_path)
        image_processor.save_pretrained(save_path)
        text_tokenizer.save_pretrained(save_path)
    
    print("Training completed!")
    return trainer, model, image_processor, text_tokenizer, history