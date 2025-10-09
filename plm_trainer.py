from glob import glob
import os 

import torch 
from func_utils.pydataloader import SynthDogDataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback, Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence


import wandb
import gc

from func_utils.trainer_utils import print_trainable_prams

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()
wandb.login()


def get_synth_images_json_path(data_root= os.path.join('synthdog','outputs'), split='train'):
    ipath = os.path.join(data_root, '*', split, '*.jpg')
    json_path = os.path.join(data_root, '*', split, 'metadata.jsonl')

    return glob(ipath), glob(json_path)

def load_plm_and_processor(ckpt_path):
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"
    )

    model_name = "saved_models/plm_1b_base"
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    return prepare_model_for_kbit_training(model), processor


root_path = os.path.join('synth-text-generator', 'outputs')
train_image_path, train_json_metadata = get_synth_images_json_path(data_root=root_path, split='train')
val_image_path, val_json_metadata = get_synth_images_json_path(data_root=root_path, split='validation')


max_token_size = 1056
sample_size = int(input('sample size: '))
fetch_from_supabase = False
model, processor = load_plm_and_processor('saved_models/plm_1b_base')

train_synthdataset = SynthDogDataset(image_path=train_image_path,output_jsons_path=train_json_metadata, 
                                     image_feature_extractor=processor, 
                                     text_tokenizer=None, max_token_size=max_token_size, sample_size=sample_size, 
                                     read_images_from_supabase=fetch_from_supabase, split='train')
val_sample_size = int(input('Validation sample size: ')) 
val_synthdataset = SynthDogDataset(image_path=val_image_path,output_jsons_path=val_json_metadata, 
                                   image_feature_extractor=processor, 
                                   text_tokenizer=None, max_token_size=max_token_size, sample_size=val_sample_size, 
                                   read_images_from_supabase=fetch_from_supabase, split='validation') 

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # self-attention
        # "gate_proj", "up_proj", "down_proj"      # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()

model_config_version = 'v7'
run_name = input('Run name: ')
checkpoint_name = input('Checkpoint name: ')

if run_name == '':
    run_name = "plm_1b_finetune" 

if checkpoint_name == '':
    checkpoint_name = run_name

batch_size = int(input('Batch size: '))
run_name = run_name + f"_{sample_size if sample_size > 0 else 'all'}_samples_{model_config_version}"
    
wandb.init(project="ocr model", name=run_name)

lr = float(input('Learning rate: ')) # recommended : 1e-4, 5e-5 >=. 

num_epochs = int(input('Number of epochs : '))
eval_steps = 100
grad_accumulation = int(input("Grad accum: ")) 
print(f"Grad accumulation step: {grad_accumulation}")
steps_per_epoch = len(train_synthdataset)/(batch_size*1*grad_accumulation)
total_training_steps = int(steps_per_epoch * num_epochs)
if total_training_steps > 25000:
    eval_steps = 500

print(f'Total training steps: {total_training_steps}')
print(f'Eval steps & Save steps: {eval_steps}')
ckpt_path = 'checkpoints'
os.makedirs(ckpt_path, exist_ok=True)
max_grad_norm = float(input('Max Grad Norm: ')) # recommended : 1.0
num_beams = 1
use_deepspeed = True if input("Use deepspeed (y/n)?").strip().lower()[0] == 'y' else False
eval_strategy = 'epoch' if input("Save & Eval strategy (epoch/steps) | (e/s)").strip().lower()[0] == 'e' else 'steps'

print(f"Using deepspeed: {use_deepspeed}")
print(f"Eval strategy: {eval_strategy}")
if eval_strategy == 's':
    print(f"Eval steps: {eval_steps}")
training_args = Seq2SeqTrainingArguments(
        output_dir=f"./{ckpt_path}/{checkpoint_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=lr,  
        optim='adamw_torch',
        lr_scheduler_type="cosine",
        num_train_epochs=num_epochs,
        warmup_ratio=0.05,  
        logging_steps=50,
        logging_strategy="steps",
        save_total_limit=3,
        bf16=True,
        max_grad_norm=max_grad_norm,  
        
        weight_decay=0.01,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        predict_with_generate=True,
        generation_max_length=max_token_size,
        generation_num_beams=num_beams,
        report_to=["wandb"],
        run_name=run_name,
        save_safetensors=True,
        eval_strategy=eval_strategy,
        save_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,  
        greater_is_better=False,
        deepspeed="deepspeed_config.json" if use_deepspeed else None
        )



model.add_cross_attention = True
model.config.max_length = max_token_size
model.config.decoder.max_length = max_token_size
model.config.min_length = 1
model.config.decoder.min_length = 1
model.config.no_repeat_ngram_size = 0
model.config.repetition_penalty = 1.5
model.config.length_penalty = 1.0 
model.config.num_beams = num_beams
model.config.use_cache = False  
model.config.decoder.use_cache = False  
model.config.is_encoder_decoder = True
model.config.do_sample = False  
model.config.tie_word_embeddings = True

if num_beams > 1:
    model.config.early_stopping = True

print_trainable_prams(model)

early_stop = int(input('Early stopping: '))
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stop
)

# ovmodel.gradient_checkpointing()
peak_mem = torch.cuda.max_memory_allocated()
print(f"The model is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

print()
# ovmodel.gradient_checkpointing_enable()

def collate_fn_with_prompt(batch, pad_token_id=processor.tokenizer.pad_token_id):
    """
    Collates already-processed outputs with instruction, [IMG], image tensors, and label token ids.
    Does not re-tokenize prompt—just pads and batches.
    """
    # Each item: {'pixel_values', 'input_ids', 'attention_mask', 'labels', ...}
    pixel_values = []
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pixel_values.append(item['pixel_values'])
        input_ids.append(item['input_ids'])
        attention_mask.append(item['attention_mask'])
        labels.append(item['labels'])

    # Pad sequences (input_ids, attention_mask, labels) to max length in batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)

    # Mask labels padding token for loss computation
    labels_for_loss = labels_padded.clone()
    labels_for_loss[labels_padded == pad_token_id] = -100

    # Stack pixel_values if tensor shape matches (else use torch.cat or list as needed)
    pixel_values_batch = torch.stack(pixel_values)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'pixel_values': pixel_values_batch,
        'labels': labels_for_loss,
    }

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_synthdataset,
        eval_dataset=train_synthdataset,
        data_collator=collate_fn_with_prompt,
        tokenizer=processor.tokenizer,
        # compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
    )

trainer.train()
torch.distributed.destroy_process_group()
print('DONE')