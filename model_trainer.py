from glob import glob
import os 

import torch 
from func_utils.pydataloader import SynthDogDataset
from encoder_decoder_model import (
    init_dit_bart_models_fixed, add_lora_to_decoder, 
    load_pretrained_enc_dec_model, init_dit_dbart_models, load_pretrained_iprocessor_tokenizer
)


from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback

import wandb
import gc

from func_utils.trainer_utils import *

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
gc.collect()
wandb.login()

decode_to_portuguese = lambda x : x.replace('Ġ','').encode('iso-8859-1').decode('utf-8')

def get_synth_images_json_path(data_root= os.path.join('synthdog','outputs'), split='train'):
    ipath = os.path.join(data_root, '*', split, '*.jpg')
    json_path = os.path.join(data_root, '*', split, 'metadata.jsonl')

    return glob(ipath), glob(json_path)


torch.cuda.empty_cache()

root_path1 = os.path.join('synthdog', 'outputs_ol')
root_path2 = os.path.join('synth-text-generator', 'outputs')
root_choice_ind = int(input(f"""
Choose the root path: 
[1] - {root_path1}
[2] - {root_path2}
"""))
root_path = root_path1 if root_choice_ind == 1 else root_path2
train_image_path, train_json_metadata = get_synth_images_json_path(data_root=root_path, split='train')
val_image_path, val_json_metadata = get_synth_images_json_path(data_root=root_path, split='validation')

base_model_choice_ind = int(input(f"""
Choose the base model arch:
[1] - DiT + Bart
[2] - DiT + Donut-MBart
[3] - Dit768 + Donut-MBart
[4] - Base-Dit768 + Donut-MBart
"""))

base_model_path = None
if base_model_choice_ind == 1:
    processor, text_tokenizer = init_dit_bart_models_fixed(load_model=False)
elif base_model_choice_ind == 2:
    processor, text_tokenizer = init_dit_dbart_models(load_model=False)
elif base_model_choice_ind == 3:
    base_model_path = 'saved_models/dit768_dbart/'
    processor, text_tokenizer = load_pretrained_iprocessor_tokenizer(base_model_path)
elif base_model_choice_ind == 4:
    base_model_path = 'saved_models/mydit768_dbart/'
    processor, text_tokenizer = load_pretrained_iprocessor_tokenizer(base_model_path)
else:
    print("Wrong model choice. Quitting ...")
    exit()

max_token_size = 1056
sample_size = int(input('sample size: '))
fetch_from_supabase = False
train_synthdataset = SynthDogDataset(image_path=train_image_path,output_jsons_path=train_json_metadata, image_feature_extractor=processor, 
                                     text_tokenizer=text_tokenizer, max_token_size=max_token_size, sample_size=sample_size, 
                                     read_images_from_supabase=fetch_from_supabase, split='train')
val_sample_size = int(input('Validation sample size: ')) 
val_synthdataset = SynthDogDataset(image_path=val_image_path,output_jsons_path=val_json_metadata, image_feature_extractor=processor, 
                                   text_tokenizer=text_tokenizer, max_token_size=max_token_size, sample_size=val_sample_size, 
                                   read_images_from_supabase=fetch_from_supabase, split='validation') 

model_config_version = 'v8'
run_name = input('Run name: ')
checkpoint_name = input('Checkpoint name: ')

if run_name == '':
    run_name = "dit_bart" 

if checkpoint_name == '':
    checkpoint_name = run_name

batch_size = int(input('Batch size: '))
run_name = run_name + f"_{sample_size if sample_size > 0 else 'all'}_samples_{model_config_version}"
    
wandb.init(project="ocr model", name=run_name)

lr = float(input('Learning rate: ')) # recommended : 1e-4, 5e-5 >=. 

num_epochs = int(input('Number of epochs : '))
eval_steps = 100
grad_accumulation = 16 if base_model_choice_ind >= 3 else 1
print(f"Grad accumulation step: {grad_accumulation}")
steps_per_epoch = len(train_synthdataset)/(batch_size*1*grad_accumulation)
total_training_steps = int(steps_per_epoch * num_epochs)
if total_training_steps > 25000:
    eval_steps = 3500
save_steps = eval_steps

print(f'Total training steps: {total_training_steps}')
print(f'Eval steps & Save steps: {eval_steps}')
ckpt_path = 'checkpoints'
os.makedirs(ckpt_path, exist_ok=True)
max_grad_norm = float(input('Max Grad Norm: ')) # recommended : 10
num_beams = 1
use_deepspeed = True if input("Use deepspeed (y/n)?").strip().lower()[0] == 'y' else False
eval_strategy = 'epoch' if input("Save & Eval strategy (epoch/steps) | (e/s)").strip().lower()[0] == 'e' else 'steps'
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
        fp16=True if base_model_choice_ind == 3 else False,
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
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,  
        greater_is_better=False,
        deepspeed="deepspeed_config.json" if use_deepspeed else None
        )

load_model_choice = int(input("""
Load model type:
                          
[1] - Base Encoder & Decoder
[2] - Pre-trained Encoder & Decoder                          

>>> 
"""))

r=32
alpha=r*2
dropout=0.35
target_modules = [
        "q_proj", "k_proj", "v_proj", "out_proj"
]
modules_to_save = None

if load_model_choice == 1:
    if base_model_choice_ind == 1:
        _, _, ovmodel = init_dit_bart_models_fixed()
        ovmodel = add_lora_to_decoder(ovmodel, r=r, alpha=alpha, dropout=dropout, target_modules=target_modules, modules_to_save=modules_to_save)
    elif base_model_choice_ind == 2:
        _, _, ovmodel = init_dit_dbart_models()
    else:
        ovmodel = load_pretrained_enc_dec_model('saved_models/dit768_dbart/', None, None, lora_applied=False, device_map='cuda')
else:
    ckpt_path = input('Relative ckpt path: ')
    if base_model_choice_ind == 1:
        ovmodel = load_pretrained_enc_dec_model(ckpt_path, r=r, alpha=alpha, dropout=dropout, 
                                                target_modules=target_modules, modules_to_save=modules_to_save)
    elif base_model_choice_ind == 2:
        decoder = "naver-clova-ix/donut-base"
        ovmodel = load_pretrained_enc_dec_model(ckpt_path, base_encoder_model=None, 
                                             base_decoder_model="naver-clova-ix/donut-base", 
                                             lora_applied=False, 
                                             new_tokens=['Ã', 'Ê', 'Â']
                                            )
    else:
        ovmodel = load_pretrained_enc_dec_model(ckpt_path, None, None, lora_applied=False, device_map='cuda')

if model_config_version == 'v7':
    ovmodel = unfreeze_all_params(ovmodel, unfreeze_encoder=False, unfreeze_decoder=True)
    ovmodel = unfreeze_last_n_encoder(ovmodel, unfreeze_last_n_layer_block=1, unfreeze_attention_layers=True, skip_encoder=True, skip_decoder=True)

ovmodel.add_cross_attention = True
ovmodel.config.max_length = max_token_size
ovmodel.config.decoder.max_length = max_token_size
ovmodel.config.min_length = 1
ovmodel.config.decoder.min_length = 1
ovmodel.config.no_repeat_ngram_size = 0
ovmodel.config.repetition_penalty = 1.5
ovmodel.config.length_penalty = 1.0 
ovmodel.config.num_beams = num_beams
ovmodel.config.use_cache = False  
ovmodel.config.decoder.use_cache = False  
ovmodel.config.is_encoder_decoder = True
ovmodel.config.do_sample = False  
ovmodel.config.tie_word_embeddings = True
ovmodel.config.decoder.dropout = dropout
ovmodel.config.decoder.attention_dropout = 0.15
ovmodel.config.decoder.decoder_layerdrop = 0.1
if num_beams > 1:
    ovmodel.config.early_stopping = True

if model_config_version == 'v8':
    print_trainable_prams(ovmodel)

early_stop = int(input('Early stopping: '))
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=early_stop
)

# ovmodel.gradient_checkpointing()
peak_mem = torch.cuda.max_memory_allocated()
print(f"The model is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

print()
# ovmodel.gradient_checkpointing_enable()

trainer = setup_dit_bart_training(
        train_synthdataset, val_synthdataset, training_args=training_args, model=ovmodel, 
        text_tokenizer=text_tokenizer,
        run_name = run_name, 
        callbacks=[early_stopping_callback],
        max_length=max_token_size
    )

# save_model_path = 'saved_models'
# os.makedirs(save_model_path, exist_ok=True)
# model_save_path = f"{save_model_path}/{run_name}_final_model"
# try:
#     history = trainer.train()
#     trainer.save_model(model_save_path)
# except Exception as e:
#     print(e)
#     trainer.save_model(save_model_path)

trainer.train()
torch.distributed.destroy_process_group()
print('DONE')