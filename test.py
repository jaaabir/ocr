from encoder_decoder_model import load_pretrained_enc_dec_model
import os 


base_model_path = 'saved_models/mydit768_dbart/'
ckpt = 'checkpoints/bdit_dbart/checkpoint-59500'

print(f"base model path : {os.path.exists(base_model_path)}")
print(f"ckpt path : {os.path.exists(ckpt)}")


print(load_pretrained_enc_dec_model(ckpt, None, None, lora_applied=False))

