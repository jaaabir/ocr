import torch
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import gc

def setup_memory_efficient_training():
    """
    Setup memory-efficient training for PerceptionLM.
    """
    
    # Memory optimization techniques
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

def get_memory_efficient_training_args():
    """
    Training arguments optimized for memory efficiency with PerceptionLM.
    """
    return TrainingArguments(
        output_dir="./perceptionlm_outputs",
        per_device_train_batch_size=1,  # Start with 1, increase if possible
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=1e-4,  # Lower learning rate for stability
        num_train_epochs=3,
        warmup_steps=200,
        logging_steps=50,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        save_total_limit=2,
        
        # Aggressive memory optimization
        fp16=True,
        bf16=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        
        # Essential for large models
        gradient_checkpointing=True,
        
        # Additional memory optimizations
        max_grad_norm=1.0,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        optim="adamw_torch",  # Use PyTorch AdamW instead of HF version
        
        # Disable some features to save memory
        prediction_loss_only=True,  # Don't compute other metrics during training
        load_best_model_at_end=False,  # Disable to save memory
        
        # Report memory usage
        report_to=[],  # Disable wandb/tensorboard to save memory
    )

class MemoryEfficientDataLoader:
    """
    Custom data loader with memory management.
    """
    def __init__(self, dataset, batch_size=1, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            torch.manual_seed(42)  # For reproducibility
            indices = torch.randperm(len(indices)).tolist()
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            
            # Clear cache after each batch
            if i % 10 == 0:  # Every 10 batches
                torch.cuda.empty_cache()
                gc.collect()
                
            yield self.collate_fn(batch)
    
    def collate_fn(self, batch):
        # Your existing collate function here
        pass

def train_with_memory_management(model, train_dataset, eval_dataset):
    """
    Training function with explicit memory management.
    """
    
    # Setup memory-efficient training
    setup_memory_efficient_training()
    
    # Get training arguments
    training_args = get_memory_efficient_training_args()
    
    # Custom trainer with memory management
    class MemoryEfficientTrainer(Trainer):
        def training_step(self, model, inputs):
            """
            Override training step with memory management.
            """
            # Clear cache before each step
            torch.cuda.empty_cache()
            
            # Standard training step
            model.train()
            inputs = self._prepare_inputs(inputs)
            
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            if self.args.n_gpu > 1:
                loss = loss.mean()
                
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss = loss / self.args.gradient_accumulation_steps
                
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                
            # Clear cache after backward pass
            torch.cuda.empty_cache()
            
            return loss.detach() / self.args.gradient_accumulation_steps
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=your_collate_fn,  # Your existing collate function
    )
    
    # Train with memory monitoring
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU out of memory. Try reducing batch size further or using gradient checkpointing.")
            # Clear cache and try again with even smaller batch
            torch.cuda.empty_cache()
            gc.collect()
        raise e
    
    return trainer

# Usage example:
# trainer = train_with_memory_management(perception_model, train_dataset, eval_dataset)