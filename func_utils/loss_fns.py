import torch
import torch.nn as nn
import torch.nn.functional as F
from difflib import SequenceMatcher
import numpy as np

class CRMLoss(nn.Module):
    """
    Character Reconstruction Matching Loss
    Focuses on character-level accuracy with position and frequency awareness
    """
    def __init__(self, tokenizer, temperature=1.0, char_freq_penalty=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.char_freq_penalty = char_freq_penalty
        
        # Build character frequency map for penalty
        if char_freq_penalty:
            self._build_char_freq_map()
    
    def _build_char_freq_map(self):
        """Build character frequency map from vocabulary"""
        vocab = self.tokenizer.get_vocab()
        char_counts = {}
        total_chars = 0
        
        for token in vocab.keys():
            for char in token:
                char_counts[char] = char_counts.get(char, 0) + 1
                total_chars += 1
        
        # Convert to probabilities and create penalty weights
        self.char_penalties = {}
        for char, count in char_counts.items():
            freq = count / total_chars
            # Rare characters get higher penalty
            self.char_penalties[char] = 1.0 / (freq + 1e-6)
    
    def compute_character_matching_score(self, pred_text, target_text):
        """Compute character-level matching with position awareness"""
        if not target_text:
            return torch.tensor(1.0 if not pred_text else 0.0)
        
        # Character-level alignment using SequenceMatcher
        matcher = SequenceMatcher(None, pred_text, target_text)
        matching_blocks = matcher.get_matching_blocks()
        
        total_match_score = 0.0
        target_len = len(target_text)
        
        for match in matching_blocks:
            if match.size > 0:
                # Position weight (earlier positions more important)
                pos_weight = np.exp(-0.1 * match.b)  # b is position in target
                
                # Character frequency penalty
                char_penalty = 1.0
                if self.char_freq_penalty:
                    chars_in_match = target_text[match.b:match.b + match.size]
                    char_penalty = np.mean([
                        self.char_penalties.get(c, 1.0) for c in chars_in_match
                    ])
                
                match_score = (match.size / target_len) * pos_weight * char_penalty
                total_match_score += match_score
        
        # Normalize and convert to loss
        similarity = min(1.0, total_match_score)
        return torch.tensor(1.0 - similarity, dtype=torch.float32)
    
    def forward(self, logits, labels):
        """Forward pass computing CRM loss"""
        batch_size = logits.size(0)
        device = logits.device
        
        predictions = torch.argmax(logits, dim=-1)
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_text = self.tokenizer.decode(predictions[i], skip_special_tokens=True)
            target_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)
            
            char_loss = self.compute_character_matching_score(pred_text, target_text)
            total_loss += char_loss
        
        return torch.tensor(total_loss / batch_size, device=device, requires_grad=True)


class WRMLoss(nn.Module):
    """
    Word Reconstruction Matching Loss
    Focuses on word-level accuracy with semantic similarity
    """
    def __init__(self, tokenizer, word_importance_decay=0.1, case_sensitive=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.word_importance_decay = word_importance_decay
        self.case_sensitive = case_sensitive
    
    def compute_word_similarity(self, word1, word2):
        """Compute similarity between two words"""
        if not self.case_sensitive:
            word1, word2 = word1.lower(), word2.lower()
        
        if word1 == word2:
            return 1.0
        
        # Character-level similarity for partial matches
        matcher = SequenceMatcher(None, word1, word2)
        return matcher.ratio()
    
    def compute_word_matching_score(self, pred_words, target_words):
        """Compute word-level matching with position importance"""
        if not target_words:
            return torch.tensor(1.0 if not pred_words else 0.0)
        
        total_score = 0.0
        max_len = max(len(pred_words), len(target_words))
        
        # Align words and compute similarity
        for i in range(max_len):
            # Position importance (earlier words more important)
            pos_weight = np.exp(-self.word_importance_decay * i)
            
            if i < len(target_words):
                target_word = target_words[i]
                
                if i < len(pred_words):
                    pred_word = pred_words[i]
                    word_sim = self.compute_word_similarity(pred_word, target_word)
                else:
                    word_sim = 0.0  # Missing word
                
                total_score += word_sim * pos_weight
        
        # Normalize by target length
        avg_score = total_score / len(target_words) if target_words else 0.0
        return torch.tensor(1.0 - avg_score, dtype=torch.float32)
    
    def forward(self, logits, labels):
        """Forward pass computing WRM loss"""
        batch_size = logits.size(0)
        device = logits.device
        
        predictions = torch.argmax(logits, dim=-1)
        total_loss = 0.0
        
        for i in range(batch_size):
            pred_text = self.tokenizer.decode(predictions[i], skip_special_tokens=True)
            target_text = self.tokenizer.decode(labels[i], skip_special_tokens=True)
            
            pred_words = pred_text.split()
            target_words = target_text.split()
            
            word_loss = self.compute_word_matching_score(pred_words, target_words)
            total_loss += word_loss
        
        return torch.tensor(total_loss / batch_size, device=device, requires_grad=True)


class HybridOCRLoss(nn.Module):
    """
    Hybrid OCR Loss combining CE, CRM, WRM, and additional OCR-specific objectives
    """
    def __init__(self, tokenizer, loss_config=None):
        super().__init__()
        self.tokenizer = tokenizer
        
        # Default loss configuration
        self.config = loss_config or {
            'ce_weight': 1.0,
            'crm_weight': 0.4,
            'wrm_weight': 0.6,
            'length_penalty_weight': 0.1,
            'confidence_penalty_weight': 0.05
        }
        
        # Initialize loss components
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.crm_loss = CRMLoss(tokenizer)
        self.wrm_loss = WRMLoss(tokenizer)
    
    def compute_length_penalty(self, predictions, targets):
        """Penalize significant length differences"""
        batch_size = predictions.size(0)
        total_penalty = 0.0
        
        for i in range(batch_size):
            pred_len = (predictions[i] != self.tokenizer.pad_token_id).sum().float()
            target_len = (targets[i] != self.tokenizer.pad_token_id).sum().float()
            
            if target_len > 0:
                length_ratio = pred_len / target_len
                # Penalty increases quadratically with length difference
                penalty = torch.abs(length_ratio - 1.0) ** 2
                total_penalty += penalty
        
        return total_penalty / batch_size
    
    def compute_confidence_penalty(self, logits):
        """Penalize overconfident predictions on wrong tokens"""
        # Get prediction confidence
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        # Penalty for very high confidence (overconfidence)
        overconfidence_penalty = torch.mean(
            torch.relu(max_probs - 0.95)  # Penalty for >95% confidence
        )
        
        return overconfidence_penalty
    
    def forward(self, logits, labels):
        """Forward pass combining all loss components"""
        device = logits.device
        predictions = torch.argmax(logits, dim=-1)
        
        # Base cross-entropy loss
        ce_loss = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # CRM and WRM losses
        crm_loss = self.crm_loss(logits, labels)
        wrm_loss = self.wrm_loss(logits, labels)
        
        # Additional penalties
        length_penalty = self.compute_length_penalty(predictions, labels)
        confidence_penalty = self.compute_confidence_penalty(logits)
        
        # Combine all losses
        total_loss = (
            self.config['ce_weight'] * ce_loss +
            self.config['crm_weight'] * crm_loss +
            self.config['wrm_weight'] * wrm_loss +
            self.config['length_penalty_weight'] * length_penalty +
            self.config['confidence_penalty_weight'] * confidence_penalty
        )
        
        return total_loss


# Correct way to use custom loss with Seq2SeqTrainer
def create_custom_loss_function(tokenizer, loss_config=None):
    """Create a custom loss function that can be passed to compute_loss_func"""
    
    # Default loss configuration
    config = loss_config or {
        'ce_weight': 1.0,
        'crm_weight': 0.3,
        'wrm_weight': 0.5,
        'length_penalty_weight': 0.1,
        'confidence_penalty_weight': 0.05
    }
    
    # Initialize loss components
    crm_loss = CRMLoss(tokenizer)
    wrm_loss = WRMLoss(tokenizer)
    ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    def custom_loss_function(model, inputs):
        """
        Custom loss function compatible with Seq2SeqTrainer.compute_loss_func
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        device = logits.device
        predictions = torch.argmax(logits, dim=-1)
        
        # Base cross-entropy loss
        ce_loss_val = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # CRM and WRM losses
        crm_loss_val = crm_loss(logits, labels)
        wrm_loss_val = wrm_loss(logits, labels)
        
        # Additional penalties
        length_penalty = compute_length_penalty(predictions, labels, tokenizer)
        confidence_penalty = compute_confidence_penalty(logits)
        
        # Combine all losses
        total_loss = (
            config['ce_weight'] * ce_loss_val +
            config['crm_weight'] * crm_loss_val +
            config['wrm_weight'] * wrm_loss_val +
            config['length_penalty_weight'] * length_penalty +
            config['confidence_penalty_weight'] * confidence_penalty
        )
        
        return total_loss
    
    return custom_loss_function

def compute_length_penalty(predictions, targets, tokenizer):
    """Helper function for length penalty"""
    batch_size = predictions.size(0)
    total_penalty = 0.0
    
    for i in range(batch_size):
        pred_len = (predictions[i] != tokenizer.pad_token_id).sum().float()
        target_len = (targets[i] != tokenizer.pad_token_id).sum().float()
        
        if target_len > 0:
            length_ratio = pred_len / target_len
            penalty = torch.abs(length_ratio - 1.0) ** 2
            total_penalty += penalty
    
    return total_penalty / batch_size

def compute_confidence_penalty(logits):
    """Helper function for confidence penalty"""
    probs = F.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1)[0]
    overconfidence_penalty = torch.mean(torch.relu(max_probs - 0.95))
    return overconfidence_penalty


# Usage example with Seq2SeqTrainer
def setup_custom_ocr_training(model, tokenizer, train_dataset, eval_dataset, training_args):
    """Setup Seq2SeqTrainer with custom CRM/WRM losses"""
    
    # Configure custom loss
    loss_config = {
        'ce_weight': 1.0,
        'crm_weight': 0.3,      # Character-level matching
        'wrm_weight': 0.5,      # Word-level matching  
        'length_penalty_weight': 0.1,
        'confidence_penalty_weight': 0.05
    }
    
    # Create custom loss function
    custom_loss_func = create_custom_loss_function(tokenizer, loss_config)
    
    # Create Seq2SeqTrainer with custom loss
    from transformers import Seq2SeqTrainer
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_loss_func=custom_loss_func  # THIS is the correct parameter name
    )
    
    return trainer