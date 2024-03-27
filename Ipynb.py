def tokenize_fn(batch):
    # Tokenize sentences
    inputs = tokenizer(batch['sentence'], truncation=True)

    # Tokenize and pad labels
    labels = tokenizer(batch['label'], padding='max_length', truncation=True, return_tensors='pt')
    
    # Return tokenized inputs and labels
    return {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': labels.input_ids.squeeze()  # Squeeze to remove the extra dimension
    }
