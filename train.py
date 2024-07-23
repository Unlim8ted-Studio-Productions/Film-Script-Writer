def main(): 
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
    from datasets import load_from_disk
    import torch
    from transformers import AdamW, get_linear_schedule_with_warmup

    # Load the dataset from disk
    dataset = load_from_disk('dataset')

    # Split the dataset into train and test sets
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')  # Use a smaller model instead of the previouse gpt-2

    # GPT-2 doesn't have a padding token by default, so we set it to the eos_token
    tokenizer.pad_token = tokenizer.eos_token

    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.config.pad_token_id = tokenizer.pad_token_id

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the text with padding and truncation
        tokenized_text = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512  # Set to a smaller length for quicker training
        )
        # Set labels to be the same as input_ids for language modeling
        tokenized_text['labels'] = tokenized_text['input_ids']
        return tokenized_text

    # Apply the tokenization function to the train and test datasets with multiprocessing
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Increase batch size
        per_device_eval_batch_size=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_accumulation_steps=8,  # Accumulate gradients
    )

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(tokenized_train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=total_steps)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        optimizers=(optimizer, scheduler)  # Pass optimizer and scheduler
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('fine_tuned_gpt2')
    tokenizer.save_pretrained('fine_tuned_gpt2')

if __name__ == "__main__":
    main()


#def main():
#    import torch
#    from transformers import (
#        GPT2LMHeadModel,
#        GPT2Tokenizer,
#        Trainer,
#        TrainingArguments,
#        DataCollatorForLanguageModeling,
#        get_linear_schedule_with_warmup
#    )
#    from datasets import load_from_disk
#    from tqdm import tqdm
#    import os
#
#    # Load the dataset from disk
#    print('\033[1m' +"Loading dataset from disk\033[0m")
#    with tqdm(total=1, desc="Loading Dataset") as pbar:
#        dataset = load_from_disk('dataset')
#        pbar.update(1)
#    print('\033[1m' +"Finished loading dataset.\n\033[0m")
#
#    # Split the dataset into train and test sets
#    print('\033[1m' +"Splitting dataset into train and test sets\033[0m")
#    with tqdm(total=1, desc="Splitting Dataset") as pbar:
#        train_test_split = dataset.train_test_split(test_size=0.1)
#        train_dataset = train_test_split['train']
#        test_dataset = train_test_split['test']
#        pbar.update(1)
#    print("Finished splitting dataset.\n\033[0m")
#
#    # Load the tokenizer
#    print("\033[1m Loading tokenizer\033[0m")
#    with tqdm(total=1, desc="Loading Tokenizer") as pbar:
#        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#        tokenizer.pad_token = tokenizer.eos_token
#        pbar.update(1)
#    print("Finished loading tokenizer.\n\033[0m")
#
#    # Load the GPT-2 model
#    print('\033[1m' +"Loading GPT-2 model\033[0m")
#    with tqdm(total=1, desc="Loading Model") as pbar:
#        model = GPT2LMHeadModel.from_pretrained('gpt2')
#        pbar.update(1)
#    print("Finished loading GPT-2 model.\n\033[0m")
#
#    with tqdm(total=2, desc="\033[1mConfiguring Model\033[0m") as pbar:
#        # Set pad token to eos token
#        model.config.pad_token_id = tokenizer.pad_token_id
#        pbar.update(1)
#
#        # Enable gradient checkpointing
#        model.gradient_checkpointing_enable()
#        pbar.update(1)
#    print('\n')
#
#    # Tokenize the dataset with dynamic padding
#    def tokenize_function_dynamic_padding(examples):
#        tokenized_text = tokenizer(examples['text'], truncation=True)
#        tokenized_text['labels'] = tokenized_text['input_ids'].copy()
#        return tokenized_text
#
#    # Apply tokenization without max_length for dynamic padding
#    print('\033[1m' +"Tokenizing training dataset\033[0m")
#    # Use tqdm to display the progress bar for the tokenization process
#    tokenized_train_dataset = train_dataset.map(
#        tokenize_function_dynamic_padding,
#        batched=True,
#        desc="Tokenizing Training Data",  # Adding description for the progress bar
#    )
#    print("Finished tokenizing training dataset.\n\033[0m")
#
#    print('\033[1m' +"Tokenizing evaluation dataset\033[0m")
#    # Use tqdm to display the progress bar for the tokenization process
#    tokenized_test_dataset = test_dataset.map(
#        tokenize_function_dynamic_padding,
#        batched=True,
#        desc="Tokenizing Evaluation Data",  # Adding description for the progress bar
#    )
#    print("Finished tokenizing evaluation dataset.\n\033[0m")
#
#    # Data collator for dynamic padding
#    print('\033[1m' +"Creating data collator")
#    with tqdm(total=1, desc="Creating Data Collator") as pbar:
#        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#        pbar.update(1)
#    print("Finished creating data collator.\n\033[0m")
#
#    print('\033[1m' +"Creating trainer instance\033[0m")
#    # Define the training arguments with optimizations
#    with tqdm(total=2, desc="Initializing Trainer Arguments") as pbar:
#        training_args = TrainingArguments(
#            output_dir='./results',                  # Output directory for model checkpoints
#            overwrite_output_dir=True,               # Overwrite the content of the output directory
#            num_train_epochs=3,                      # Number of training epochs
#            per_device_train_batch_size=8,           # Batch size per device during training
#            per_device_eval_batch_size=8,            # Batch size for evaluation
#            gradient_accumulation_steps=4,           # Accumulate gradients to simulate larger batch size
#            warmup_steps=200,                        # Fewer warmup steps
#            weight_decay=0.01,                       # Weight decay for regularization
#            logging_dir='./logs',                    # Directory for storing logs
#            logging_steps=10,                        # Log every 10 steps
#            evaluation_strategy="steps",             # Evaluate every `eval_steps` steps
#            eval_steps=500,                          # Evaluation step interval
#            save_steps=1000,                         # Save checkpoint every `save_steps`
#            save_total_limit=2,                      # Limit the total number of checkpoints
#            load_best_model_at_end=True,             # Load the best model when finished training
#            dataloader_num_workers=8,                # Use multiple workers for data loading
#            optim="adamw_torch",                     # Use the faster AdamW optimizer from PyTorch
#        )
#
#        pbar.update(1)
#        pbar.desc="Initializing Trainer Instance"
#        # Initialize the Trainer
#        trainer = Trainer(
#            model=model,                             # The GPT-2 model to train
#            args=training_args,                      # Training arguments
#            train_dataset=tokenized_train_dataset,   # Subset for training
#            eval_dataset=tokenized_test_dataset,     # Subset for evaluation
#            data_collator=data_collator,             # Use dynamic padding data collator
#        )
#        pbar.update(1)
#    print("Finished creating trainer instance.\n\033[0m")
#
#    print('\033[1m' +"Training\033[1m")
#    # Train the model with tqdm progress bar
#    # This step will automatically display a progress bar thanks to the Trainer's built-in logging
#    trainer.train()
#    print("Finished training.\n")
#
#    # Save the fine-tuned model and tokenizer
#    print('\033[1m' +"Saving model and tokenizer\033[0m")
#    with tqdm(total=1, desc="Saving Model and Tokenizer") as pbar:
#        model.save_pretrained('fine_tuned_gpt2')
#        tokenizer.save_pretrained('fine_tuned_gpt2')
#        pbar.update(1)
#    print("Finished saving model and tokenizer.\n")
#    
#if __name__ == "__main__":
#    main()