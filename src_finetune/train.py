from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load your instruction NL-to-SQL datasets in JSONL format
data_files = {
    "train": "fine_tune_dataset.jsonl",
    "validation": "validation_dataset.jsonl"
}
datasets = load_dataset("json", data_files=data_files)

# Tokenization function for instruction tuning format
def tokenize_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(tokenize_function, batched=True)

# Training arguments configuration
training_args = TrainingArguments(
    output_dir="./llama3.2-3b-instruct-nl2sql",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none"  # Set to your preferred logging integration or "none"
)

# Trainer instantiation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Start fine-tuning
trainer.train()

# Save LoRA adapters and fine-tuned model weights
model.save_pretrained("./llama3.2-3b-instruct-nl2sql")
