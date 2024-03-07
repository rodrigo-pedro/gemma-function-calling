import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments,
)

from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset



model_name = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True)

# We can't use multi line strings for the chat template because newlines and tabs get interpreted. We have to do this wacky string instead.
# https://huggingface.co/docs/transformers/main/en/chat_templating#notes-on-whitespace
# BOS automatically added https://huggingface.co/docs/transformers/en/model_doc/gemma#transformers.GemmaTokenizer.bos_token
# Uses same prompt format as gemma-2b/7b-it
# System prompt part of user prompt, due to the model not having a system prompt
chat_template = \
"{% for message in messages %}" \
    "{% if message['from'] == 'system' %}" \
        "{{ '<start_of_turn>user\\n' + message['value'] }}" \
    "{% elif message['from'] == 'human' %}" \
        "{% if loop.index0 == 1 %}" \
            "{{ '\\nUser Question:\\n' }}" \
        "{% else %}" \
            "{{ '<start_of_turn>user\\n' }}" \
        "{% endif %}" \
        "{{ message['value'] + '<end_of_turn>' }}" \
    "{% elif message['from'] == 'gpt' %}" \
        "{{ '<start_of_turn>model\\n'  + message['value'] + ' ' + '<end_of_turn>' }}" \
    "{% elif message['from'] == 'function_response' %}" \
        "{{ '<start_of_turn>user\\n'  + message['value'] + ' ' + '<end_of_turn>' }}" \
    "{% endif %}" \
    "{% if not loop.last %}" \
        "{{ '\\n' }}" \
    "{% endif %}" \
"{% endfor %}" \
"{% if not add_generation_prompt is defined %}" \
    "{% set add_generation_prompt = false %}" \
"{% endif %}" \
"{% if add_generation_prompt %}" \
    "{{ '\\n<start_of_turn>model\\n' }}" \
"{% endif %}"

tokenizer.chat_template = chat_template

dataset_train = load_dataset("hypervariance/function-calling-sharegpt", split="train")
dataset_train = dataset_train.map(
    lambda x: {
        "formatted_chat": tokenizer.apply_chat_template(
            x["conversations"], tokenize=False,
        )
    }
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", quantization_config=bnb_config, attn_implementation="flash_attention_2"
)

model.config.use_cache = False  # Disable cache for fine-tuning. We always want to use the most recent values.
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,  # 64
    bias="none",
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    fp16=False,
    bf16=True,
    save_strategy="epoch",  # save at end of epoch. Alternative is "steps", which saves at every "save_steps"
    logging_steps=1,
    learning_rate=2e-4,
    group_by_length=True,
    lr_scheduler_type="constant",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset_train,
    peft_config=peft_config,
    max_seq_length=2048,
    dataset_text_field="formatted_chat",
    args=training_args,
    packing=True,
)

trainer.train()
trainer.save_model("gemma-2b-function-calling")
