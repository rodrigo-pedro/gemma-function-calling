# gemma-function-calling

Repository containing the code used to train [gemma-2b-function-calling](https://huggingface.co/rodrigo-pedro/gemma-2b-function-calling). The model is a finetuned version of [gemma-2b-it](https://huggingface.co/google/gemma-2b-it) on the [hypervariance/function-calling-sharegpt](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt) dataset.

Code comes in script form and as a jupyter notebook. Also works with any other gemma model by changing the `model_name` variable in the script. Trained on 1 epoch.

## IMPORTANT - Requesting access

Before you can use the model, you must first to request access to the Gemma models in Hugging Face. You can do this by going to the [gemma-2b-it](https://huggingface.co/google/gemma-2b-it) model page and requesting access there.

Once you have access to the model, remember to authenticate with Hugging Face as described in this [guide](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication).

## Usage

Make sure you have the `peft` package installed. You can install it with `pip install peft`.

```python
from transformers import AutoModelForCausalLM , AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("rodrigo-pedro/gemma-2b-function-calling", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("rodrigo-pedro/gemma-2b-function-calling", trust_remote_code=True, device_map="auto")

inputs = tokenizer(prompt,return_tensors="pt").to(model.device)

outputs = model.generate(**inputs,do_sample=True,temperature=0.1,top_p=0.95,max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
```

You can also use sharegpt formatted prompts:

```python
from transformers import AutoModelForCausalLM , AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("rodrigo-pedro/gemma-2b-function-calling", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("rodrigo-pedro/gemma-2b-function-calling", trust_remote_code=True, device_map="auto")

chat = [
  {
      "from": "system",
      "value": "SYSTEM PROMPT",
  },
  {
      "from": "human",
      "value": "USER QUESTION"
  },
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs,do_sample=True,temperature=0.1,top_p=0.95,max_new_tokens=100)

print(tokenizer.decode(outputs[0]))
```

## Prompt template

```text
You are a helpful assistant with access to the following functions. Use them if required -
{
    "name": "function name",
    "description": "function description",
    "parameters": {
        "type": "type (object/number/string)",
        "properties": {
            "property_1": {
                "type": "type",
                "description": "property description"
            }
        },
        "required": [
            "property_1"
        ]
    }
}

To use these functions respond with:
<functioncall> {"name": "function_name", "arguments": {"arg_1": "value_1", "arg_1": "value_1", ...}} </functioncall>

Edge cases you must handle:
 - If there are no functions that match the user request, you will respond politely that you cannot help.

User Question:
USER_QUESTION
```

Function calls are enclosed in `<functioncall>` `</functioncall>`.

The model was trained using the same delimiters as [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it):

```text
<bos><start_of_turn>user
Write a hello world program<end_of_turn>
<start_of_turn>model
```

Use `<end_of_turn>` stop sequence to prevent the model from generating further text.
