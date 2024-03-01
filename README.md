# gemma-function-calling

Repository containing the code used to finetune [gemma-2b-it](https://huggingface.co/google/gemma-2b-it) on the [hypervariance/function-calling-sharegpt](https://huggingface.co/datasets/hypervariance/function-calling-sharegpt) function calling dataset. Also works with any other gemma model. Comes in script form and as a jupyter notebook. Trained on 1 epoch.

## Prompt template

```
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
<USER_QUESTION>
```
