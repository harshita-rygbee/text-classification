import os
import time
from functools import partial
import json
from box import Box

import bitsandbytes as bnb
import pandas as pd
import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

start = time.perf_counter()
seed = 33

llama_label_mapping = {
   "anger": "Anger",
   "joy": "Joy",
   "optimism": "Optimism",
   "sadness": "Sadness"
}

label_names_for_prompt = ", ".join(llama_label_mapping.values())
DEFAULT_PROMPT = f"You are given a list of labels: {label_names_for_prompt}. Your task is to classify the following tweets into one of these labels."
# print(DEFAULT_PROMPT)


def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """

    # Get number of GPU device and set maximum memory
    # n_gpus = torch.cuda.device_count()
    # max_memory = f'{40960}MB'

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch the model efficiently on the available resources
        # max_memory={i: max_memory for i in range(n_gpus)},
    )

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token as EOS token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_our_data(
    path='train.csv',
    prompt=DEFAULT_PROMPT,
    sep=',',
    head=None,
):

    table = pd.read_csv(path, sep=sep).sample(frac=1)
    if head is not None and head != 'none':
        table = table.head(head)

    table['instruction'] = prompt
    table['input'] = table['text']
    table['output'] = table['labels'].apply(lambda x: llama_label_mapping[x])
    return Dataset.from_pandas(table[['instruction', 'input', 'output']], split="train")


def create_prompt_formats(sample):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """

    # Initialize static strings for the prompt template
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    END_KEY = "### End"

    # Combine a prompt with the static strings
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['input']}" if sample["input"] else None
    response = f"{RESPONSE_KEY}\n{sample['output']}"
    end = f"{END_KEY}"

    # Create a list of prompt template elements
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    # Join prompt template elements into a single string to create the prompt template
    formatted_prompt = "\n\n".join(parts)

    # Store the formatted prompt template in a new key "text"
    sample["text"] = formatted_prompt

    return sample


def get_max_length(model):
    """
    Extracts maximum token length from the model configuration

    :param model: Hugging Face model
    """

    # Initialize a "max_length" variable to store maximum sequence length as null
    max_length = None
    # Find maximum sequence length in the model configuration and save it in "max_length" if found
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    # Set "max_length" to 1024 (default value) if maximum sequence length is not found in the model configuration
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
    dataset: Dataset,
):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset (Dataset): Instruction dataset
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)

    # Apply preprocessing to each batch of the dataset & and remove "instruction", "input", "output", and "text" fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"]
    )

    # Filter out samples that have "input_ids" exceeding "max_length"
    length = len(dataset)
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    print(f'Filtered {length - len(dataset)}/{length} samples')
    return dataset.shuffle(seed=seed)


def find_all_linear_names(model):
    """
    Find modules to apply LoRA to.

    :param model: PEFT model
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )


def prepare_peft_model(model, lora_config):
    """ Prepares the pre-trained model for fine-tuning """

    # Enable gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)

    # Get LoRA module names
    target_modules = find_all_linear_names(model)

    # Create PEFT configuration for these modules and wrap the model to PEFT
    peft_config = LoraConfig(target_modules=target_modules, **lora_config)

    return get_peft_model(model, peft_config)


def fine_tune(
    model,
    tokenizer,
    training_args,
    train_dataset,
    eval_dataset,
):
    """ Prepares and fine-tunes the pre-trained model """

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False

    # Launch training and log metrics
    print("Training...")

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    print(train_result.metrics)

    if eval_dataset is not None:
        eval_result = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
        print(eval_result)

    trainer.save_state()

    # Save model
    print("Saving last checkpoint of the model...")
    os.makedirs(training_args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(training_args.output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == '__main__':

    with open('config.json', 'r') as f:
        config = Box(json.load(f))

    model_name = f"meta-llama/Llama-2-{config.llama_num_params}b-hf"
    bnb_config = BitsAndBytesConfig(**config.bitsandbytes)
    model, tokenizer = load_model(model_name, bnb_config)

    max_length = get_max_length(model)

    # Load train and eval datasets
    train_dataset = load_our_data(
        os.path.join(config.datapath, config.train_file),
        sep='\t',
        head=config.top_n_samples,
    )
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, train_dataset)
    print(f'Number of train prompts: {len(train_dataset)}')
    print(f'Column names are: {train_dataset.column_names}')

    if config.trainer.evaluation_strategy != "none":
        eval_dataset = load_our_data(
            os.path.join(config.datapath, config.eval_file),
            sep='\t',
            head=config.top_n_samples,
        )
        eval_dataset = preprocess_dataset(tokenizer, max_length, seed, eval_dataset)
        print(f'Number of eval prompts: {len(eval_dataset)}')
    else:
        eval_dataset = None

    model = prepare_peft_model(model=model, lora_config=config.LoRA)

    training_args = TrainingArguments(output_dir=config.output_dir, **config.trainer)

    print(f"Data and model prepared in {time.perf_counter() - start} seconds, starting fine-tuning")

    fine_tune(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Load fine-tuned weights
    model = AutoPeftModelForCausalLM.from_pretrained(
        training_args.output_dir,
        device_map="auto",
        torch_dtype=config.bitsandbytes.bnb_4bit_compute_dtype
    )
    # Merge the LoRA layers with the base model
    model = model.merge_and_unload()

    print("about to save model")
    # Save fine-tuned model at a new location
    output_merged_dir = os.path.join(config.output_dir, f"classification_{config.llama_num_params}")
    os.makedirs(output_merged_dir, exist_ok=False)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # Save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_merged_dir)
