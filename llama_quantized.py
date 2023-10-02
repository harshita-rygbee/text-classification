import time

import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class ListDataset(Dataset):
     def __init__(self, original_list):
        self.original_list = original_list
     def __len__(self):
        return len(self.original_list)

     def __getitem__(self, i):
        return self.original_list[i]



def get_data():

    train_df = pd.read_table('data/newsgroup_train_2_classes.csv').sample(frac=1)
    test_df = pd.read_table('data/newsgroup_test_2_classes.csv')
    
    readable_labels = {
        'soc.religion.christian': 'Christianity', 
        'sci.med': 'Medicine'
    }

    label_column_name = "label_name"
    train_df = train_df.rename(columns={label_column_name: "labels"})
    test_df = test_df.rename(columns={label_column_name: "labels"})
    train_df[label_column_name] = train_df[label_column_name].apply(lambda x: readable_labels[x])
    test_df[label_column_name] = test_df[label_column_name].apply(lambda x: readable_labels[x])
    label_names = list(train_df[label_column_name].unique())
    label_names_for_prompt = '", "'.join(label_names)
    intro = f'USER: You are given a list of labels: "{label_names_for_prompt}". Your task is to classify the following text into one of these labels. Here are some examples:\n'
    examples = '\n'.join('"' + train_df["query"] + '" => "' + train_df["labels"] + '"') + '\n\n'
    full_prompt = intro + examples
    # tests = (full_prompt + 'Please label the following query: "' + test_df['query'] + '"\n\nSYSTEM:').tolist()
    tests = (full_prompt + 'Please classify the query "' + test_df['text'] + '"into one of the given labels. Only reply with the label.\n\nSYSTEM:').tolist()
    return test_df, tests


def get_pipe(
    straight_pipe=False,
    llama_params=7,
    load_in_8bit=False,
    load_in_4bit=False,
    **kwargs
):
    # straight_pipe: load the pipeline directly from transformers,
    #                instead of defining the model and tokenizer separately
    #                to test 4 and 8 bit loading and other optional args

    llama_string = f"meta-llama/Llama-2-{llama_params}b-chat-hf"
    if straight_pipe:
        return transformers.pipeline(
            model=llama_string,
            task='text-generation',
            device_map='auto',
            # torch_dtype=torch.float16,
            **kwargs,
        )

    model = AutoModelForCausalLM.from_pretrained(
        llama_string,
        device_map='auto',
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,

        # torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(llama_string)

    return transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        device_map='auto',
        # torch_dtype=torch.float16,
        **kwargs,
    )


test_df, tests = get_data()
# dataset = ListDataset(tests)
pipe = get_pipe(straight_pipe=True, load_in_8bit=True)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

start = time.perf_counter()

results = pipe(
    tests,
    # dataset,
    temperature=0.001,
    do_sample=True,
    top_k=10,
    repetition_penalty=1.1,
    return_full_text=False,
    batch_size=16,
    num_workers=2,

)

print(time.perf_counter() - start, 'seconds')

test_df['raw_output'] = [i[0]['generated_text'].strip('\n').replace('\n', ' ') for i in results]
test_df.to_csv('raw_outputs.tsv', index=False, sep='\t')

labels = []
for i in results:
    text = i[0]['generated_text'].strip('\n').replace('\n', ' ')
    if text.startswith('Label: '):
        labels.append(text.split('Label: ')[-1])
    elif text.count('"') > 1:
        labels.append(text.split('"')[-2])
    else:
        labels.append('')
        print('no label found for:', text)

test_df['generated_labels'] = labels
test_df.to_csv('labels.tsv', index=False, sep='\t')

print('number of correct classes:', test_df[test_df['generated_labels']==test_df['labels']].shape[0])
