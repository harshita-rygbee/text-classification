import argparse
import configparser
import json
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from tqdm import trange
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from itertools import chain
import random
import numpy as np

def seed_everything(seed : int) :
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() :
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train text classifier")
    parser.add_argument('--model', type=str, help='model config name')
    parser.add_argument('--dataset', type=str, help='dataset config name')
    parser.add_argument('--max_length', type=int, default=10, help='max length of input')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    return parser.parse_args()


def get_train_val_test_df(config_name='tripclick', config_file_name='data_config.ini', label_field='class_labels'):
    cp_data = configparser.ConfigParser()
    cp_data.read(config_file_name)
    cp_data = cp_data[config_name]
    data_folder = cp_data['data_folder']
    print(f"Data folder: {data_folder}")

    dfs = []
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(data_folder, cp_data[f'{split}_file'])
        df = pd.read_table(file_path).dropna()

        # This only works with the original files that had the labels as a string
        df['labels'] = df[label_field].apply(lambda x: x.split(","))
        # df['labels'] = df[label_field].apply(eval)
        df = df.drop(label_field, axis=1)

        dfs.append(df)
    return dfs


def get_dataloader(df, label_binarizer, tokenizer, max_length, batch_size, split='train', text_field='search_term'):
    df['one_hot_labels'] = label_binarizer.transform(df['labels']).tolist()
    encodings = tokenizer(df[text_field].tolist(), max_length=max_length, pad_to_max_length=True, return_tensors='pt')
    data = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(list(df.one_hot_labels.values)),
    )
    sampler = SequentialSampler(data) if split == 'train' else SequentialSampler(data)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)


def training_loop(model, train_loader, optimizer, loss_func, device, num_labels):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_loader):
        if step % 5000 == 0:
            print(f"Step {step} of {len(train_loader)}")
        optimizer.zero_grad()

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        loss = loss_func(logits.view(-1, num_labels), b_labels.type_as(logits).view(-1, num_labels))
        loss.backward()
        optimizer.step()
        # scheduler.step()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print(f"Train loss: {tr_loss / nb_tr_steps}", flush=True)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    logit_preds, true_labels, pred_labels = [], [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outs = model(b_input_ids, attention_mask=b_input_mask)
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred).to('cpu').numpy()

        b_logit_pred = b_logit_pred.detach().cpu().numpy()
        b_labels = b_labels.to('cpu').numpy()

        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    return pred_labels, true_labels


def get_prediction_for_query(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
        # predicted_class_id = logits.argmax().item()
        # return model(**inputs)
        return logits


def get_metrics(pred_labels, true_labels, threshold=0.5):
    pred_bools = [pl > threshold for pl in pred_labels]
    true_bools = [tl == 1 for tl in true_labels]
    f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
    flat_accuracy = accuracy_score(true_bools, pred_bools) * 100
    print()
    print(f'F1   Accuracy: {f1_accuracy}')
    print(f'Flat Accuracy: {flat_accuracy}')


def main():
    freeze_bert = False
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = get_train_val_test_df(args.dataset)
    # num_labels = train_df['labels'].nunique()
    label_names = list(set(list(chain.from_iterable(train_df['labels']))))
    num_labels = len(label_names)
    label_binarizer = MultiLabelBinarizer().fit(train_df['labels'])
    label_to_index = {k: v for v, k in enumerate(label_binarizer.classes_)}
    for name in label_binarizer.classes_:
        print(label_binarizer.transform([[name]]))

    cp_model = configparser.ConfigParser()
    cp_model.read('llm_config.ini')
    data_config_name = args.dataset
    model_config = cp_model[args.model]
    model_path = model_config['name']
    model_name = args.model

    #TODO: read dfs based on args.dataset
    json.dump(label_to_index, open(f"label_to_index_{data_config_name}.json", "w"))

    # set up model stuff
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)  # tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    if freeze_bert:
        model.bert.requires_grad_(False)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True)
    loss_func = BCEWithLogitsLoss()

    train_loader = get_dataloader(train_df, label_binarizer, tokenizer, args.max_length, args.batch_size, 'train', 'query')
    val_loader = get_dataloader(val_df, label_binarizer, tokenizer, args.max_length, args.batch_size, 'val', 'query')
    test_loader = get_dataloader(test_df, label_binarizer, tokenizer, args.max_length, args.batch_size, 'test', 'query')
    print(list(model.parameters()))
    for _ in trange(args.epochs, desc="Epoch"):
        training_loop(model, train_loader, optimizer, loss_func, device, num_labels)
        pred_labels, true_labels = evaluate(model, val_loader, device)
        get_metrics(pred_labels, true_labels)

    pred_labels, true_labels = evaluate(model, val_loader, device)
    val_df['predictions'] = [x.tolist() for x in pred_labels]
    val_df.to_csv(f"multilabel_{model_name}_{data_config_name}_{args.epochs}_val.tsv", sep='\t', index=False)

    pred_labels, true_labels = evaluate(model, test_loader, device)
    get_metrics(pred_labels, true_labels)
    test_df['predictions'] = [x.tolist() for x in pred_labels]
    test_df.to_csv(f"multilabel_{model_name}_{data_config_name}_{args.epochs}.tsv", sep='\t', index=False)

    # for online inference
    query = "find opthalmologist near me please"
    encoding = tokenizer.encode_plus(query)
    encoding


if __name__ == '__main__':
    seed_everything(42)
    main()
