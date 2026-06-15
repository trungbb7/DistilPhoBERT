import os
import pandas as pd
import numpy as np
from transformers import AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score
import copy

batch_size = int(os.environ.get("BATCH_SIZE", 64))


class EarlyStoping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.count = 0
        self.min_delta = min_delta
        self.best_model_weights = None
        self.is_stop = False
        self.best_score = 0

    def __call__(self, current_score, model):
        if self.best_model_weights is None:
            self.best_score = current_score
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif current_score < self.best_score + self.min_delta:
            self.count += 1
            print(f"Patience count: {self.count}")
            if self.count == self.patience:
                self.is_stop = True
        else:
            print(f"Update best socre: {self.best_score:.4f} to {current_score:.4f}")
            self.best_score = current_score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.count = 0


# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("trungbb8/distilphobert")


train_df = pd.read_csv(
    "/workspace/pos-train-processed.csv",
    keep_default_na=False,
    na_values=[""],
)
test_df = pd.read_csv(
    "/workspace/pos-test-processed.csv",
    keep_default_na=False,
    na_values=[""],
)


# Group Sentences
def group_sentence(df, tokenizer):
    tokens = []
    cur_tokens = []
    ids = []
    cur_ids = []
    labels = []
    cur_labels = []
    org = []
    cur_o = []
    inside = []
    cur_ins = []
    for item in df.itertuples():
        token = item.token
        id = item.input_id
        label = item.label
        o = item.org
        ins = item.inside

        cur_tokens.append(token)
        cur_ids.append(id)
        cur_labels.append(label)
        cur_o.append(o)
        cur_ins.append(ins)
        if token == tokenizer.sep_token:
            tokens.append(" ".join(cur_tokens))
            ids.append(cur_ids)
            labels.append(cur_labels)
            org.append(cur_o)
            inside.append(cur_ins)
            cur_tokens = []
            cur_ids = []
            cur_labels = []
            cur_o = []
            cur_ins = []
    return tokens, ids, labels, org, inside


train_tokens, train_ids, train_labels, train_org, train_inside = group_sentence(
    train_df, tokenizer
)
test_tokens, test_ids, test_labels, test_org, test_inside = group_sentence(
    test_df, tokenizer
)


# Group into chunk
label2id = {
    "-100": -100,
    "Ns": 0,
    "N": 1,
    "V": 2,
    "C": 3,
    "A": 4,
    "E": 5,
    "R": 6,
    "CH": 7,
    "P": 8,
    "Cc": 9,
    "NNP": 10,
    "Ne": 11,
    "M": 12,
    "Nc": 13,
    "Nu": 14,
    "L": 15,
    "Nb": 16,
    "T": 17,
    "X": 18,
    "Ny": 19,
    "FW": 20,
    "I": 21,
    "Ni": 22,
    "Z": 23,
    "Vb": 24,
    "O": 25,
    "Vy": 26,
    "I-NP": 27,
    "V    ": 28,
    "Ab": 29,
    "B-NP": 30,
    "NNPY": 31,
    "Cb": 32,
    "NPP": 33,
}
id2label = {v: k for k, v in label2id.items()}
max_length = 256
pad_token_id = tokenizer.pad_token_id
pad_label_id = -100
pad_inside_id = 0


def group_data(
    data_ids,
    data_labels,
    data_inside,
    max_length=max_length,
    pad_token_id=pad_token_id,
    pad_label_id=pad_label_id,
    pad_inside_id=pad_inside_id,
):

    grouped_ids = []
    grouped_labels = []
    grouped_inside = []

    for i in range(len(data_ids)):
        ids = data_ids[i] + [pad_token_id] * (max_length - len(data_ids[i]))
        labels = data_labels[i] + [pad_label_id] * (max_length - len(data_labels[i]))
        inside = data_inside[i] + [pad_inside_id] * (max_length - len(data_inside[i]))
        grouped_ids.append(ids)
        grouped_labels.append(labels)
        grouped_inside.append(inside)

    return grouped_ids, grouped_labels, grouped_inside


grouped_train_ids, grouped_train_labels, grouped_train_inside = group_data(
    train_ids, train_labels, train_inside, max_length=max_length
)
grouped_test_ids, grouped_test_labels, grouped_test_inside = group_data(
    test_ids, test_labels, test_inside, max_length=max_length
)

# Prepare dataset


class NERDataset(Dataset):
    def __init__(self, ids, labels, inside):
        self.ids = ids
        self.labels = labels
        self.inside = inside

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.ids[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "inside": torch.tensor(self.inside[idx], dtype=torch.long),
            "attention_mask": torch.ones(len(self.ids[idx]), dtype=torch.long),
        }
        return item

    def __len__(self):
        return len(self.ids)


dataset = NERDataset(grouped_train_ids, grouped_train_labels, grouped_train_inside)
val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = NERDataset(grouped_test_ids, grouped_test_labels, grouped_test_inside)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Eval
def eval(model, dataloader, test=False):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_preds = []
        insides = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]
            inside = batch["inside"]
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            y_preds.append(pred_ids)
            y_true.append(labels.numpy())
            insides.append(inside)

    y_true = [
        id2label[label_id]
        for batch in y_true
        for sentence in batch
        for label_id in sentence
    ]
    y_preds = [
        id2label[label_id]
        for batch in y_preds
        for sentence in batch
        for label_id in sentence
    ]
    insides = [i for batch in insides for sentence in batch for i in sentence]

    y_true = np.array(y_true)
    y_preds = np.array(y_preds)
    insides = np.array(insides)

    y_true = y_true[insides.astype(bool)]
    y_preds = y_preds[insides.astype(bool)]

    model.train()
    f1 = f1_score(y_true, y_preds, average="micro")
    if test:
        print(classification_report(y_true, y_preds))
    return f1


# Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=34).to(device)
model = AutoModelForTokenClassification.from_pretrained(
    "trungbb8/distilphobert", num_labels=34
).to(device)
model.train()

criteria = torch.nn.CrossEntropyLoss()
learning_rate = 1e-5
epochs = 30

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
early_stoping = EarlyStoping(patience=5, min_delta=0.001)

global_step = 0
for epoch in range(30):
    print(f"Epoch: {epoch + 1}")
    for batch in train_dataloader:
        global_step += 1
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        inside = batch["inside"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        logits = outputs.logits.view(-1, model.config.num_labels)
        labels = labels.view(-1)
        loss = criteria(logits, labels)

        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if global_step % 100 == 0:
            print(f"Global step: {global_step} - Loss: {loss.item()}")

    # eval
    f1 = eval(model, val_dataloader)
    print(f"Epoch: {epoch + 1} - F1 score: {f1}")
    early_stoping(f1, model)
    if early_stoping.is_stop:
        print(f"Early stoping at {epoch} epoch")
        break

if early_stoping.best_model_weights is not None:
    model.load_state_dict(early_stoping.best_model_weights)


model.save_pretrained("distilphobert-pos-finetuned")


# Eval
eval(model, test_dataloader, test=True)
