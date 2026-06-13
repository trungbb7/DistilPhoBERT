import os
import pandas as pd
from transformers import AutoModelForTokenClassification
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from seqeval.metrics import classification_report, f1_score
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


tokenizer = AutoTokenizer.from_pretrained("trungbb8/distilphobert")


train_df = pd.read_csv(
    "/workspace/ner-train-processed.csv",
    keep_default_na=False,
    na_values=[""],
)
test_df = pd.read_csv(
    "/workspace/ner-test-processed.csv",
    keep_default_na=False,
    na_values=[""],
)


# Group Sentences
def group_sentence(df):
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
        if token == "</s>":
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
    train_df
)
test_tokens, test_ids, test_labels, test_org, test_inside = group_sentence(test_df)


# Group into chunk
label2id = {
    "-100": -100,
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}
id2label = {v: k for k, v in label2id.items()}
max_length = 140
pad_token_id = tokenizer.pad_token_id
pad_label_id = -100
pad_inside_id = 0


def group_data(
    data_ids,
    data_labels,
    data_inside,
    max_length=140,
    pad_token_id=1,
    pad_label_id=-100,
    pad_inside_id=0,
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

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Eval
def eval(model, dataloader, test=False):
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]
            inside = batch["inside"]
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            for i in range(logits.shape[0]):
                pred_ids = logits[i].argmax(dim=1).cpu().numpy()
                lbls = labels[i]
                pred_ids = pred_ids[inside[i].bool()]
                lbls = lbls[inside[i].bool()]

                y_true.append([id2label[id.item()] for id in lbls])
                y_pred.append([id2label[id.item()] for id in pred_ids])

    model.train()
    f1 = f1_score(y_true, y_pred)
    if test:
        print(classification_report(y_true, y_pred))
    return f1


# Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AutoModelForTokenClassification.from_pretrained(
    "trungbb8/distilphobert", num_labels=9
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
    print(f"Epoch: {epoch} - F1 score: {f1}")
    early_stoping(f1, model)
    if early_stoping.is_stop:
        print(f"Early stoping at {epoch} epoch")
        break

if early_stoping.best_model_weights is not None:
    model.load_state_dict(early_stoping.best_model_weights)


model.save_pretrained("distilphobert-ner-finetuned")


# Eval
eval(model, test_dataloader, test=True)
