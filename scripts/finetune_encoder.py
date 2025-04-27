import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F

from ldm.modules.encoders.modules import FrozenCLIPEmbedder

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 3
lr = 1e-5
max_length = 77
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_every_n_steps = 1000  # Save every 1000 batches

# === Dataset ===
class CocoCountingDataset(torch.utils.data.Dataset):
    def __init__(self, split="train", tokenizer=None, max_length=77):
        self.dataset = load_dataset("conceptual_captions", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.number_words = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        caption = self.dataset[idx]['caption'].lower()
        label = int(any(word in caption for word in self.number_words))  # label 1 if counting word exists

        if label == 0:
            caption = "one object."

        encoding = self.tokenizer(caption, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask, label

# === Model ===
model = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", device=device, max_length=max_length)

for param in model.transformer.parameters():
    param.requires_grad = True

model = model.to(device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.transformer.parameters()), lr=lr)

# === Dataloader ===
dataset = CocoCountingDataset(split="train", tokenizer=model.tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# === Training ===
model.train()
global_step = 0
for epoch in range(epochs):
    total_loss = 0
    preds, targets = [], []

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(tqdm(dataloader)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        loss = torch.mean(torch.norm(embeddings, dim=-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Mock "classification" for precision/recall: use embedding norm as pseudo-score
        scores = torch.norm(embeddings[:, 0, :], dim=-1)  # CLS token norm
        pred_labels = (scores > scores.mean()).long()

        preds.extend(pred_labels.cpu().tolist())
        targets.extend(labels.cpu().tolist())

        global_step += 1

        # === Save checkpoint mid-epoch
        if global_step % save_every_n_steps == 0:
            checkpoint_path = os.path.join(save_dir, f"clip_rope_step{global_step}.pth")
            torch.save(model.transformer.state_dict(), checkpoint_path)
            print(f"[Checkpoint] Saved at step {global_step}")

    # === End of epoch logging ===
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='binary')
    print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}")
    print(f"Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")

    # Save after each epoch
    checkpoint_path = os.path.join(save_dir, f"clip_rope_epoch{epoch+1}.pth")
    torch.save(model.transformer.state_dict(), checkpoint_path)
    print(f"[Checkpoint] Saved model after epoch {epoch+1}")

# === Final Save ===
torch.save(model.transformer.state_dict(), "./clip_rope_finetuned_final.pth")
print("[Final Save] Fine-tuned text encoder saved!")
