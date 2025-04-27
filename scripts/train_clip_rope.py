import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from ldm.modules.encoders.modules import FrozenCLIPEmbedder

# === Config ===
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 3
lr = 1e-5
max_length = 77
save_path = "./clip_rope_finetuned.pth"

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

        if not any(word in caption for word in self.number_words):
            caption = "one object."  # fallback dummy caption

        encoding = self.tokenizer(caption, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return input_ids, attention_mask

# === Model ===
model = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14", device=device, max_length=max_length)

# ❗ Unfreeze only transformer parameters
for param in model.transformer.parameters():
    param.requires_grad = True

model = model.to(device)

# ❗ Optimizer on transformer parameters
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.transformer.parameters()), lr=lr)

# === Dataloader ===
dataset = CocoCountingDataset(split="train", tokenizer=model.tokenizer, max_length=max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# === Training ===
model.train()
for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask in tqdm(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        # Simple L2 loss
        loss = torch.mean(torch.norm(embeddings, dim=-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}")

# === Save the fine-tuned transformer
torch.save(model.transformer.state_dict(), save_path)
print(f"Fine-tuned text encoder saved to {save_path}")
