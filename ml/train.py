import torch
import pickle
import music21 as m21

# === ТВОЙ КОД === #
# Функция перевода MIDI → токены
def midi_to_tokens(midi_path):
    score = m21.converter.parse(midi_path)
    tokens = []
    for el in score.flat.notes:
        if isinstance(el, m21.note.Note):
            pitch = str(el.pitch)
            dur = el.quarterLength
            tokens.append(f"{pitch}_{dur}")
        elif isinstance(el, m21.chord.Chord):
            pitches = ".".join(str(p) for p in el.pitches)
            dur = el.quarterLength
            tokens.append(f"{pitches}_{dur}")
    return tokens

# === Здесь ты собираешь датасет === #
midi_paths = ["data/song1.mid", "data/song2.mid"]  # список твоих файлов
all_tokens = []
for path in midi_paths:
    all_tokens.extend(midi_to_tokens(path))

vocab = sorted(set(all_tokens))
token2idx = {tok: i for i, tok in enumerate(vocab)}
idx2token = {i: tok for tok, i in token2idx.items()}
sequences = [token2idx[tok] for tok in all_tokens]

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_len):
        self.sequences = sequences
        self.seq_len = seq_len
    def __len__(self):
        return len(self.sequences) - self.seq_len
    def __getitem__(self, idx):
        x = self.sequences[idx:idx+self.seq_len]
        y = self.sequences[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

dataloader = torch.utils.data.DataLoader(MusicDataset(sequences, 32), batch_size=16, shuffle=True)

# === Модель === #
class TokenAndPositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=5000):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = torch.nn.Embedding(max_len, embed_dim)
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)

class MusicTransformer(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=32, n_heads=2, n_layers=2, max_len=300):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(vocab_size, embed_dim, max_len)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                                         dim_feedforward=512, dropout=0.1, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = torch.nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

# === Тренировка === #
model = MusicTransformer(len(vocab))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(2):  # для теста маленький цикл
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, len(vocab)), y.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# === Сохраняем всё нужное === #
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "token2idx": token2idx,
        "idx2token": idx2token
    }, f)

print("Модель сохранена в model.pkl")
