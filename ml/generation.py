import torch
import pickle
import music21 as m21

# === Функция для сохранения токенов в MIDI === #
def tokens_to_midi(tokens, out_file="generated.mid"):
    s = m21.stream.Stream()
    for t in tokens:
        try:
            pitch, dur = t.split("_")
            n = m21.note.Note(pitch)
            n.quarterLength = float(dur)
            s.append(n)
        except:
            pass
    s.write("midi", fp=out_file)
    print(f"🎶 Сохранён файл {out_file}")

# === Загружаем обученную модель === #
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

vocab = saved["vocab"]
token2idx = saved["token2idx"]
idx2token = saved["idx2token"]

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

model = MusicTransformer(len(vocab))
model.load_state_dict(saved["model_state"])
model.eval()

# === Генерация === #
def generate_music(seed="C4_1.0", length=20):
    tokens = [seed]
    for _ in range(length):
        x = torch.tensor([[token2idx[t] for t in tokens[-10:]]])  # последние 10 токенов
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out[0, -1], dim=0)
            idx = torch.multinomial(probs, 1).item()
            tokens.append(idx2token[idx])
    return tokens

tokens = generate_music()
tokens_to_midi(tokens)
import torch
import pickle
import music21 as m21

# === Функция для сохранения токенов в MIDI === #
def tokens_to_midi(tokens, out_file="generated.mid"):
    s = m21.stream.Stream()
    for t in tokens:
        try:
            pitch, dur = t.split("_")
            n = m21.note.Note(pitch)
            n.quarterLength = float(dur)
            s.append(n)
        except:
            pass
    s.write("midi", fp=out_file)
    print(f"🎶 Сохранён файл {out_file}")

# === Загружаем обученную модель === #
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

vocab = saved["vocab"]
token2idx = saved["token2idx"]
idx2token = saved["idx2token"]

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

model = MusicTransformer(len(vocab))
model.load_state_dict(saved["model_state"])
model.eval()

# === Генерация === #
def generate_music(seed="C4_1.0", length=20):
    tokens = [seed]
    for _ in range(length):
        x = torch.tensor([[token2idx[t] for t in tokens[-10:]]])  # последние 10 токенов
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out[0, -1], dim=0)
            idx = torch.multinomial(probs, 1).item()
            tokens.append(idx2token[idx])
    return tokens

tokens = generate_music()
tokens_to_midi(tokens)
print("Генерация завершена")