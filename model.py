import music21 as m21
import numpy as np
import torch
from collections import Counter
import glob 
import random

# Чтение MIDI файлов
midi_path = glob.glob("/home/goida/Desktop/otus_home_work/final_project/ml/dataset/*.mid")  # Add your MIDI file paths here
all_tokens = []

# Функция для чтения MIDI и преобразования в токены
def midi_to_tokens(midi_path):
    """Читает MIDI и кодирует в последовательность токенов (нота+длительность)."""
    score = m21.converter.parse(midi_path)
    tokens = []
    for el in score.flat.notes:
        if isinstance(el, m21.note.Note):
            pitch = str(el.pitch)  # например 'C4'
            dur = el.quarterLength  # длительность в четвертях
            tokens.append(f"{pitch}_{dur}")
        elif isinstance(el, m21.chord.Chord):
            # Для аккорда: список нот
            pitches = ".".join(str(p) for p in el.pitches)
            dur = el.quarterLength
            tokens.append(f"{pitches}_{dur}")
    return tokens

# Собираем все токены из всех MIDI файлов
for path in midi_path:
    tokens = midi_to_tokens(path)
    all_tokens.extend(tokens)

# print(f"Total tokens: {len(all_tokens)}")
# print(all_tokens[:20])

# Создаем словарь токенов
vocab =  sorted(set(all_tokens))
token2idx = {token: idx for idx, token in enumerate(vocab)}
idx2token = {idx: token for token, idx in token2idx.items()}

# Кодируем всю последовательность в индексы
sequences = [token2idx[tok] for tok in all_tokens]

# Dataset и DataLoader
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

seq_len = 32
dataset = MusicDataset(sequences, seq_len = seq_len)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


# print("Пример батча:")
# for x, y in dataloader:
#     print("X:", x.shape, "Y:", y.shape)
#     break

#  Модель Transformer 

class TokenAndPositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=5000):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = torch.nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x shape: [batch, seq_len]
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        return self.token_emb(x) + self.pos_emb(positions)  # [batch, seq_len, embed_dim]


class MusicTransformer(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=32, n_heads=2, n_layers=2, max_len=300):
        super().__init__()
        self.embedding = TokenAndPositionalEmbedding(vocab_size, embed_dim, max_len)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first= True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = torch.nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = self.transformer(x)  # [seq_len, batch, embed_dim]
        return self.fc(x)        # [batch, seq_len, vocab_size]
    

vocab_size = 10
model = MusicTransformer(vocab_size)

batch = 2
seq_len = 10
x = torch.randint(0, vocab_size, (2, 10))  # [batch, seq_len]

print("input:", x.shape)
y = model(x)
print("output:", y.shape)
model = MusicTransformer(vocab_size=len(vocab))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Тренировка модели
for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, len(vocab)), y.view(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} | Loss {loss.item()}")    

# Инференс - генерация музыки
def generate_music(model, seed_tokens, token2idx, idx2token, max_len=100, top_k=5, device="cpu"):
    model.eval()
    generated = seed_tokens[:]

    x = torch.tensor(seed_tokens, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq]

    for _ in range(max_len):
        with torch.no_grad():
            out = model(x)  # [1, seq, vocab_size]
            next_token_logits = out[0, -1, :]  # последний токен: [vocab_size]

            # top-k выбор (чтобы не всегда argmax)
            values, indices = torch.topk(next_token_logits, k=top_k)
            probs = torch.softmax(values, dim=-1).cpu().numpy()
            next_token = indices[torch.multinomial(torch.tensor(probs), 1)].item()

        generated.append(next_token)

        # обновляем вход (добавляем новый токен)
        x = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

    return [idx2token[idx] for idx in generated]

# Допустим, модель и словари уже загружены
seed = [token2idx["C4_0.5"], token2idx["E4_1.0"], token2idx["G4_0.5"]]

generated_tokens = generate_music(
    model,
    seed,
    token2idx,
    idx2token,
    max_len=50,
    top_k=5
)

print("Generated tokens:", generated_tokens[:20])



# Функция для сохранения токенов в MIDI файл
def tokens_to_midi(tokens, out_file="generated.mid"):
    s = m21.stream.Stream()
    for t in tokens:
        try:
            pitch, dur = t.split("_")
            n = m21.note.Note(pitch)
            n.quarterLength = float(dur)
            s.append(n)
        except Exception as e:
            print(f"Ошибка на токене {t}: {e}")
            pass
    s.write("midi", fp=out_file)
    print(f"Saved {out_file}")

# Тестовые данные 
test_tokens = [
    "C4_1.0",  # нота C4, четвертная
    "D4_0.5",  # нота D4, восьмая
    "E4_2.0",  # нота E4, половинная
    "G4_1.0",  # нота G4, четвертная
]

tokens_to_midi(test_tokens, "test.mid")







if __name__ == "__main__":
    print("This script is in the top-level code environment")