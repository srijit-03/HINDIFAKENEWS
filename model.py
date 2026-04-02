import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv("dataset.csv", index_col=0)

# Keep only required columns
df = df[["text", "label"]]

# Clean data
df["text"] = df["text"].astype(str)
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["text", "label"])

print("Dataset loaded:", len(df))

# ==============================
# LOAD HINVEC MODEL
# ==============================
model_name = "Sailesh97/Hinvec"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# ==============================
# EMBEDDING FUNCTION
# ==============================
def get_hinvec_embedding(text):

    tokens = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=32,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**tokens)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    return emb


# ==============================
# CNN MODEL
# ==============================
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * ((input_dim - 2)//2), 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# ==============================
# CREATE FEATURES
# ==============================
X = []
y = []

for i, row in df.iterrows():
    try:
        emb = get_hinvec_embedding(row["text"])
        X.append(emb)
        y.append(int(row["label"]))
    except Exception as e:
        print("Skipping row:", i, e)

print("Total samples:", len(X))

# 🚨 SAFETY CHECK
if len(X) == 0:
    raise ValueError("❌ No data available after preprocessing!")

# ==============================
# NUMPY CONVERSION
# ==============================
X = np.array(X)
y = np.array(y)

# ==============================
# TRAIN TEST SPLIT (ON NUMPY)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==============================
# CONVERT TO TENSOR
# ==============================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# ==============================
# MODEL INIT
# ==============================
input_dim = X_train.shape[1]
model_cnn = CNNModel(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)

# ==============================
# TRAINING
# ==============================
epochs = 10

for epoch in range(epochs):
    model_cnn.train()

    optimizer.zero_grad()

    outputs = model_cnn(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ==============================
# EVALUATION
# ==============================
model_cnn.eval()

with torch.no_grad():
    outputs = model_cnn(X_test)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_test).sum().item() / len(y_test)

print("Accuracy:", accuracy)

# ==============================
# SAVE MODEL
# ==============================
torch.save(model_cnn.state_dict(), "cnn_model.pth")

print("✅ Model saved as cnn_model.pth")