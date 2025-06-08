import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("sample_data.csv")

# Preprocess
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Model
class SentimentNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Output: positive or negative

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SentimentNet(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
epochs = 20
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.savefig("training_loss.png")
plt.show()

# Evaluate
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = torch.argmax(predictions, axis=1)
    print("\nPredictions on Test Set:")
    for i in range(len(X_test)):
        print(f"Text: {df['text'].iloc[i]} → Prediction: {'Positive' if predicted_classes[i]==1 else 'Negative'}")
