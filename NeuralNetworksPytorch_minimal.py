import torch
import torch.nn as nn

from Dataprep2 import finalrunner

# Hyperparameter (einfach anpassbar)
SHEET = 3      # Excel-Sheet Index
EPOCHS = 200   # Trainingsdurchläufe
LR = 0.01      # Lernrate

# 1) Daten laden: X (N×1), Y (N×3)
X_df, Y_df = finalrunner(SHEET)
X = torch.tensor(X_df.values, dtype=torch.float32)
Y = torch.tensor(Y_df.values, dtype=torch.float32)

# 2) Modell: nur eine Linearschicht 1 -> 3
#    y = X * W^T + b  (W: 3×1, b: 3)
model = nn.Linear(1, 3)

# 3) Verlustfunktion (MSE) und Optimierer (SGD)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# 4) Training: voller Batch
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
        print(f"Epoch {epoch}/{EPOCHS} - loss={loss.item():.6f}")

# 5) Gewichte und Beispielvorhersagen anzeigen
print("Gewichte (W):\n", model.weight.data)
print("Bias (b):\n", model.bias.data)

with torch.no_grad():
    print("Vorhersagen (erste 5 Zeilen):")
    print(model(X[:5]))

