import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def blending_weights(errors):
    """
    Compute weights for each dataset using inverse squared errors.
    Errors with smaller variance get higher weights.
    """
    inv_sq = 1 / (errors ** 2 + 1e-10)
    sum_inv_sq = np.sum(inv_sq, axis=0, keepdims=True)
    return inv_sq / sum_inv_sq


def blend_estimates(estimates, errors):
    """
    Blend multiple dataset estimates into one corrected product.
    Also compute the overall uncertainty of the blended estimate.
    """
    weights = blending_weights(errors)
    blended_estimate = np.sum(weights * estimates, axis=0)
    blended_uncertainty = 1 / np.sum(1 / (errors ** 2 + 1e-10), axis=0)
    return blended_estimate, blended_uncertainty


def simulate_precipitation_datasets(T=800, seed=42):
    """
    Generate synthetic precipitation datasets and their errors
    to simulate satellite/rain gauge uncertainty for TC.
    """
    np.random.seed(seed)
    true_precip = np.sin(np.linspace(0, 20, T)) + np.random.normal(0, 0.05, size=T)
    datasets, errors = [], []
    for i in range(3):  # simulate 3 input datasets
        noise = np.random.normal(0, 0.3 + 0.1 * i, size=T)
        x_i = true_precip + noise
        err = np.abs(noise)
        datasets.append(x_i)
        errors.append(err)
    return true_precip, np.array(datasets), np.array(errors)



def create_sequences(data, targets, seq_length):
    """
    Convert time series data into LSTM sequences.
    Each input sequence has length `seq_length`.
    """
    X_seq, y_seq = [], []
    for t in range(len(data) - seq_length):
        X_seq.append(data[t:t+seq_length, :])
        y_seq.append(targets[t+seq_length])
    return np.array(X_seq), np.array(y_seq)


def normalize(data):
    """
    Normalize features to zero mean and unit variance.
    """
    mean, std = data.mean(axis=0), data.std(axis=0) + 1e-6
    return (data - mean) / std, mean, std



class CustomLSTMCell(nn.Module):
    """
    Custom implementation of a single LSTM cell
    with input, forget, output gates, and cell state.
    """
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        # Input gate
        self.W_i, self.U_i, self.b_i = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Parameter(torch.zeros(hidden_dim))
        # Forget gate
        self.W_f, self.U_f, self.b_f = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Parameter(torch.zeros(hidden_dim))
        # Candidate cell state
        self.W_c, self.U_c, self.b_c = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Parameter(torch.zeros(hidden_dim))
        # Output gate
        self.W_o, self.U_o, self.b_o = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t, h_prev, c_prev):
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev) + self.b_i)
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev) + self.b_f)
        c_tilde = torch.tanh(self.W_c(x_t) + self.U_c(h_prev) + self.b_c)
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev) + self.b_o)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTMModel(nn.Module):
    """
    Full LSTM sequence model built from CustomLSTMCell.
    Predicts precipitation at final time step.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm_cell = CustomLSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq):
        batch_size = x_seq.size(0)
        h_t, c_t = torch.zeros(batch_size, self.hidden_dim), torch.zeros(batch_size, self.hidden_dim)
        for t in range(self.sequence_length):
            x_t = x_seq[:, t, :]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
        return self.fc(h_t).squeeze(-1)


def train_lstm(model, X_train, y_train, epochs=200, lr=1e-3):
    """
    Train the LSTM model using MSE loss and Adam optimizer.
    """
    model.train()
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    return model



def evaluate(y_true, y_pred):
    """
    Compute standard evaluation metrics: RMSE, MAE, R2.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}



def main():
    # Step 1: Simulate raw datasets
    true_precip, datasets, errors = simulate_precipitation_datasets(T=800)

    # Step 2: Apply Triple Collocation blending
    tc_precip, tc_uncertainty = blend_estimates(datasets, errors)

    # Step 3: Build feature set (precip + auxiliaries)
    auxiliaries = np.random.randn(len(tc_precip), 3)  # simulate other predictors
    features = np.column_stack((tc_precip, auxiliaries))
    features, _, _ = normalize(features)
    target = tc_precip

    # Step 4: Create LSTM sequences
    seq_length = 10
    X_seq, y_seq = create_sequences(features, target, seq_length)

    # Step 5: Train/test split
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Step 6: Initialize and train LSTM
    input_dim, hidden_dim = features.shape[1], 64
    model = LSTMModel(input_dim, hidden_dim, 1, seq_length)
    model = train_lstm(model, X_train, y_train, epochs=200)

    # Step 7: Predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Step 8: Evaluation
    metrics = evaluate(y_test, y_pred)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Step 9: Display sample predictions
    print("\nSample Results:")
    print("Observed:", y_test[:10])
    print("Predicted:", y_pred[:10])
    print("Uncertainty:", tc_uncertainty[seq_length:seq_length+10])


if __name__ == "__main__":
    main()
