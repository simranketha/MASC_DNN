import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# def create_data_loader(train_x,corrupted_trainy,test_x,original_testy):
#     batch_size = 128
    
#     train_x=torch.tensor(train_x.clone().detach().requires_grad_(True), dtype=torch.float32)
#     test_x=torch.tensor(test_x.clone().detach().requires_grad_(True), dtype=torch.float32)
    
#     corrupted_trainy = torch.tensor(corrupted_trainy.clone().detach(), dtype=torch.long)
#     original_testy = torch.tensor(original_testy.clone().detach(), dtype=torch.long)
    
#     train_dataset = TensorDataset(train_x, corrupted_trainy)
#     test_dataset = TensorDataset(test_x, original_testy)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader,test_loader

def create_data_loader(train_x, corrupted_trainy, test_x, original_testy, batch_size=128):

    train_x = train_x.float()
    test_x = test_x.float()

    corrupted_trainy = corrupted_trainy.long()
    original_testy = original_testy.long()

    train_dataset = TensorDataset(train_x, corrupted_trainy)
    test_dataset = TensorDataset(test_x, original_testy)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)
    
    def compute_flops(self, batch_size=1):
        # For Linear: 2 * input_dim * num_classes * batch_size
        input_dim = self.linear.in_features
        num_classes = self.linear.out_features
        flops = 2 * input_dim * num_classes * batch_size
        return flops
    
def training_probe(model, train_loader, dev, epochs=20, lr=1e-3, batch_size=128):
    
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"FLOPs for forward pass (batch size {batch_size}): {model.compute_flops(batch_size)}")
    flops_train=model.compute_flops(batch_size)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)        
            loss.backward()                     
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

    return flops_train
    
def inference(model, test_loader, dev):
    model.eval()
    correct, total = 0, 0
    total_flops = 0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(dev), y_batch.to(dev)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # Compute FLOPs for this batch
            batch_flops = model.compute_flops(batch_size=x_batch.size(0))
            total_flops += batch_flops

    accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Approximate total FLOPs for inference: {total_flops:,}")

    return accuracy, total_flops
