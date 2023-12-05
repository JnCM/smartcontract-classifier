import torch
from inf721_loaddata import load_data
from inf721_model import SmartContractClassifier
from inf721_dataset import SmartContractDataset
from inf721_metrics import show_results
from torch.utils.data import DataLoader

def calc_test_loss(model, dataloader, loss_function):
    with torch.no_grad():
        total_loss = 0.0
        for x_t, y_t in dataloader:
            yhat_t = model(x_t)
            loss = loss_function(yhat_t, y_t)
            total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        return average_loss

def train_model(num_epochs=15, learning_rate=0.0001, weight_decay=0.1, batch_size=32):
    
    X_train, y_train, X_test, y_test = load_data()
    train_dataset = SmartContractDataset(X_train, y_train)
    test_dataset = SmartContractDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SmartContractClassifier()
    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        total_train_loss = 0
        for x_t, y_t in train_loader:
            optimizer.zero_grad()
            yhat_t = model(x_t)
            loss = loss_function(yhat_t, y_t)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        average_test_loss = calc_test_loss(model, test_loader, loss_function)
        train_losses.append(average_train_loss)
        test_losses.append(average_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {average_train_loss} Test Loss: {average_test_loss}")
    
    show_results(num_epochs, model, X_train, y_train, X_test, y_test, train_losses, test_losses)

    return model


if __name__ == "__main__":
    outpath = "../model/scclassifier.pth"
    model_trained = train_model()
    torch.save(model_trained.state_dict(), outpath)
