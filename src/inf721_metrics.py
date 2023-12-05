import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def show_results(num_epochs, model, X_train, y_train, X_test, y_test, train_losses, test_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b')
    plt.plot(range(1, num_epochs + 1), test_losses, marker='x', linestyle='-', color='g')
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.show()

    y_pred = model(X_train)
    print("\nTrain results:")
    print(classification_report(y_train.detach().numpy(), y_pred.round().detach().numpy(), zero_division=0))
    y_pred = model(X_test)
    print("\nTest results:")
    print(classification_report(y_test.detach().numpy(), y_pred.round().detach().numpy(), zero_division=0))
