import torch
from inf721_loaddata import load_data
from inf721_model import SmartContractClassifier
from sklearn.metrics import classification_report

def evaluate(model_path='../model/smclassifier.pth'):
    X_train, y_train, X_test, y_test = load_data()

    model = SmartContractClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_pred = model(X_train)
    print("\nTrain results:")
    print(classification_report(y_train.detach().numpy(), y_pred.round().detach().numpy(), zero_division=0))
    y_pred = model(X_test)
    print("\nTest results:")
    print(classification_report(y_test.detach().numpy(), y_pred.round().detach().numpy(), zero_division=0))

    return model

if __name__ == "__main__":
    model_path = "../model/scclassifier.pth"
    model_evaluated = evaluate(model_path)
