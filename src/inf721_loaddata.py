import torch, re
from datasets import load_dataset
from gensim.models import Word2Vec
import numpy as np

def pre_proccess(examples):
    contracts = []
    targets = []

    for contract in examples['source_code']:
        sequence = re.sub(r'\s{2,}', ' ', re.sub(r'/\*.*?\*/', '', re.sub(r'//.*', '', contract), flags=re.DOTALL)).replace('\n', ' ').split()
        contracts.append(sequence)

    for result in examples['slither']:
        targets.append(0 if str(result) == '[4]' else 1)

    return {'sequences': contracts, 'targets': targets}

def load_data():
    train_dataset = load_dataset('mwritescode/slither-audited-smart-contracts', 'big-multilabel', split='train[:50%]')
    test_dataset = load_dataset('mwritescode/slither-audited-smart-contracts', 'big-multilabel', split='test')
    
    train_dataset = train_dataset.map(pre_proccess, batched=True, batch_size=5000)
    test_dataset = test_dataset.map(pre_proccess, batched=True, batch_size=5000)

    w2v_model_train = Word2Vec(train_dataset['sequences'], vector_size=100, window=3, min_count=1, workers=4)
    w2v_model_test = Word2Vec(test_dataset['sequences'], vector_size=100, window=3, min_count=1, workers=4)
    
    def create_features_train(examples):
        features = []
        for contract in examples['sequences']:
            word_vectors = [w2v_model_train.wv[word] for word in contract if word in w2v_model_train.wv]
            if word_vectors:
                features.append(np.mean(word_vectors, axis=0))
            else:
                features.append(np.zeros(w2v_model_train.vector_size))

        return {'features': features}

    def create_features_test(examples):
        features = []
        for contract in examples['sequences']:
            word_vectors = [w2v_model_test.wv[word] for word in contract if word in w2v_model_test.wv]
            if word_vectors:
                features.append(np.mean(word_vectors, axis=0))
            else:
                features.append(np.zeros(w2v_model_test.vector_size))

        return {'features': features}
    
    train_dataset = train_dataset.map(create_features_train, batched=True, batch_size=5000)
    test_dataset = test_dataset.map(create_features_test, batched=True, batch_size=5000)

    X_train = torch.tensor(train_dataset['features'], dtype=torch.float32)
    y_train = torch.tensor(train_dataset['targets'], dtype=torch.float32).reshape(-1, 1)

    X_test = torch.tensor(test_dataset['features'], dtype=torch.float32)
    y_test = torch.tensor(test_dataset['targets'], dtype=torch.float32).reshape(-1, 1)

    return X_train, y_train, X_test, y_test
