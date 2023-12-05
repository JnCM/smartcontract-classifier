# Classificação de Contratos Inteligentes Vulneráveis e Seguros Utilizando Redes Neurais Artificiais

Este repositório contém os códigos-fonte para reproduzir os resultados do Trabalho Final da disciplina INF721 - Redes Neurais Profundas, da Universidade Federal de Viçosa. O presente trabalho apresenta um modelo de Rede Neural Artificial, com o objetivo de classificar contratos inteligentes escritos na linguagem de programação Solidity em contratos seguros (0) e vulneráveis (1). A arquitetura do modelo definida é um Multilayer Perceptron (MLP), sendo que sua versão já treinada para avaliação pode ser encontrada em `./model/scclassifier.pth`.

## Dependências

```bash
pip install -r requirements.txt --no-cache-dir
```

## Reprodução dos resultados

A partir da pasta `src/`, execute o seguinte comando:

```bash
python inf721_inference.py
```

## Treinamento do modelo

A partir da pasta `src/`, execute o seguinte comando:

```bash
python inf721_train.py
```
