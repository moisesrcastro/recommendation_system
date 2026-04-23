# Marketplace Recomendador

Projeto de um sistema de recomendação de produtos baseado em embeddings e carrinho de compras, com interface interativa em Gradio.

## Como funciona

- Usuário adiciona produtos ao carrinho  
- O sistema calcula um vetor médio dos itens selecionados  
- Um modelo de ML gera scores de recomendação  
- Produtos são ordenados por relevância  

## Funcionalidades

- Listagem de produtos  
- Adicionar e remover itens do carrinho  
- Limpar carrinho  
- Recomendações em tempo real  
- Cálculo do total do carrinho  

## Tecnologias

- Python  
- Gradio  
- Pandas  
- NumPy  
- Keras / TensorFlow  
- SQLAlchemy  

## Estrutura

src/
├── database/
│ ├── connection.py
│ ├── repository.py
│
├── recommender/
│ ├── model.py
│
app.py
## Estrutura
