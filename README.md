# garranchoOCR

Um projeto em desenvolvimento para reconhecimento de manuscritos cursivos e garranchados usando técnicas de Machine Learning. O objetivo é treinar um modelo capaz de entender letras manuscritas, começando pela sua própria letra.

## Status

🚧 **Em desenvolvimento** 🚧

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **TensorFlow**: Biblioteca para criação e treinamento de modelos de Machine Learning.
- **Keras**: API de alto nível para construção e treinamento de redes neurais.
- **NumPy**: Biblioteca para computação numérica.
- **Matplotlib**: Biblioteca para visualização de dados.

## Funcionalidades

- Treinamento de um modelo de reconhecimento de caracteres manuscritos.
- Avaliação da acurácia do modelo em um conjunto de dados de teste.
- Visualização de imagens testadas e suas previsões.
- Paginamento para exibição de múltiplas imagens.

## Estrutura do Projeto

```
garranchoOCR/
│
├── data/                   # Pasta para dados (imagens, etc.)
│   ├── train/              # Dados de treinamento
│   │   ├── A/              # Imagens da letra A
│   │   ├── a/              # Imagens da letra a
│   │   └── ...             # Outras letras
│   └── test/               # Dados de teste
│
├── models/                 # Modelos treinados
│
├── notebooks/              # Notebooks para exploração de dados
│
├── requirements.txt        # Dependências do projeto
├── README.md               # Documentação do projeto
└── main.py                 # Código principal do projeto
```

## Instalação

Para instalar as dependências do projeto, utilize o seguinte comando:

```bash
pip install -r requirements.txt
```

## Como Usar

1. Clone este repositório:
    ```bash
    git clone https://github.com/gabrielcamurcab/garrancho-ocr-python.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd garrancho-ocr-python
    ```
3. Execute o script principal:
    ```bash
    python main.py
    ```

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir um issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

Para mais informações, entre em contato:

- Gabriel Camurça Bezerra
- gabriel.camurca@outlook.com
- https://github.com/gabrielcamurcab