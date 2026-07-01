# Trabalho 2 – Redes Neurais Artificiais
**Classificação de Cães e Gatos**  
Disciplina de Inteligência Artificial – Ciência da Computação – IFSC Lages
Alunos: Felipe R Branco, Bernardo Floriani, Cesar Machado, Joao Vitor Sutil
---

## Estrutura do Projeto

```
/
├── datasets/
│   ├── train/
│   │   ├── cat/        # imagens de gatos para treino
│   │   └── dog/        # imagens de cachorros para treino
│   ├── validation/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
│       ├── cat/
│       └── dog/
│
├── mlp_manual.ipynb        # Etapa 1 – MLP do zero com NumPy
├── cnn_pytorch.ipynb       # Etapa 2 – CNN com PyTorch
└── transfer_learning.ipynb # Etapa 3 – Transfer Learning (ResNet-18 + MobileNet V2)
```

---

## Dependências

Instale todas as dependências com o comando abaixo (pode rodar numa célula do notebook):

```bash
pip install torch torchvision numpy matplotlib Pillow scikit-learn
```

> Se o `pip` normal não funcionar no VS Code, use:
> ```bash
> import sys
> !{sys.executable} -m pip install torch torchvision numpy matplotlib Pillow scikit-learn
> ```

---

## Como Rodar

### Opção 1 – VS Code (local)
1. Instale a extensão **Jupyter** no VS Code
2. Abra o notebook desejado (`.ipynb`)
3. Selecione o interpretador Python correto (canto superior direito)
4. Execute as células em ordem com `Shift + Enter`

### Opção 2 – Google Colab (recomendado sem GPU local)
1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do notebook
3. Faça upload da pasta `datasets/` ou monte o Google Drive
4. Ative a GPU em **Runtime → Change runtime type → GPU**
5. Execute as células normalmente

### Opção 3 – Jupyter Notebook
```bash
pip install jupyter
jupyter notebook
```
Abra o arquivo `.ipynb` no navegador.

---

## Descrição dos Notebooks

### `mlp_manual.ipynb` – Etapa 1
Implementação de uma MLP **do zero**, usando apenas NumPy. Nenhum framework de deep learning é utilizado.

- Imagens: 64×64, tons de cinza, vetorizadas
- Arquitetura: entrada (4096) → oculta (128, ReLU) → saída (1, Sigmoid)
- Funções implementadas manualmente: forward pass, ReLU, sigmoid, Binary Cross-Entropy, backpropagation, gradiente descendente
- Saídas: acurácia, loss, tempo de treino, matriz de confusão, F1-score

### `cnn_pytorch.ipynb` – Etapa 2
CNN construída do zero com PyTorch. Sem modelos pré-treinados.

- Imagens: 224×224, RGB (3 canais), tensores
- Arquitetura: 3 blocos Conv → BatchNorm → ReLU → MaxPool + 2 camadas FC
- Data Augmentation: flip horizontal (geométrico), rotação 10° e ColorJitter (cor)
- Saídas: acurácia, loss, tempo de treino, matriz de confusão, F1-score

### `transfer_learning.ipynb` – Etapa 3
Fine-tuning de dois modelos pré-treinados do TorchVision.

| Modelo        | Camada final substituída | Parâmetros congelados |
|---------------|--------------------------|----------------------|
| ResNet-18     | `fc` (512 → 2)           | Todas exceto `fc`    |
| MobileNet V2  | `classifier` (1280 → 2)  | Todas exceto `classifier` |

- Data Augmentation: mesmas 3 técnicas da Etapa 2
- Normalização ImageNet nos pré-processamentos
- Saídas: acurácia, loss, tempo de treino, matriz de confusão e gráfico comparativo entre os dois modelos

---

## Observações

- A variável `DATASET_DIR` no início de cada notebook aponta para a pasta `datasets/`. Altere se o seu dataset estiver em outro caminho.
- Os notebooks foram desenvolvidos e testados com Python 3.10.
