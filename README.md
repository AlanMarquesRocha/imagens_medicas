# 🚀 Projeto de classificação de imagens médicas 🩺

Este repositório une _Deep Learning_ (DL) e Inteligência Artificial (IA) generativa para diagnóstico de **pneumonia em imagens de raios‑X**. Aqui você encontrará um pipeline completo, desde o download do _dataset_ até a geração de relatórios inteligentes por LLM.

![image](https://github.com/user-attachments/assets/39475ce2-c99e-4c72-95fa-b6fb4126c0c4)

---

## 🎯 Tecnologias Utilizadas

- **Linguagem**: [Python 3.13.3.](https://www.python.org/downloads/)
- **Deep Learning**: [PyTorch.](https://pytorch.org/) & [torchvision.](https://pytorch.org/vision/stable/index.html)
- **Pré-processamento**: torchvision.transforms, [OpenCV.](https://docs.opencv.org/)
- **Métricas & Visualização**: [scikit-learn.](https://scikit-learn.org/stable/), [matplotlib](https://matplotlib.org/), [seaborn.](https://seaborn.pydata.org/)
- **Geração de PDF**: [FPDF](https://pypi.org/project/fpdf/)
- **LLM de Análise**: Chat-GPT [via OpenAI Python SDK.](https://platform.openai.com/api-keys)

---

## Etapas do Projeto (No Google Colab) 🛠️

1. Carregar um _dataset_ médico de imagens (usaremos o _dataset_ de raios-X de pneumonia ``NIH Chest X-rays``, disponível [**aqui**](https://www.kaggle.com/datasets/nih-chest-xrays/data)).
2. Pré-processamento das imagens.
3. Segmentação (se aplicável).
4. Extração de características e classificação usando ``ResNet-18`` com _transfer learning_.
5. Avaliação com métricas completas.
6. Geração de relatório com análise por LLM (Chat-GPT).
7. Exportação para PDF.
---

1. **Instalação de Dependências**  
   Execute:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configuração do Ambiente**  
   - Defina sua `API_KEY` do OpenAI  (Mais informações sobre API_KEY [**aqui.**](https://platform.openai.com/api-keys)).
   - Verifique se CUDA está ativa para treinamento acelerado.

3. **Download e Preparação do Dataset**  
   - Baixe e extraia o conjunto de raios‑X (`chest_xray.zip`).
   - Organize pastas em `train/`, `val/`, `test/`.
   - **Se preferir rode o código diretamente pelo google Colab.**

4. **Pré-processamento de Dados**  
   - Normalização e aumentos (_flip_, rotação, escala).
   - Pipeline de transforms em `torchvision.transforms`.
   - Utilize outras técnicas de pré-processamento (AHE, [CLAHE](https://explore.albumentations.ai/transform/CLAHE), etc..), caso seja necessário.

5. **Topologia da Rede CNN**  
   Veja detalhes abaixo ⬇️
   
📊 **Detalhes da Topologia (ResNet-18)**

| ⚙️ Camada           | 📐 Saída                 | 🔢 Parâmetros    |
| :------------------ | :---------------------- | ---------------: |
| Conv2d-22           | `[-1, 128, 28, 28]`     | 147,456          |
| BatchNorm2d-23      | `[-1, 128, 28, 28]`     | 256              |
| Conv2d-24           | `[-1, 128, 28, 28]`     | 8,192            |
| BatchNorm2d-25      | `[-1, 128, 28, 28]`     | 256              |
| ReLU-26             | `[-1, 128, 28, 28]`     | 0                |
| **BasicBlock-27**   | `[-1, 128, 28, 28]`     | 0                |
| Conv2d-28           | `[-1, 128, 28, 28]`     | 147,456          |
| BatchNorm2d-29      | `[-1, 128, 28, 28]`     | 256              |
| ReLU-30             | `[-1, 128, 28, 28]`     | 0                |
| Conv2d-31           | `[-1, 128, 28, 28]`     | 147,456          |
| BatchNorm2d-32      | `[-1, 128, 28, 28]`     | 256              |
| ReLU-33             | `[-1, 128, 28, 28]`     | 0                |
| **BasicBlock-34**   | `[-1, 128, 28, 28]`     | 0                |
| Conv2d-35           | `[-1, 256, 14, 14]`     | 294,912          |
| BatchNorm2d-36      | `[-1, 256, 14, 14]`     | 512              |
| ReLU-37             | `[-1, 256, 14, 14]`     | 0                |
| Conv2d-38           | `[-1, 256, 14, 14]`     | 589,824          |
| BatchNorm2d-39      | `[-1, 256, 14, 14]`     | 512              |
| Conv2d-40           | `[-1, 256, 14, 14]`     | 32,768           |
| BatchNorm2d-41      | `[-1, 256, 14, 14]`     | 512              |
| ReLU-42             | `[-1, 256, 14, 14]`     | 0                |
| **BasicBlock-43**   | `[-1, 256, 14, 14]`     | 0                |
| Conv2d-44           | `[-1, 256, 14, 14]`     | 589,824          |
| BatchNorm2d-45      | `[-1, 256, 14, 14]`     | 512              |
| ReLU-46             | `[-1, 256, 14, 14]`     | 0                |
| Conv2d-47           | `[-1, 256, 14, 14]`     | 589,824          |
| BatchNorm2d-48      | `[-1, 256, 14, 14]`     | 512              |
| ReLU-49             | `[-1, 256, 14, 14]`     | 0                |
| **BasicBlock-50**   | `[-1, 256, 14, 14]`     | 0                |
| Conv2d-51           | `[-1, 512, 7, 7]`       | 1,179,648        |
| BatchNorm2d-52      | `[-1, 512, 7, 7]`       | 1,024            |
| ReLU-53             | `[-1, 512, 7, 7]`       | 0                |
| Conv2d-54           | `[-1, 512, 7, 7]`       | 2,359,296        |
| BatchNorm2d-55      | `[-1, 512, 7, 7]`       | 1,024            |
| Conv2d-56           | `[-1, 512, 7, 7]`       | 131,072          |
| BatchNorm2d-57      | `[-1, 512, 7, 7]`       | 1,024            |
| ReLU-58             | `[-1, 512, 7, 7]`       | 0                |
| **BasicBlock-59**   | `[-1, 512, 7, 7]`       | 0                |
| Conv2d-60           | `[-1, 512, 7, 7]`       | 2,359,296        |
| BatchNorm2d-61      | `[-1, 512, 7, 7]`       | 1,024            |
| ReLU-62             | `[-1, 512, 7, 7]`       | 0                |
| Conv2d-63           | `[-1, 512, 7, 7]`       | 2,359,296        |
| BatchNorm2d-64      | `[-1, 512, 7, 7]`       | 1,024            |
| ReLU-65             | `[-1, 512, 7, 7]`       | 0                |
| **BasicBlock-66**   | `[-1, 512, 7, 7]`       | 0                |
| AdaptiveAvgPool2d-67| `[-1, 512, 1, 1]`       | 0                |
| Linear-68           | `[-1, 2]`               | 1,026            |
| **Total**           | —                        | **11,177,538**   |
| Parâmetros treináveis   | —                        | 1,026            |
| Parâmetros não-treináveis| —                        | 11,176,512       |


7. **Treinamento do Modelo**  
   - _Fine‑tune_ de ``ResNet-18`` pré-treinada (Outras topologias de rede podem ser utilizadas: VGG16, VGG19, U-net, etc.).
   - _Scheduler_ e otimização com `Adam` ou `SGD`.
   - Monitoramento de perda (_loss_) e acurácia (_accuracy_) por época.

8. **Avaliação do Modelo**  
   - Cálculo de Acurácia, Precisão, _Recall_, F1-Score.  
   - Matriz de confusão.
   - Curva ROC & AUC.

9. **Geração de Relatório com LLM**  
   - Envia métricas e gráficos ao Chat-GPT. 
   - Recebe _insights_ e dicas de _hyperparameter tuning_

10. **Exportação para PDF**  
   - Classe `PDF` customizada em FPDF.
   - Relatório final pronto para apresentação.
---

## 🤖 Modelo de LLM (Chat-GPT)

Utilizamos o **Chat-GPT** para interpretar resultados e gerar recomendações automáticas.

```python
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")
```

**Como funciona**:  
1. Prepara dicionário de métricas e gráficos  
2. Envia para o _endpoint_ de chat  
3. Recebe análise textual sobre performances e ajustes de parâmetro
---
