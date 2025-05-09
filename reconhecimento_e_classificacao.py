# -*- coding: utf-8 -*-
"""aula04_reconhecimento_e_classificacao.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GMh38lBrz80SMz3j-h-95ZFkubRU15R3

# **Sistema de Classificação de Imagens Médicas com ResNet-18 e Análise com LLM**

Este notebook implementa um sistema completo para:

1. Carregar um dataset médico de imagens (usaremos o dataset de raios-X de pneumonia)
2. Pré-processamento das imagens
3. Segmentação (se aplicável)
4. Extração de características e classificação usando ResNet-18 com transferência de aprendizado
5. Avaliação com métricas completas
6. Geração de relatório com análise por LLM (Chat-GPT)
7. Exportação para PDF

### **Instalando os pacotes adicionais**
"""

# Instalação de pacotes adicionais
!pip install -q torch torchvision torchsummary
!pip install -q matplotlib seaborn sklearn
!pip install -q fpdf2
# Atualizar a instalação do pacote openai
!pip install -q --upgrade openai

"""### **Importando as bibliotecas necessárias e Configurando o Ambiente**"""

# Importações necessárias
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, models
from torchsummary import summary
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix)
from PIL import Image
import requests
from io import BytesIO
import zipfile
import time
import openai
from fpdf import FPDF
from IPython.display import display, Markdown
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Carregando apenas o dataset de treino e dividir em treino/validação
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

# Verificar se CUDA está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de execução: {device}")

"""## **1. Download e Preparação do Dataset**
Vamos usar o dataset "Chest X-Ray Images (Pneumonia)" disponível no Kaggle.
Como alternativa, usaremos uma versão disponível publicamente.
"""

# Download do dataset (versão pública alternativa)
!wget -q https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded -O chest_xray.zip

# Extrair o arquivo zip
with zipfile.ZipFile('chest_xray.zip', 'r') as zip_ref:
    zip_ref.extractall('chest_xray')

# Verificar a estrutura do diretório
base_dir = 'chest_xray/chest_xray'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

print("Diretórios existentes:")
print(f"Treino: {train_dir} - Existe: {os.path.exists(train_dir)}")
print(f"Teste: {test_dir} - Existe: {os.path.exists(test_dir)}")

# Fazendo verificação das classes: PNEUMONIA e NORMAL
classes = os.listdir(train_dir)
print(f"Classes: {classes}")

"""## **2. Pré-processamento de Dados**

Vamos definir transformações para normalizar e aumentar os dados.
"""

# Definir transformações
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Carregabdo o dataset de treino completo
full_train = ImageFolder(train_dir, data_transforms['train'])

# Dividindo em treino (80%) e validação (20%), Se quiser, pode dividir em Treino, teste e validação.
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

# Carregando o dataset de teste
test_dataset = ImageFolder(test_dir, data_transforms['test'])

# Atualizando os transforms para os conjuntos divididos
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

# Criando dataloaders
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
class_names = full_train.classes

print(f"Tamanho dos datasets: {dataset_sizes}")
print(f"Nomes das classes: {class_names}")

# Função para visualizar imagens (mantida a mesma)
def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# Pegar um batch de imagens de treino
inputs, classes = next(iter(dataloaders['train']))

# Selecionar apenas as primeiras 4 imagens do batch
inputs = inputs[:4]
classes = classes[:4]

# Criar figura com 4 subplots
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i+1)  # 1 linha, 4 colunas, posição i+1
    imshow(inputs[i], title=class_names[classes[i]])

plt.tight_layout()  # Ajusta o espaçamento entre as imagens
plt.show()

"""## **3. Modelo ResNet-18 com Transferência de Aprendizado**
Vamos carregar uma ResNet-18 pré-treinada e adaptá-la para nosso problema.
"""

# Carregando o modelo pré-treinado
model = models.resnet18(pretrained=True)

# Congelando todos os parâmetros
for param in model.parameters():
    param.requires_grad = False

# Substituindo a camada fully connected
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

# MMostra o resumo do modelo
summary(model, (3, 224, 224))

# Definindo a função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Atualização do learning rate ou taxa de aprendizagem do modelo
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

"""## **4. Treinamento do Modelo**"""

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Função para treinar o modelo de aprendizado profundo

    Parâmetros:
    model: modelo neural a ser treinado
    criterion: função de perda
    optimizer: otimizador para atualização dos pesos
    scheduler: agendador para ajuste do learning rate
    num_epochs: número de épocas de treinamento
    """

    since = time.time()  # Marca o início do treinamento

    # Inicializa os melhores pesos como uma cópia dos pesos atuais do modelo
    best_model_wts = model.state_dict().copy()
    best_acc = 0.0  # Melhor acurácia inicializada como zero

    # Loop pelas épocas de treinamento
    for epoch in range(num_epochs):
        print(f'Época {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Cada época tem fase de treino e validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Configura o modelo para modo de treino
            else:
                model.eval()    # Configura o modelo para modo de avaliação

            # Inicializa métricas
            running_loss = 0.0
            running_corrects = 0

            # Itera sobre os dados
            for inputs, labels in dataloaders[phase]:
                # Move os dados para o dispositivo (GPU/CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zera os gradientes dos parâmetros
                optimizer.zero_grad()

                # Forward (passagem para frente)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Calcula as saídas
                    _, preds = torch.max(outputs, 1)  # Obtém as previsões
                    loss = criterion(outputs, labels)  # Calcula a perda

                    # Backward (retropropagação) + otimização apenas na fase de treino
                    if phase == 'train':
                        loss.backward()  # Calcula gradientes
                        optimizer.step()  # Atualiza pesos

                # Atualiza estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Ajusta o learning rate na fase de treino
            if phase == 'train':
                scheduler.step()

            # Calcula métricas da época
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Perda: {epoch_loss:.4f} Acurácia: {epoch_acc:.4f}')

            # Atualiza os melhores pesos se a acurácia de validação melhorou
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()  # Faz cópia dos pesos atuais

        print()  # Espaço entre épocas

    # Exibe estatísticas finais do treinamento
    time_elapsed = time.time() - since
    print(f'Treinamento completo em {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Melhor acurácia de validação: {best_acc:4f}')

    # Carrega os melhores pesos encontrados no modelo
    model.load_state_dict(best_model_wts)

    return model  # Retorna o modelo treinado

# Treinar o modelo
model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

# Salvar o modelo treinado
torch.save(model.state_dict(), 'pneumonia_resnet18.pth')

"""## **5. Avaliação do Modelo**
Vamos calcular todas as métricas solicitadas no conjunto de teste.
"""

def evaluate_model(model, dataloader):


    # Configura o modelo para modo de avaliação (desativa dropout, batch norm, etc.)
    model.eval()

    # Listas para armazenar resultados
    all_preds = []      # Armazenará as previsões do modelo
    all_labels = []     # Armazenará os rótulos verdadeiros
    all_probs = []      # Armazenará as probabilidades de cada classe

    # Desativa cálculo de gradientes para maior eficiência
    with torch.no_grad():
        # Itera sobre os lotes de dados
        for inputs, labels in dataloader:
            # Move os dados para o dispositivo (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - calcula as saídas do modelo
            outputs = model(inputs)

            # Obtém as previsões (índices das classes com maior probabilidade)
            _, preds = torch.max(outputs, 1)

            # Calcula as probabilidades usando softmax (converte saídas em probabilidades [0-1])
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Armazena resultados convertendo para numpy e movendo para CPU
            all_preds.extend(preds.cpu().numpy())       # Adiciona previsões do lote atual
            all_labels.extend(labels.cpu().numpy())      # Adiciona rótulos verdadeiros
            all_probs.extend(probs.cpu().numpy())        # Adiciona probabilidades das classes

    # Retorna todos os resultados coletados
    return all_labels, all_preds, all_probs

# Executar essa função, caso a função acima não funcione
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_labels, all_preds, all_probs

# Avaliar no conjunto de teste
test_labels, test_preds, test_probs = evaluate_model(model, dataloaders['test'])

# Cálculo das métricas:

# Acurácia:
accuracy = accuracy_score(test_labels, test_preds)

# Precisão:
precision = precision_score(test_labels, test_preds, average='weighted')

# Recall
recall = recall_score(test_labels, test_preds, average='weighted')

# f1-score
f1 = f1_score(test_labels, test_preds, average='weighted')

print(f'Acurácia: {accuracy:.4f}')
print(f'Precisão: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Matriz de confusão
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# Curva ROC e AUC
fpr = {}
tpr = {}
roc_auc = {}

# Calcular para cada classe
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve((np.array(test_labels) == i).astype(int),
                                 np.array(test_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotar todas as curvas ROC
plt.figure(figsize=(8,6))
colors = ['blue', 'red']
for i, color in zip(range(len(class_names)), colors):
    plt.plot(fpr[i], tpr[i], color=color,
             label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC Multiclasse')
plt.legend(loc="lower right")
plt.show()

"""## **6. Pós-processamento e Análise com LLM**
Vamos usar o Chat-GPT para analisar os resultados e gerar recomendações.
"""

OPENAI_API_KEY = 'sua_API_KEY_AQUI'

# Configurando a API do OpenAI (versão atualizada)
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# Função para gerar relatório com agente de IA
def generate_chatgpt_report(metrics, cm, roc_auc, class_names):
    # Preparar dados para o prompt
    metrics_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    cm_str = "\n".join([f"{class_names[i]}: {row}" for i, row in enumerate(cm)])
    auc_str = "\n".join([f"{class_names[i]}: {roc_auc[i]:.4f}" for i in range(len(class_names))])

    prompt = f"""
    Você é um especialista em aprendizado profundo (Deep Learning) aplicado à medicina. Analise os resultados de um modelo de classificação de
    imagens médicas (raios-X de tórax para diagnóstico de pneumonia) e forneça um relatório detalhado com recomendações para melhorias,
    incluindo análise de desempenho, matriz de confusão, curvas ROC e AUC. Ademais, verifique possíveis melhorias no pré-processamento e
    no modelo de rede neural convolucional proposto.

    **Contexto:**
    - Modelo: ResNet-18 com transferência de aprendizado
    - Classes: {', '.join(class_names)}
    - Dataset: Raios-X de tórax para diagnóstico de pneumonia


    **Métricas do Modelo:**
    {metrics_str}

    **Matriz de Confusão:**
    {cm_str}

    **AUC para cada classe:**
    {auc_str}

    **Forneça:**
    1. ANÁLISE GERAL DO DESEMPENHO DO MODELO
    2. PONTOS FORTES E FRACOS BASEADOS NAS MÉTRICAS
    3. ANÁLISE DA MATRIZ DE CONFUSÃO - ONDE O MODELO MAIS ERRA?
    4. INTERPRETAÇÃO DAS CURVAS ROC E AUC
    5. RECOMENDAÇÕES TÉCNICAS PARA MELHORAR O MODELO
    6. SUGESTÕES PARA AUMENTO/BALANCEAMENTO DOS DADOS
    7. CONSIDERAÇÕES CLÍNICAS SOBRE A APLICAÇÃO PRÁTICA
    8. LIMITAÇÕES ÉTICAS E POSSÍVEIS VIÉSES
    9. CONCLUSÕES E PARECER FINAL

    O relatório deve ser técnico mas claro, com foco em insights acionáveis para melhorar o modelo. Não utilize ** para deixar o texto em negrito.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um especialista em aprendizado profundo (Deep Learning) aplicado à medicina com vasta experiência em classificação de imagens médicas."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=5000
    )

    return response.choices[0].message.content

# Gerar o relatório com ChatGPT
metrics = {
    'Acurácia': accuracy,
    'Precisão': precision,
    'Recall': recall,
    'F1-Score': f1
}

chatgpt_report = generate_chatgpt_report(metrics, cm, roc_auc, class_names)

# Mostrar o relatório
display(Markdown("# BioDatA - Biomedical Data Analytics Research Group"))
display(Markdown("## Relatório de Classificação de Imagens Médicas"))
display(Markdown(chatgpt_report))

"""## 7. **Geração do Relatório em PDF**"""

# Criando a classe PDF com as especificações do documento.
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        # cabeçaho do documento
        self.cell(0, 12, 'BioDatA - Biomedical Data Analytics Research Group', 0, 1, 'C')
        self.cell(0, 10, 'Relatório de Classificação de Imagens Médicas', 0, 1, 'C')

        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

# Função para gerar o PDF
def create_pdf_report_with_chatgpt(metrics, cm, roc_auc, report, class_names):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Seção de Métricas
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Métricas de Desempenho:", ln=1)
    pdf.set_font("Arial", size=12)

    for metric, value in metrics.items():
        pdf.cell(0, 10, f"{metric}: {value:.4f}", ln=1)

    pdf.ln(10)

    # Seção de Matriz de Confusão
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Matriz de Confusão", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Arial", size=12)

    # Ajuste o tamanho das colunas
    col_width = pdf.w / (len(class_names) + 1)  # Colunas mais estreitas
    row_height = 6  # Altura menor

    # Cabeçalho com texto quebrado
    pdf.cell(col_width, row_height, "", border=1)
    for class_name in class_names:
      pdf.cell(col_width, row_height, class_name[:5], border=1, align='C')  # Texto mais curto
    pdf.ln(row_height)

    # Linhas da matriz
    for i, row in enumerate(cm):
      pdf.cell(col_width, row_height, class_names[i][:5], border=1)
      for val in row:
        pdf.cell(col_width, row_height, str(val), border=1, align='C')
      pdf.ln(row_height)

    # Seção AUC
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Área sob a Curva ROC (AUC)", ln=1)
    pdf.set_font("Arial", size=12)

    for i, auc_val in roc_auc.items():
        pdf.cell(0, 10, f"{class_names[i]}: {auc_val:.4f}", ln=1)

    pdf.ln(10)

    # Seção de Análise
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Análise e Recomendações geradas por IA", ln=1)
    pdf.set_font("Arial", size=12)

    # Adicionar o relatório em parágrafos
    paragraphs = report.split('\n\n')
    for para in paragraphs:
        if para.strip():
            pdf.multi_cell(0, 10, para.strip())
            pdf.ln(5)

    # Salvar o PDF
    pdf.output("relatorio_classificacao_medica_chatgpt.pdf")
    return "relatorio_classificacao_medica_chatgpt.pdf"

# Gerar o PDF com análise do ChatGPT
pdf_path = create_pdf_report_with_chatgpt(metrics, cm, roc_auc, chatgpt_report, class_names);
print(f"Relatório gerado e salvo como: {pdf_path}");