# 🚛 Sistema de Detecção de Pontos Cegos para Veículos Pesados

## 🧭 Visão Geral

Este projeto implementa um **sistema inteligente de detecção de objetos em pontos cegos** voltado para **veículos pesados**, utilizando **visão computacional** e **inteligência artificial**.
O sistema baseia-se no modelo **YOLOv8** para detectar objetos em tempo real a partir de câmeras, estimando a distância e emitindo **alertas visuais** quando há risco de proximidade.

---

## 🧩 Arquitetura do Sistema

### 🖥️ **Hardware**

| Componente                   | Descrição                                     | Especificação Recomendada                                  |
| ---------------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| **Câmera USB/Webcam**        | Captura de imagem em tempo real               | ≥ 640x480 px, 30 FPS, campo de visão amplo                 |
| **Unidade de Processamento** | Execução da inferência e do pipeline de vídeo | CPU multi-core (Intel i5 / Ryzen 5), RAM 8GB, GPU opcional |
| **Display/Monitor**          | Exibição dos alertas e informações            | Resolução mínima HD                                        |
| **Sistema de Montagem**      | Fixação da câmera e display em veículos       | Suporte resistente à vibração                              |

---

### 💻 **Software**

| Item                              | Descrição                                          |
| --------------------------------- | -------------------------------------------------- |
| **Sistema Operacional**           | Compatível com **Windows**, **Linux** ou **macOS** |
| **Linguagem**                     | **Python 3.7+**                                    |
| **Principais Bibliotecas**        | `opencv-python`, `ultralytics`, `numpy`, `torch`   |
| **Framework de IA**               | **Ultralytics YOLOv8**                             |
| **Gerenciamento de Dependências** | `pip`                                              |

---

### 🤖 **Modelo de Inteligência Computacional**

| Parâmetro                | Descrição                                 |
| ------------------------ | ----------------------------------------- |
| **Modelo Base**          | YOLOv8 (You Only Look Once v8)            |
| **Tipo**                 | CNN (Rede Neural Convolucional)           |
| **Paradigma**            | Detecção em tempo real (single-pass)      |
| **Backbone**             | CSPDarknet com aprimoramentos estruturais |
| **Head**                 | Detecção multi-escala (anchor-free)       |
| **Treinamento Original** | Dataset COCO (80 classes)                 |

**Modelos Suportados:**

* `yolov8n.pt` – leve, rápido (~6ms/frame)
* `yolov8s.pt` – mais preciso (~8ms/frame)

**Capacidades:**

* Detecção de múltiplas classes simultâneas
* Estimativa de distância baseada em escala de bounding box
* Operação em tempo real (>30 FPS)

---

## ⚙️ Customizações e Esforços de Desenvolvimento

O sistema foi **personalizado e aprimorado** para uso em **veículos pesados**, com foco em robustez, precisão e responsividade.
A seguir, as principais customizações realizadas sobre o código padrão:

### 1. 🧮 Algoritmo de Estimativa de Distância

```python
def estimar_distancia_por_altura(box):
    altura_pix = abs(box[3] - box[1])
    if altura_pix == 0:
        return float('inf')
    distancia = fator_calibracao / altura_pix
    return distancia, altura_pix
```

* Baseado em **projeção monocular e calibração empírica**
* Proteção contra divisão por zero
* Fator de calibração configurável via código

---

### 2. 🚨 Sistema de Alerta Visual Inteligente

```python
cor = (0, 255, 0)
if distancia <= 1.0:
    cor = (0, 0, 255)
```

* Feedback visual instantâneo
* Limiar configurável para alertas (padrão: 1.0m)
* Uso de **código de cores universal (verde → seguro, vermelho → alerta)**

---

### 3. 🎯 Filtro de Confiança Adaptativo

```python
if conf < 0.5:
    continue
```

* Threshold ajustável para reduzir falsos positivos
* Equilíbrio entre precisão e sensibilidade

---

### 4. 🧾 Interface de Informação Detalhada

```python
texto_display = f"{label}: {distancia:.2f}m (h:{int(altura_pix)}px)"
```

* Mostra **classe, distância e altura do objeto em pixels**
* Ideal para calibração e depuração

---

### 5. ⚡ Otimizações de Performance

* Processamento **1 detecção/frame**
* Uso eficiente de memória via `numpy`
* Pipeline enxuto para **tempo real em hardware limitado**

---

### 6. 🧰 Robustez e Usabilidade

* Tratamento de erro em falhas de captura (`if not ret: break`)
* Encerramento seguro via tecla `ESC`
* Suporte a múltiplas câmeras configuráveis

---

## 🚀 Instruções de Instalação

1. **Clone o repositório**

   ```bash
   git clone <[url-do-repositorio](https://github.com/Brenoch/Projeto-de-Sistema-Embarcado_QUI.git)>
   cd Projeto-de-Sistema-Embarcado_QUI-main
   ```

2. **Instale as dependências**

   ```bash
   pip install ultralytics opencv-python numpy torch
   ```

3. **Verifique os modelos**

   * `model/yolov8n.pt`
   * `model/yolov8s.pt`

---

## ▶️ Execução do Sistema

```bash
python run.py
```

Durante a execução:

* Objetos detectados aparecerão com **caixas coloridas**
* **Verde** → distância segura
* **Vermelho** → risco de colisão (≤ 1.0m)
* Para encerrar: pressione `ESC`

---

## ⚙️ Calibração

1. Coloque um objeto conhecido a **1 metro** da câmera
2. Observe a altura em pixels
3. Ajuste o parâmetro no código:

```python
fator_calibracao = altura_em_pixels_a_1m * 1.0
```

---

## 🧱 Estrutura do Projeto

```
Projeto-de-Sistema-Embarcado_QUI-main/
├── README.md
├── run.py
├── model/
│   ├── yolov8n.pt
│   └── yolov8s.pt
└── models/
    └── yolov8s.pt
```

---

## 🔧 Parâmetros Configuráveis

| Parâmetro          | Descrição                           | Valor Padrão |
| ------------------ | ----------------------------------- | ------------ |
| `fator_calibracao` | Constante para cálculo de distância | 600          |
| `conf`             | Confiança mínima da detecção        | 0.5          |
| `distancia_alerta` | Limiar de alerta (m)                | 1.0          |
| `fonte_video`      | Índice da câmera                    | 0            |

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.7+**
* **OpenCV 4.x** – Processamento de vídeo
* **Ultralytics YOLOv8** – Detecção de objetos
* **NumPy** – Manipulação numérica
* **PyTorch** – Inferência de redes neurais

---

## 💡 Aplicações

* **Caminhões e ônibus** → Detecção de pedestres e veículos laterais
* **Veículos de carga** → Apoio em manobras e estacionamento
* **Equipamentos pesados** → Segurança em áreas industriais

---

## ⚠️ Limitações

* A precisão depende da **calibração adequada**
* Condições de **iluminação** interferem na detecção
* O sistema é **auxiliar** e não substitui a atenção do condutor

---

## 🧰 Solução de Problemas

| Problema                        | Solução                                 |
| ------------------------------- | --------------------------------------- |
| `No module named 'ultralytics'` | `pip install ultralytics`               |
| Câmera não detectada            | Alterar `cv2.VideoCapture(1)`           |
| Detecção imprecisa              | Ajustar `fator_calibracao` e iluminação |

---

## 📊 Desempenho dos Modelos

| Modelo      | Velocidade | Precisão  | Recomendado Para       |
| ----------- | ---------- | --------- | ---------------------- |
| **YOLOv8n** | Alta       | Boa       | Tempo real / embarcado |
| **YOLOv8s** | Média      | Excelente | Ambientes críticos     |
