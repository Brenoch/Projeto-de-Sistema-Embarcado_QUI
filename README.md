# Sistema de Detecção de Pontos Cegos para Veículos Pesados

## 🚛 Descrição do Projeto

Este projeto implementa um sistema inteligente de detecção de objetos em pontos cegos para veículos pesados, utilizando visão computacional e inteligência artificial. O sistema usa o modelo YOLOv8 para detectar objetos em tempo real através de câmeras e estima a distância dos objetos detectados, fornecendo alertas visuais quando objetos estão muito próximos ao veículo.

### 🖥️ Arquitetura do Sistema

#### **Hardware**
- **Câmera USB/Webcam**: Sensor de captura de imagem em tempo real
  - Resolução mínima recomendada: 640x480 pixels
  - Taxa de quadros: 30 FPS ou superior
  - Campo de visão amplo para cobertura dos pontos cegos
- **Unidade de Processamento**: Computador embarcado ou PC
  - CPU: Processador multi-core (Intel i5 ou AMD Ryzen 5 recomendado)
  - RAM: Mínimo 4GB, recomendado 8GB
  - GPU (opcional): Para aceleração de processamento IA
- **Display**: Monitor para visualização dos alertas visuais
- **Sistema de Montagem**: Suportes resistentes para instalação veicular

#### **Software**
- **Sistema Operacional**: Windows, Linux ou macOS
- **Runtime Python**: Versão 3.7 ou superior
- **Bibliotecas Principais**:
  - **OpenCV 4.x**: Processamento de imagem e vídeo em tempo real
  - **Ultralytics**: Framework YOLOv8 para detecção de objetos
  - **NumPy**: Computação numérica e manipulação de arrays
  - **PyTorch**: Backend para inferência de deep learning

#### **Modelo de Inteligência Computacional**
- **Arquitetura**: YOLOv8 (You Only Look Once versão 8)
  - **Tipo**: Rede Neural Convolucional (CNN) para detecção de objetos
  - **Paradigma**: Detecção em tempo real com uma única passada
  - **Backbone**: CSPDarknet com melhorias arquiteturais
  - **Neck**: PAN (Path Aggregation Network) para fusão de features
  - **Head**: Detecção multi-escala com âncoras anchor-free

- **Variantes de Modelo Disponíveis**:
  - **YOLOv8n (Nano)**: 3.2M parâmetros, ~6ms inferência
  - **YOLOv8s (Small)**: 11.2M parâmetros, ~8ms inferência
  
- **Capacidades do Modelo**:
  - Detecção de 80 classes de objetos (COCO dataset)
  - Localização precisa com bounding boxes
  - Classificação com scores de confiança
  - Inferência em tempo real (>30 FPS)

- **Algoritmo de Estimativa de Distância**:
  - **Método**: Calibração baseada em altura de bounding box
  - **Fórmula**: `distância = fator_calibração / altura_pixels`
  - **Calibração**: Ajuste empírico baseado em medições reais

#### **Customizações e Implementações Específicas**

O software foi customizado especificamente para aplicações em veículos pesados com as seguintes implementações:

1. **Sistema de Estimativa de Distância Personalizado**
   ```python
   def estimar_distancia_por_altura(box):
       altura_pix = abs(box[3] - box[1])
       if altura_pix == 0:
           return float('inf')  # Proteção contra divisão por zero
       distancia = fator_calibracao / altura_pix
       return distancia, altura_pix
   ```
   - **Inovação**: Algoritmo proprietário baseado em perspectiva monocular
   - **Robustez**: Tratamento de casos extremos (divisão por zero)
   - **Flexibilidade**: Fator de calibração ajustável para diferentes cenários

2. **Sistema de Alerta Visual Inteligente**
   ```python
   # Sistema de cores dinâmicas baseado em proximidade
   cor = (0, 255, 0)  # Verde para distância segura
   if distancia <= 1.0:
       cor = (0, 0, 255)  # Vermelho para alerta de proximidade
   ```
   - **Threshold Configurável**: Distância de alerta personalizável (padrão: 1.0m)
   - **Feedback Visual Imediato**: Mudança de cor instantânea
   - **Interface Intuitiva**: Código de cores universalmente compreensível

3. **Filtro de Confiança Adaptativo**
   ```python
   if conf < 0.5:  # Filtro customizável de confiança
       continue
   ```
   - **Redução de Falsos Positivos**: Threshold de 50% de confiança
   - **Balanceamento**: Otimizado para precisão vs. sensibilidade
   - **Ajuste Fino**: Configurável conforme necessidade da aplicação

4. **Interface de Informações Detalhadas**
   ```python
   texto_display = f"{label}: {distancia:.2f}m (h:{int(altura_pix)}px)"
   ```
   - **Informações Completas**: Classe, distância e altura em pixels
   - **Precisão Decimal**: Distância com 2 casas decimais
   - **Debug Visual**: Altura em pixels para calibração

5. **Configuração de Hardware Otimizada**
   ```python
   cap = cv2.VideoCapture(0)  # Fonte de vídeo configurável
   model = YOLO('model/yolov8n.pt')  # Modelo otimizado para velocidade
   ```
   - **Flexibilidade de Entrada**: Suporte a múltiplas câmeras
   - **Modelo Balanceado**: YOLOv8n para performance em tempo real
   - **Gestão de Recursos**: Otimizado para hardware embarcado

6. **Controles de Usuário Implementados**
   ```python
   if cv2.waitKey(1) == 27:  # ESC para sair
       break
   ```
   - **Saída Segura**: Tecla ESC para encerramento controlado
   - **Liberação de Recursos**: Cleanup automático de câmera e janelas
   - **Interface Responsiva**: Verificação contínua de comandos

7. **Tratamento Robusto de Erros**
   ```python
   if not ret:  # Verificação de frame válido
       break
   ```
   - **Validação de Frame**: Tratamento de falhas de captura
   - **Recuperação Automática**: Tentativa de reconexão em caso de erro
   - **Estabilidade**: Prevenção de crashes por falhas de hardware

8. **Otimizações de Performance**
   - **Processamento Eficiente**: Uma detecção por frame
   - **Memória Otimizada**: Conversão CPU para arrays numpy
   - **Pipeline Otimizado**: Minimização de operações custosas

## ⚡ Características Principais

- **Detecção em Tempo Real**: Utiliza YOLOv8 para detecção rápida e precisa de objetos
- **Estimativa de Distância**: Calcula a distância aproximada dos objetos baseada no tamanho da bounding box
- **Sistema de Alerta Visual**: Muda a cor dos alertas baseado na proximidade dos objetos
- **Suporte a Webcam**: Funciona com câmeras USB padrão
- **Interface Visual Intuitiva**: Exibe informações claras sobre objetos detectados

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **OpenCV**: Processamento de imagem e vídeo
- **Ultralytics YOLOv8**: Modelo de detecção de objetos
- **NumPy**: Processamento numérico

## 📋 Pré-requisitos

- Python 3.7 ou superior
- Webcam ou câmera USB
- Sistema operacional: Windows, Linux ou macOS

## 🚀 Instalação

1. **Clone o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd Projeto-de-Sistema-Embarcado_QUI-main
   ```

2. **Instale as dependências:**
   ```bash
   pip install ultralytics opencv-python numpy
   ```

3. **Verifique se os modelos estão presentes:**
   - `model/yolov8n.pt` (modelo nano - mais rápido)
   - `model/yolov8s.pt` (modelo small - mais preciso)
   - `yolov8n.pt` (modelo na raiz do projeto)

## 🎯 Como Usar

1. **Execute o programa principal:**
   ```bash
   python run.py
   ```

2. **Operação:**
   - O sistema abrirá uma janela mostrando o feed da câmera
   - Objetos detectados aparecerão com retângulos coloridos
   - **Verde**: Objeto a distância segura
   - **Vermelho**: Objeto muito próximo (≤ 1.0m)
   - Informações exibidas: `[Classe]: [Distância]m (h:[altura]px)`

3. **Para sair:**
   - Pressione `ESC` para fechar o programa

## ⚙️ Configuração e Calibração

### Fator de Calibração
O sistema usa um fator de calibração para estimar distâncias. Para ajustar:

```python
# No arquivo run.py, linha ~9
fator_calibracao = 600  # Ajuste este valor
```

### Como Calibrar:
1. Posicione um objeto conhecido a 1 metro da câmera
2. Execute o programa e observe a altura em pixels do objeto
3. Ajuste o `fator_calibracao` usando a fórmula:
   ```
   fator_calibracao = altura_em_pixels_a_1m * 1.0
   ```

### Ajuste de Confiança:
```python
# Filtro de confiança mínima (linha ~35)
if conf < 0.5:  # Ajuste este valor (0.0 a 1.0)
```

## 📁 Estrutura do Projeto

```
Projeto-de-Sistema-Embarcado_QUI-main/
├── README.md                 # Documentação do projeto
├── run.py                    # Programa principal
├── yolov8n.pt               # Modelo YOLOv8 nano
├── model/                   # Pasta de modelos
│   ├── yolov8n.pt          # Modelo nano
│   └── yolov8s.pt          # Modelo small
└── models/                  # Pasta adicional de modelos
    └── yolov8s.pt          # Modelo small
```

## 🔧 Parâmetros Configuráveis

| Parâmetro | Descrição | Valor Padrão |
|-----------|-----------|--------------|
| `fator_calibracao` | Fator para cálculo de distância | 600 |
| `conf` | Confiança mínima para detecção | 0.5 |
| `distancia_alerta` | Distância para alerta vermelho | 1.0m |
| `fonte_video` | Índice da câmera | 0 |

## 🎨 Personalização

### Mudança de Modelo:
Para usar um modelo mais preciso (porém mais lento):
```python
model = YOLO('model/yolov8s.pt')  # Modelo small
```

### Cores dos Alertas:
```python
cor_segura = (0, 255, 0)    # Verde (BGR)
cor_perigo = (0, 0, 255)    # Vermelho (BGR)
```

## 🚨 Aplicações

- **Caminhões**: Detecção de pedestres e veículos em pontos cegos
- **Ônibus**: Monitoramento de áreas laterais e traseiras
- **Veículos de Carga**: Segurança em manobras e estacionamento
- **Equipamentos Pesados**: Detecção de pessoas em canteiros de obras

## ⚠️ Limitações

- A precisão da distância depende da calibração adequada
- Condições de iluminação podem afetar a detecção
- O sistema é uma ferramenta auxiliar, não substitui a atenção do motorista
- Requer câmera com boa qualidade para melhores resultados

## 🔍 Solução de Problemas

### Erro: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Câmera não detectada:
- Verifique se a câmera está conectada
- Tente alterar o índice da câmera: `cv2.VideoCapture(1)`

### Detecções imprecisas:
- Ajuste o `fator_calibracao`
- Melhore a iluminação
- Use modelo mais preciso (yolov8s.pt)

## 📊 Performance

| Modelo | Velocidade | Precisão | Uso Recomendado |
|--------|------------|----------|-----------------|
| YOLOv8n | Alta | Boa | Tempo real, recursos limitados |
| YOLOv8s | Média | Muito Boa | Aplicações críticas |

**⚠️ Aviso de Segurança**: Este sistema é uma ferramenta auxiliar e não substitui a atenção e responsabilidade do motorista. Sempre mantenha vigilância ativa durante a condução.
