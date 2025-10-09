from ultralytics import YOLO
import cv2
import numpy as np

# Carrega o modelo YOLOv8 pré-treinado (pode trocar para outro modelo se quiser)
model = YOLO('model/yolov8n.pt')

# Fator de calibração para estimar distância com base na altura da bounding box
# AJUSTE ESTE VALOR. Comece com algo em torno de 600 e calibre.
# Para calibrar: (altura do objeto em pixels a 1m) * 1.0
fator_calibracao = 600

# Abre a câmera (0 para webcam)
cap = cv2.VideoCapture(0)

def estimar_distancia_por_altura(box):
    altura_pix = abs(box[3] - box[1])
    if altura_pix == 0:
        return float('inf')  # evita divisão por zero
    distancia = fator_calibracao / altura_pix
    return distancia, altura_pix

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza a detecção no frame atual
    results = model(frame)

    # Extrai as bounding boxes detectadas
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    for box, cls, conf in zip(boxes, classes, confidences):
        # Filtra detecções com baixa confiança
        if conf < 0.5:
            continue

        distancia, altura_pix = estimar_distancia_por_altura(box)
        x1, y1, x2, y2 = map(int, box)

        # Desenha o retângulo e o texto para TODAS as detecções
        # para que você possa calibrar e ver o que está sendo detectado.
        label = model.names[int(cls)]
        texto_display = f"{label}: {distancia:.2f}m (h:{int(altura_pix)}px)"

        # Muda a cor se estiver perto
        cor = (0, 255, 0) # Verde por padrão
        if distancia <= 1.0:
            cor = (0, 0, 255) # Vermelho se perto

        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(frame, texto_display, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


    cv2.imshow('Deteccao de Objetos', frame)

    if cv2.waitKey(1) == 27:  # Sai com ESC
        break

cap.release()
cv2.destroyAllWindows()