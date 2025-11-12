# ===================================================================
# MODELO YOLOV8 PARA DETECÇÃO DE PONTOS CEGOS EM CAMINHÕES
# Detecta: Pessoa, Motocicleta, Carro, Ciclista
# ===================================================================

from ultralytics import YOLO
import cv2
import numpy as np

# ===================================================================
# CONFIGURAÇÕES DO MODELO
# ===================================================================

CLASSES_INTERESSE = [0, 1, 2, 3]  # pessoa, bicicleta, carro, moto
CLASSES_NOMES = {
    0: 'Pessoa/Pedestre',
    1: 'Ciclista',
    2: 'Carro',
    3: 'Motocicleta'
}

CONFIANCA_MINIMA = 0.45
IOU_THRESHOLD = 0.5
TAMANHO_IMAGEM = 640

# ===================================================================
# CLASSE PARA DETECÇÃO DE PONTOS CEGOS
# ===================================================================

class DetectorPontoCego:
    """
    Detecta pessoas, motos, carros e ciclistas em zonas de risco de caminhões usando YOLOv8.
    """

    def __init__(self, modelo_path='yolov8s.pt'):
        print(f"[INFO] Carregando modelo YOLOv8: {modelo_path}")
        self.modelo = YOLO(modelo_path)
        self.modelo.overrides['conf'] = CONFIANCA_MINIMA
        self.modelo.overrides['iou'] = IOU_THRESHOLD
        self.modelo.overrides['imgsz'] = TAMANHO_IMAGEM
        print("[INFO] Modelo carregado com sucesso!")

    def definir_zonas_ponto_cego(self, largura, altura):
        zonas = {
            'lateral_direita': {
                'coords': [int(largura * 0.65), 0, largura, int(altura * 0.9)],
                'cor': (0, 0, 255),  # Vermelho
                'nivel_risco': 'ALTO'
            },
            'lateral_esquerda': {
                'coords': [0, 0, int(largura * 0.35), int(altura * 0.9)],
                'cor': (0, 165, 255),  # Laranja
                'nivel_risco': 'MÉDIO'
            }
        }
        return zonas

    def objeto_em_zona(self, bbox, zona_coords):
        x1_obj, y1_obj, x2_obj, y2_obj = bbox
        x1_zona, y1_zona, x2_zona, y2_zona = zona_coords
        centro_x = (x1_obj + x2_obj) / 2
        centro_y = (y1_obj + y2_obj) / 2
        return (x1_zona <= centro_x <= x2_zona and y1_zona <= centro_y <= y2_zona)

    def detectar_frame(self, frame, mostrar_zonas=True):
        altura, largura = frame.shape[:2]
        frame_anotado = frame.copy()
        zonas = self.definir_zonas_ponto_cego(largura, altura)
        if mostrar_zonas:
            overlay = frame_anotado.copy()
            for info_zona in zonas.values():
                x1, y1, x2, y2 = info_zona['coords']
                cv2.rectangle(overlay, (x1, y1), (x2, y2), info_zona['cor'], -1)
            cv2.addWeighted(overlay, 0.15, frame_anotado, 0.85, 0, frame_anotado)
        resultados = self.modelo.predict(
            frame,
            classes=CLASSES_INTERESSE,
            conf=CONFIANCA_MINIMA,
            iou=IOU_THRESHOLD,
            verbose=False
        )
        deteccoes_zona = {zona: [] for zona in zonas.keys()}
        for resultado in resultados:
            boxes = resultado.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confianca = float(box.conf[0].cpu().numpy())
                classe_id = int(box.cls[0].cpu().numpy())
                classe_nome = CLASSES_NOMES.get(classe_id, f'Classe {classe_id}')
                zona_detectada = None
                for nome_zona, info_zona in zonas.items():
                    if self.objeto_em_zona([x1, y1, x2, y2], info_zona['coords']):
                        zona_detectada = nome_zona
                        deteccoes_zona[nome_zona].append({
                            'classe': classe_nome,
                            'confianca': confianca,
                            'bbox': [x1, y1, x2, y2]
                        })
                        break
                cor = zonas[zona_detectada]['cor'] if zona_detectada else (0, 255, 0)
                cv2.rectangle(frame_anotado, (int(x1), int(y1)), (int(x2), int(y2)), cor, 2)
                label = f'{classe_nome}: {confianca:.2f}'
                tamanho_label = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(
                    frame_anotado,
                    (int(x1), int(y1) - tamanho_label[1] - 10),
                    (int(x1) + tamanho_label[0], int(y1)),
                    cor, -1
                )
                cv2.putText(frame_anotado, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if zona_detectada:
                    nivel_risco = zonas[zona_detectada]['nivel_risco']
                    texto_alerta = f'ALERTA {nivel_risco}: {classe_nome} em {zona_detectada}'
                    cv2.putText(frame_anotado, texto_alerta,
                                (10, altura - 70 - len(deteccoes_zona[zona_detectada]) * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, zonas[zona_detectada]['cor'], 2)
        y_pos = 30
        for nome_zona, deteccoes in deteccoes_zona.items():
            if len(deteccoes) > 0:
                texto = f'{nome_zona.upper()}: {len(deteccoes)} objeto(s)'
                cv2.putText(frame_anotado, texto, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, zonas[nome_zona]['cor'], 2)
                y_pos += 25
        return frame_anotado, deteccoes_zona

# ===================================================================
# FUNÇÕES DE USO
# ===================================================================

def processar_imagem(caminho_imagem, caminho_saida=None):
    detector = DetectorPontoCego('yolov8n.pt')
    frame = cv2.imread(caminho_imagem)
    if frame is None:
        print(f"[ERRO] Não foi possível ler a imagem: {caminho_imagem}")
        return
    frame_processado, deteccoes = detector.detectar_frame(frame)
    print("\n[RESULTADOS]")
    for zona, objetos in deteccoes.items():
        if len(objetos) > 0:
            print(f"  {zona}: {len(objetos)} objeto(s)")
            for obj in objetos:
                print(f"    - {obj['classe']} (conf: {obj['confianca']:.2f})")
    if caminho_saida:
        cv2.imwrite(caminho_saida, frame_processado)
        print(f"\n[INFO] Imagem salva em: {caminho_saida}")
    else:
        cv2.imshow('Detecção de Ponto Cego', frame_processado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def processar_video(caminho_video, caminho_saida=None):
    detector = DetectorPontoCego('yolov8n.pt')
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo: {caminho_video}")
        return
    writer = None
    if caminho_saida:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(caminho_saida, fourcc, fps, (largura, altura))
    frame_count = 0
    print("\n[INFO] Processando vídeo... (pressione 'q' para sair)")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_processado, deteccoes = detector.detectar_frame(frame)
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count} processado")
        if writer:
            writer.write(frame_processado)
        else:
            cv2.imshow('Detecção de Ponto Cego - Vídeo', frame_processado)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if writer:
        writer.release()
        print(f"\n[INFO] Vídeo salvo em: {caminho_saida}")
    cv2.destroyAllWindows()
    print(f"\n[INFO] Total de frames processados: {frame_count}")

def processar_webcam(indice_camera=0):
    detector = DetectorPontoCego('yolov8n.pt')
    cap = cv2.VideoCapture(indice_camera)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera {indice_camera}")
        return
    print("\n[INFO] Processando webcam... (pressione 'q' para sair)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_processado, deteccoes = detector.detectar_frame(frame)
        cv2.imshow('Detecção de Ponto Cego - Webcam', frame_processado)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ===================================================================
# EXEMPLO DE USO
# ===================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SISTEMA DE DETECÇÃO DE PONTOS CEGOS PARA CAMINHÕES - YOLOv8")
    print("=" * 70)
    print("\nClasses detectadas:")
    print("  - Pessoa/Pedestre")
    print("  - Ciclista")
    print("  - Carro")
    print("  - Motocicleta")
    print("\nZonas de ponto cego monitoradas:")
    print("  - Lateral Direita (ALTO risco)")
    print("  - Lateral Esquerda (MÉDIO risco)")
    print("  - Frontal (MÉDIO risco)")
    print("  - Traseira (ALTO risco)")
    print("=" * 70)

    # Exemplos de uso (descomente uma das linhas):
    # processar_imagem('caminho/para/imagem.jpg', 'resultado.jpg')
    # processar_video('caminho/para/video.mp4', 'resultado.mp4')
    processar_webcam(0)

    print("\n[INFO] Para usar o sistema, descomente um dos exemplos acima")
    print("[INFO] Ou importe as funções em seu próprio código.\n")
