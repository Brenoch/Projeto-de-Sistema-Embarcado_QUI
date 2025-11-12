# ===================================================================
# MODELO YOLOV8 PARA DETECÇÃO DE PONTOS CEGOS EM CAMINHÕES
# Detecta: Pessoa, Motocicleta, Carro, Ciclista
# ===================================================================

from ultralytics import YOLO
import cv2
import numpy as np
import math

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
# FUNÇÕES DE AVALIAÇÃO (IoU, emparelhamento, métricas, matriz confusão)
# ===================================================================

def iou(boxA, boxB):
	"""
	Box format: [x1, y1, x2, y2]
	"""
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interW = max(0.0, xB - xA)
	interH = max(0.0, yB - yA)
	interArea = interW * interH
	if interArea == 0:
		return 0.0
	boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
	boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
	return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def match_and_build_confusion(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, num_classes, iou_thresh=0.5):
	"""
	Emparelha predições com ground-truths usando IoU >= iou_thresh.
	Retorna TP/FP/FN por classe, matriz de confusão (true x pred) e confidências das predições classificadas como TP.
	Input lists correspondem a uma imagem; use em loop para dataset.
	"""
	# inicializa contadores
	TP = np.zeros(num_classes, dtype=int)
	FP = np.zeros(num_classes, dtype=int)
	FN = np.zeros(num_classes, dtype=int)
	confidences_tp = [[] for _ in range(num_classes)]
	confusion = np.zeros((num_classes, num_classes), dtype=int)  # [true][pred]

	gt_matched = [False] * len(gt_boxes)

	# ordenar predições por score decrescente
	order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
	for pi in order:
		pbox = pred_boxes[pi]
		plabel = pred_labels[pi]
		best_iou = 0.0
		best_gi = -1
		for gi, (gbox, glabel) in enumerate(zip(gt_boxes, gt_labels)):
			if gt_matched[gi]:
				continue
			cur_iou = iou(pbox, gbox)
			if cur_iou > best_iou:
				best_iou = cur_iou
				best_gi = gi
		if best_iou >= iou_thresh and best_gi >= 0:
			true_label = gt_labels[best_gi]
			gt_matched[best_gi] = True
			confusion[true_label, plabel] += 1
			if true_label == plabel:
				TP[plabel] += 1
				confidences_tp[plabel].append(pred_scores[pi])
			else:
				# class mismatch: conta como FP para predição e FN para GT
				FP[plabel] += 1
				FN[true_label] += 1
		else:
			# nenhuma GT correspondente -> FP para predição
			FP[plabel] += 1

	# GTs não casados são FN
	for gi, matched in enumerate(gt_matched):
		if not matched:
			FN[gt_labels[gi]] += 1

	return {
		'TP': TP,
		'FP': FP,
		'FN': FN,
		'confidences_tp': confidences_tp,
		'confusion': confusion
	}

def aggregate_and_compute_metrics(per_image_stats, class_id_list):
	"""
	Acumula estatísticas de várias imagens e calcula precision, recall, f1 e confidence médio por classe.
	class_id_list: list/iterable de ids de classe (ex: CLASSES_NOMES.keys()).
	"""
	num_classes = max(class_id_list) + 1
	total_TP = np.zeros(num_classes, dtype=int)
	total_FP = np.zeros(num_classes, dtype=int)
	total_FN = np.zeros(num_classes, dtype=int)
	total_conf_tp = [[] for _ in range(num_classes)]
	total_confusion = np.zeros((num_classes, num_classes), dtype=int)

	for stats in per_image_stats:
		total_TP += stats['TP']
		total_FP += stats['FP']
		total_FN += stats['FN']
		for cid in range(num_classes):
			total_conf_tp[cid].extend(stats['confidences_tp'][cid])
		total_confusion += stats['confusion']

	precision = {}
	recall = {}
	f1 = {}
	avg_confidence = {}
	for cid in class_id_list:
		tp = int(total_TP[cid])
		fp = int(total_FP[cid])
		fn = int(total_FN[cid])
		prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
		conf_list = total_conf_tp[cid]
		avg_conf = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
		precision[cid] = prec
		recall[cid] = rec
		f1[cid] = f1score
		avg_confidence[cid] = avg_conf

	return {
		'precision': precision,
		'recall': recall,
		'f1': f1,
		'avg_confidence': avg_confidence,
		'confusion_matrix': total_confusion,
		'TP': total_TP,
		'FP': total_FP,
		'FN': total_FN
	}

def evaluate_annotations(gt_annotations, pred_annotations, class_id_list=None, iou_thresh=0.5):
	"""
	gt_annotations: list de dicts {'boxes': [[x1,y1,x2,y2],...], 'labels': [int,...]}
	pred_annotations: list de dicts {'boxes': [[...],...], 'labels': [...], 'scores': [...]}
	Ambos os arrays devem ter mesma ordem/len (uma entrada por imagem).
	Retorna métricas agregadas e matriz de confusão.
	"""
	assert len(gt_annotations) == len(pred_annotations), "GT e predictions devem ter mesmo nº de imagens"
	if class_id_list is None:
		# inferir do CLASSES_NOMES por padrão
		class_id_list = sorted(list(CLASSES_NOMES.keys()))
	num_classes = max(class_id_list) + 1

	per_image_stats = []
	for gt, pred in zip(gt_annotations, pred_annotations):
		gt_boxes = gt.get('boxes', [])
		gt_labels = gt.get('labels', [])
		pred_boxes = pred.get('boxes', [])
		pred_labels = pred.get('labels', [])
		pred_scores = pred.get('scores', [1.0]*len(pred_boxes))
		stats = match_and_build_confusion(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, num_classes, iou_thresh=iou_thresh)
		per_image_stats.append(stats)

	metrics = aggregate_and_compute_metrics(per_image_stats, class_id_list)
	return metrics

def print_metrics(metrics, class_id_list=None):
	"""
	Imprime resumo das métricas por classe e a matriz de confusão.
	"""
	if class_id_list is None:
		class_id_list = sorted(list(CLASSES_NOMES.keys()))
	print("\n[AVALIAÇÃO] Métricas por classe:")
	for cid in class_id_list:
		name = CLASSES_NOMES.get(cid, str(cid))
		print(f"  {cid} - {name}: Precision={metrics['precision'][cid]:.3f}, Recall={metrics['recall'][cid]:.3f}, F1={metrics['f1'][cid]:.3f}, AvgConf={metrics['avg_confidence'][cid]:.3f}")
	print("\nMatriz de Confusão (linhas=verdadeiras, colunas=preditas):")
	print(metrics['confusion_matrix'])
	return

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

    # Exemplo de uso (comentado):
    # gt_ann = [
    #     {'boxes': [[50,50,150,200]], 'labels': [0]},
    #     ...
    # ]
    # pred_ann = [
    #     {'boxes': [[52,48,148,198]], 'labels': [0], 'scores':[0.8]},
    #     ...
    # ]
    # metrics = evaluate_annotations(gt_ann, pred_ann, class_id_list=list(CLASSES_NOMES.keys()), iou_thresh=0.5)
    # print_metrics(metrics)
