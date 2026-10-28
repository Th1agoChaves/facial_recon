from pathlib import Path
import cv2
from deepface import DeepFace

#Diretorios do projeto
BASE_DIR= Path(__file__).resolve().parent
AUTHORIZED_DIR= BASE_DIR/ "autorizado"
TEST_DIR= BASE_DIR/ "teste"
RESULT_DIR= BASE_DIR/ "resultado"

#Configuracao do reconhecimento
MODEL_NAME= "Facenet512"
DETECTOR_BACKEND= "opencv"

def supported_image(path: Path)-> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_authorized_images():
    """Retorna os caminhos das imagens de referencia disponiveis."""
    return [
        path for path in AUTHORIZED_DIR.iterdir()
        if path.is_file() and supported_image(path)
    ] if AUTHORIZED_DIR.exists() else []

def detect_faces(image):
    """Detecta rostos usando DeepFace e retorna (face, area) para cada rosto."""
    detections= DeepFace.extract_faces(
        img_path=           image,
        detector_backend=   DETECTOR_BACKEND,
        enforce_detection=  False,
        align=              True
    )

    found= []
    image_h, image_w= image.shape[:2]

    for detection in detections:
        area=   detection.get("facial_area", {})
        x=      max(0, int(area.get("x", 0)))
        y=      max(0, int(area.get("y", 0)))
        w=      int(area.get("w", 0))
        h=      int(area.get("h", 0))

        #extract_faces pode retornar a imagem inteira quando nao detecta face
        #e enforce_detection=False. Ignoramos areas invalidas/degeneradas.
        if w<= 0 or h<= 0:
            continue
        x2, y2= min(image_w, x + w), min(image_h, y + h)
        if x2<= x or y2<= y:
            continue

        crop= image[y:y2, x:x2]
        if crop.size== 0:
            continue

        #Evita considerar a imagem inteira como rosto em casos sem deteccao.
        if w>= image_w * 0.95 and h>= image_h * 0.95:
            continue

        found.append((crop, (x, y, x2, y2)))

    return found

def is_authorized(face_crop, reference_paths):
    """Compara um rosto detectado com todas as fotos autorizadas."""
    for reference_path in reference_paths:
        try:
            result = DeepFace.verify(
                img1_path=          face_crop,
                img2_path=          str(reference_path),
                model_name=         MODEL_NAME,
                detector_backend=   "skip",
                enforce_detection=  False,
                align=              False,
                silent=             True
            )
            if result.get("verified", False):
                return True, reference_path.stem
        except Exception as exc:
            print(f"  Aviso: nao foi possivel comparar com {reference_path.name}: {exc}")
    return False, None

def process_image(image_path: Path, reference_paths):
    image= cv2.imread(str(image_path))
    if image is None:
        print(f"Pulando arquivo que nao pode ser lido: {image_path.name}")
        return

    try:
        faces= detect_faces(image)
    except Exception as exc:
        print(f"Erro ao detectar rostos em {image_path.name}: {exc}")
        return

    if not faces:
        print(f"Nenhum rosto detectado em {image_path.name}.")
        #Salva a imagem com indicacao visual de que nao houve deteccao.
        cv2.putText(
            image, "NENHUM ROSTO DETECTADO", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )

    for face_crop, (x1, y1, x2, y2) in faces:
        allowed, person_name= is_authorized(face_crop, reference_paths)
        color= (0, 180, 0) if allowed else (0, 0, 255)  #RGB: verde ou vermelho
        label= "ACESSO LIBERADO" if allowed else "ACESSO NEGADO"

        #Bounding box e etiqueta preenchida para garantir legibilidade.
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y1, text_h + baseline + 8)
        cv2.rectangle(
            image,
            (x1, label_y - text_h - baseline - 6),
            (x1 + text_w + 10, label_y + baseline - 2),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            image, label, (x1 + 5, label_y - 4),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        if allowed and person_name:
            print(f"{image_path.name}: acesso liberado (referencia: {person_name})")
        else:
            print(f"{image_path.name}: acesso negado")

    output_path = RESULT_DIR / f"{image_path.stem}_resultado{image_path.suffix}"
    if not cv2.imwrite(str(output_path), image):
        print(f"Nao foi possivel salvar o resultado: {output_path}")
    else:
        print(f"Resultado salvo em: {output_path}")

def main():
    AUTHORIZED_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    reference_paths= load_authorized_images()
    test_paths= sorted(
        path for path in TEST_DIR.iterdir()
        if path.is_file() and supported_image(path)
    )

    if not reference_paths:
        print("Nenhuma foto autorizada encontrada na pasta 'autorizado/'.")
        print("Adicione pelo menos uma foto por pessoa antes de executar.")
        return
    if not test_paths:
        print("Nenhuma foto de teste encontrada na pasta 'teste/'.")
        print("Adicione imagens .jpg, .jpeg, .png, .bmp ou .webp.")
        return

    print(f"Fotos de referencia: {len(reference_paths)}")
    print(f"Fotos para testar: {len(test_paths)}")

    for image_path in test_paths:
        process_image(image_path, reference_paths)

if __name__ == "__main__":
    main()