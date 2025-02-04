import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
from onvif import ONVIFCamera

# Configurações da câmera ONVIF
IP = "192.168.1.64"  # Substitua pelo IP da sua câmera
PORT = 80  # Porta da câmera
USER = "admin"  # Usuário da câmera
PASS = "password"  # Senha da câmera

# Inicializa a câmera ONVIF
mycam = ONVIFCamera(IP, PORT, USER, PASS)
media_service = mycam.create_media_service()
profiles = media_service.GetProfiles()
token = profiles[0].token

# Configuração do stream
stream = media_service.GetStreamUri({'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}, 'ProfileToken': token})
stream_uri = stream.Uri

# Inicializa o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Captura de vídeo
cap = cv2.VideoCapture(stream_uri)

# Variáveis para detecção de movimento
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Diferença de frames para detecção de movimento
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha retângulos ao redor dos rostos detectados
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibe o número de rostos detectados
    cv2.putText(frame1, f"Faces: {len(faces)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Desenha retângulos ao redor das áreas de movimento
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibe o frame resultante
    cv2.imshow("Frame", frame1)

    # Atualiza os frames para detecção de movimento
    frame1 = frame2
    ret, frame2 = cap.read()

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(40) == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()