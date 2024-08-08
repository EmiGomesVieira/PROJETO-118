import cv2

# Crie nosso classificador de corpos
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Inicie a captura de vídeo para o arquivo de vídeo
cap = cv2.VideoCapture('walking.avi')

# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    
    # Leia o primeiro quadro
    ret, frame = cap.read()
    
    if not ret:
        break

    # Converta cada quadro em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Passe o quadro para nosso classificador de corpos
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    # Extraia as caixas delimitadoras para quaisquer corpos identificados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Exibir o quadro com as detecções
    cv2.imshow('Walking', frame)

    # Pressione a barra de espaço (32) para sair do loop
    if cv2.waitKey(1) == 32:  # 32 é a barra de espaço
        break

# Liberar a captura de vídeo e destruir todas as janelas
cap.release()
cv2.destroyAllWindows()

