import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
from threading import Thread
import time

# Función para calcular EAR (Eye Aspect Ratio)
def eye_aspect_ratio(coordinates):
    d_A = np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5]))
    d_B = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4]))
    d_C = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return (d_A + d_B) / (2 * d_C)

# Función para calcular la inclinación de la cabeza
def calculate_head_tilt(face_landmarks, width, height):
    nose = face_landmarks.landmark[1]  # Punto de la nariz
    chin = face_landmarks.landmark[152]  # Punto del mentón
    dx = (chin.x - nose.x) * width
    dy = (chin.y - nose.y) * height
    angle = np.arctan2(dy, dx)  # Ángulo de inclinación en radianes
    return np.degrees(angle)  # Convertir a grados

# Función para mostrar el mensaje de alerta y EAR
def drawing_output(frame, coordinates_left_eye, coordinates_right_eye, alert_message, ear, head_tilt_angle, fps):
    aux_image = np.zeros(frame.shape, np.uint8)
    contours1 = np.array([coordinates_left_eye])
    contours2 = np.array([coordinates_right_eye])
    cv2.fillPoly(aux_image, pts=[contours1], color=(255, 0, 0))
    cv2.fillPoly(aux_image, pts=[contours2], color=(255, 0, 0))
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)

    # Mostrar mensaje de alerta
    if alert_message:
        cv2.putText(output, alert_message, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Mostrar EAR
    cv2.rectangle(output, (0, 0), (200, 50), (255, 0, 0), -1)
    cv2.putText(output, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Mostrar ángulo de inclinación y mensaje
    cv2.putText(output, "Inclinacion: {:.1f}".format(head_tilt_angle), 
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Mostrar FPS
    cv2.putText(output, "FPS: {:.1f}".format(fps), (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output

# Función para reproducir alarma
def play_alarm():
    playsound("alarma.mp3")

# Configuración inicial
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]
EAR_THRESH = 0.2
NUM_FRAMES = 2
aux_counter = 0
alert_message = None
alarm_triggered = False

# Variables para calcular FPS
prev_time = 0

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calcular FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calcular inclinación de la cabeza
                head_tilt_angle = calculate_head_tilt(face_landmarks, width, height)
                adjusted_ear_thresh = EAR_THRESH
                

                # Dibujar la malla facial y puntos de referencia
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Resaltar nariz y mentón
                nose = face_landmarks.landmark[1]
                chin = face_landmarks.landmark[152]
                cv2.circle(frame, (int(nose.x * width), int(nose.y * height)), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(chin.x * width), int(chin.y * height)), 5, (255, 0, 0), -1)

               

                # Obtener coordenadas de los ojos
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])

                # Calcular EAR
                ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
                ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
                ear = (ear_left_eye + ear_right_eye) / 2

                # Verificar si los ojos están cerrados
                if ear < adjusted_ear_thresh:
                    aux_counter += 1
                    if aux_counter >= NUM_FRAMES * 30:  # Más de 2 segundos (asumiendo 30 fps)
                        alert_message = "¡Ojos cerrados!"
                        if not alarm_triggered:
                            alarm_triggered = True
                            Thread(target=play_alarm).start()  # Reproducir alarma en un hilo
                else:
                    aux_counter = 0
                    alert_message = None
                    alarm_triggered = False

                # Dibujar resultados en el frame
                frame = drawing_output(frame, coordinates_left_eye, coordinates_right_eye, alert_message, ear, head_tilt_angle, fps)

        cv2.imshow("Detección de Somnolencia", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Presionar 'ESC' para salir
            break

cap.release()
cv2.destroyAllWindows()
