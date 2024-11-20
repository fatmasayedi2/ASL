import csv
import copy
import cv2 as cv
import mediapipe as mp
import time
import pyttsx3
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Chargement des étiquettes de classification des points clés à partir d'un fichier CSV
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    sentence = ""  # Variable pour accumuler les lettres reconnues
    previous_letter = None  # Pour stocker la dernière lettre reconnue
    recognition_delay = 3.0  # Délai en secondes
    last_recognition_time = time.time()  # Temps de la dernière reconnaissance
    
    # Initialiser le moteur de conversion texte en parole
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Vous pouvez ajuster la vitesse de la parole ici

    voice_enabled = False  # Drapeau pour activer/désactiver la synthèse vocale

    # Fonction de rappel pour gérer les clics de souris
    def mouse_click(event, x, y, flags, param):
        nonlocal voice_enabled
        if event == cv.EVENT_LBUTTONDOWN:
            if param[0] <= x <= param[1] and param[2] <= y <= param[3]:  # Coordonnées de l'icône de micro
                voice_enabled = not voice_enabled  # Inverse l'état de la synthèse vocale
                if voice_enabled:
                    # Lire la phrase à haute voix
                    engine.say(sentence)
                    engine.runAndWait()

    cv.namedWindow('Hand Gesture Recognition')
    
    icon_x1, icon_y1 = cap_width - 60, 10  # Coordonnées en haut à droite de l'icône du micro
    icon_x2, icon_y2 = cap_width - 20, 50  # Coordonnées en bas à droite de l'icône du micro
    
    cv.setMouseCallback('Hand Gesture Recognition', mouse_click, param=(icon_x1, icon_x2, icon_y1, icon_y2))

    # Charger l'icône de microphone
    mic_icon = cv.imread('mic_icon.png')
    mic_icon = cv.resize(mic_icon, (icon_x2 - icon_x1, icon_y2 - icon_y1))

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        elif key == ord('d'):  
            if len(sentence) > 0:
                sentence = sentence[:-1]  # Supprimer la dernière lettre de la phrase
                
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Miroir horizontal de l'image pour une meilleure interaction utilisateur
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        current_time = time.time()
        elapsed_time = current_time - last_recognition_time

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calcul de la liste des points de repère (landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Classification de la main
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                recognized_letter = keypoint_classifier_labels[hand_sign_id]

                # Ajouter la lettre reconnue à la phrase seulement si le délai est écoulé
                if elapsed_time >= recognition_delay:
                    sentence += recognized_letter  # Ajouter la lettre reconnue à la phrase
                    previous_letter = recognized_letter
                    last_recognition_time = current_time  # Mettre à jour le temps de la dernière reconnaissance

                # Dessiner les landmarks et les informations de texte sur l'image de débogage
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, handedness, recognized_letter)

        # Afficher la phrase construite sur l'image de débogage
        cv.putText(debug_image, f"Sentence: {sentence}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        # Dessiner l'icône de micro
        if voice_enabled:
            debug_image[icon_y1:icon_y2, icon_x1:icon_x2] = mic_icon
        else:
            debug_image[icon_y1:icon_y2, icon_x1:icon_x2] = cv.addWeighted(mic_icon, 0.3, mic_icon, 0.7, 0)

        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
