from flask import Flask, request, jsonify
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import logging
import requests
from requests.exceptions import RequestException
import face_recognition
import os
import pickle
import util

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
ESP32_IP = "192.168.1.17"  # Adresse IP de l'ESP32
ESP32_PORT = 80
ESP32_ENDPOINT = f"http://{ESP32_IP}:{ESP32_PORT}/face_recognition"

# Configuration de la base de données
DB_DIR = './db'
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# Configuration MQTT
broker = "test.mosquitto.org"
port = 1883
mqtt_client = mqtt.Client(client_id="FlaskServer", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)

try:
    mqtt_client.connect(broker, port)
    mqtt_client.loop_start()
    logger.info("Connexion MQTT établie avec succès")
except Exception as e:
    logger.error(f"Erreur de connexion MQTT: {e}")
    mqtt_client = None


def send_command_to_esp32(command, recognized_name):
    try:
        # Format cohérent avec l'ESP32
        data = {
            "command": command,
            "name": recognized_name,
            "status": "success" if command == "OPEN" else "failure"
        }

        headers = {
            'Content-Type': 'application/json',
            'Accept-Charset': 'UTF-8'
        }

        logger.info(f"Envoi de la commande à l'ESP32: {data}")
        response = requests.post(ESP32_ENDPOINT, json=data, headers=headers, timeout=5)

        logger.info(f"Réponse de l'ESP32: Status={response.status_code}, Content={response.text}")
        response.raise_for_status()
        logger.info(f"Commande {command} envoyée avec succès à l'ESP32")
        return True
    except RequestException as e:
        logger.error(f"Erreur lors de l'envoi de la commande à l'ESP32: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"Détails de l'erreur: {e.response.text}")
        return False


def recognize_face(image):
    try:
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)

        if not face_locations:
            logger.info("Aucun visage détecté dans l'image")
            return "no_persons_found"

        name = util.recognize(image, DB_DIR)

        if name in ['unknown_person', 'no_persons_found']:
            logger.info(f"Visage non reconnu: {name}")
            return name

        logger.info(f"Visage reconnu: {name}")
        return name

    except Exception as e:
        logger.error(f"Erreur lors de la reconnaissance faciale: {e}")
        return "error"


@app.route('/face-recognition', methods=['POST'])
def face_recognition_api():
    try:
        if 'image' not in request.files:
            return jsonify({"status": "error", "message": "Aucune image fournie"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Aucun fichier sélectionné"}), 400

        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"status": "error", "message": "Image vide"}), 400

        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Impossible de décoder l'image"}), 400

        result = recognize_face(img)

        if result not in ['unknown_person', 'no_persons_found', 'error']:
            if mqtt_client:
                try:
                    mqtt_client.publish("esp32/door", "OPEN")
                    logger.info("Commande OPEN envoyée via MQTT")
                except Exception as e:
                    logger.error(f"Erreur lors de la publication MQTT: {e}")

            if not send_command_to_esp32("OPEN", result):
                logger.warning("La commande MQTT a été envoyée mais la commande HTTP à l'ESP32 a échoué")

        return jsonify({
            "status": "success",
            "match": result,
            "message": "Visage reconnu" if result not in ['unknown_person', 'no_persons_found', 'error'] else "Visage non reconnu"
        })

    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")
        return jsonify({"status": "error", "message": "Erreur interne du serveur"}), 500


if __name__ == '__main__':
    try:
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur: {e}")