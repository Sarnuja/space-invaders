# /// script
# dependencies = ["websockets", "opencv-python", "numpy"]
# ///

import asyncio
import cv2
import numpy as np
import websockets

ADRESSE_SERVEUR = "ws://localhost:8765"

# --- Variables Globales ---
score_actuel = "0"  # On stocke le score ici
VERT_MIN = np.array([35, 50, 50])
VERT_MAX = np.array([85, 255, 255])
TAILLE_MIN = 500 

ZONE_GAUCHE = 0.35
ZONE_DROITE = 0.65
SEUIL_TIR   = 0.35
SEUIL_ENTER = 0.75

def analyser_image(image) -> str | None:
    hauteur, largeur = image.shape[:2]
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(image_hsv, VERT_MIN, VERT_MAX)
    masque = cv2.erode(masque,  None, iterations=2)
    masque = cv2.dilate(masque, None, iterations=2)

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    plus_grand = max(contours, key=cv2.contourArea)
    aire = cv2.contourArea(plus_grand)

    if aire < TAILLE_MIN:
        return None

    moments = cv2.moments(plus_grand)
    if moments["m00"] == 0: return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    x, y = cx / largeur, cy / hauteur

    cv2.drawContours(image, [plus_grand], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)

    if y > SEUIL_ENTER: return "ENTER"
    if y < SEUIL_TIR:   return "FIRE"
    if x < ZONE_GAUCHE: return "LEFT"
    elif x > ZONE_DROITE: return "RIGHT"

    return None

def dessiner_interface(image, commande):
    """Dessine les zones, le score et la commande."""
    hauteur, largeur = image.shape[:2]
    
    # --- Dessin du Score (En haut à droite) ---
    cv2.rectangle(image, (largeur - 180, 10), (largeur - 10, 60), (0, 0, 0), -1) # Fond noir
    cv2.putText(image, f"SCORE: {score_actuel}", (largeur - 170, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # --- Commande actuelle ---
    texte_action = commande or "..."
    cv2.putText(image, f"Action: {texte_action}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2)

    # Lignes des zones (simplifié pour la lisibilité)
    x_g = int(largeur * ZONE_GAUCHE)
    x_d = int(largeur * ZONE_DROITE)
    cv2.line(image, (x_g, 0), (x_g, hauteur), (255, 255, 255), 1)
    cv2.line(image, (x_d, 0), (x_d, hauteur), (255, 255, 255), 1)

async def reception_score(connexion):
    """Tâche de fond pour recevoir le score depuis le serveur."""
    global score_actuel
    try:
        async for message in connexion:
            # On suppose que le serveur envoie juste le chiffre du score
            score_actuel = str(message)
    except Exception:
        pass

async def controleur_vision():
    webcam = cv2.VideoCapture(0)
    
    try:
        async with websockets.connect(ADRESSE_SERVEUR) as connexion:
            print("Connecté ! Lecture du score activée.")
            
            # On lance la réception des messages en tâche de fond
            asyncio.create_task(reception_score(connexion))

            while True:
                succes, image = webcam.read()
                if not succes: break

                image = cv2.flip(image, 1)
                commande = analyser_image(image)

                if commande:
                    await connexion.send(commande)

                dessiner_interface(image, commande)
                cv2.imshow("Controleur Vision", image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(controleur_vision())
