# /// script
# dependencies = ["websockets", "opencv-python", "numpy"]
# ///

import asyncio
import cv2
import numpy as np
import websockets

ADRESSE_SERVEUR = "ws://localhost:8765"

# ── Plage de couleur verte en espace HSV ──────────────────────
# On baisse un peu la saturation minimale (50 au lieu de 100) pour mieux détecter
VERT_MIN = np.array([35, 50, 50])
VERT_MAX = np.array([85, 255, 255])

# Taille minimale de l'objet détecté (évite les petits points de bruit)
TAILLE_MIN = 500  

# ── Zones de contrôle (en pourcentage de l'image) ─────────────
ZONE_GAUCHE  = 0.35   # objet < 35% de la largeur  → LEFT
ZONE_DROITE  = 0.65   # objet > 65% de la largeur  → RIGHT
SEUIL_TIR    = 0.35   # objet < 35% de la hauteur  → FIRE
SEUIL_ENTER  = 0.75   # objet > 75% de la hauteur  → ENTER


def analyser_image(image):
    """
    Analyse l'image pour trouver l'objet vert et renvoyer une commande.
    """
    hauteur, largeur = image.shape[:2]

    # Conversion BGR → HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masque pour isoler le vert
    masque = cv2.inRange(image_hsv, VERT_MIN, VERT_MAX)

    # Nettoyage du bruit
    masque = cv2.erode(masque, None, iterations=2)
    masque = cv2.dilate(masque, None, iterations=2)

    # Affiche le masque pour debug (si c'est noir, l'objet n'est pas vu)
    cv2.imshow("Masque vert (Debug)", masque)

    # Trouve les contours
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # On prend le plus gros objet vert
    plus_grand = max(contours, key=cv2.contourArea)
    aire = cv2.contourArea(plus_grand)

    if aire < TAILLE_MIN:
        return None

    # Calcul du centre (Moments)
    M = cv2.moments(plus_grand)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Normalisation (0.0 à 1.0)
    x = cx / largeur
    y = cy / hauteur

    # Dessin sur l'image pour voir ce qui se passe
    cv2.drawContours(image, [plus_grand], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)

    # Logique de commande
    if y > SEUIL_ENTER:
        return "ENTER"
    if y < SEUIL_TIR:
        return "FIRE"
    if x < ZONE_GAUCHE:
        return "LEFT"
    if x > ZONE_DROITE:
        return "RIGHT"

    return None


def dessiner_zones(image):
    """Dessine les lignes de séparation sur la vidéo."""
    h, l = image.shape[:2]
    x_g, x_d = int(l * ZONE_GAUCHE), int(l * ZONE_DROITE)
    y_f, y_e = int(h * SEUIL_TIR), int(h * SEUIL_ENTER)

    cv2.line(image, (x_g, 0), (x_g, h), (255, 255, 255), 1)
    cv2.line(image, (x_d, 0), (x_d, h), (255, 255, 255), 1)
    cv2.line(image, (0, y_f), (l, y_f), (0, 255, 255), 1)
    cv2.line(image, (0, y_e), (l, y_e), (0, 255, 0), 1)


async def controleur_vision():
    """Boucle principale de capture et d'envoi WebSocket."""
    webcam = cv2.VideoCapture(0)
    
    # On essaye de se connecter au serveur du jeu
    try:
        async with websockets.connect(ADRESSE_SERVEUR) as connexion:
            print("✅ Connecté au serveur Space Invaders !")
            
            while True:
                succes, image = webcam.read()
                if not succes:
                    break

                # Effet miroir pour que la droite soit la droite
                image = cv2.flip(image, 1)
                
                # Visualisation
                dessiner_zones(image)
                commande = analyser_image(image)

                # ENVOI DE LA COMMANDE
                # On envoie la commande à chaque frame pour que le mouvement soit fluide
                if commande:
                    await connexion.send(commande)
                    # On affiche dans la console pour vérifier
                    print(f"Envoi : {commande}")

                # Affichage texte sur la vidéo
                cv2.putText(image, f"Action: {commande}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("CV Controller - Niveau 3", image)

                # Touche 'q' pour quitter
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                # Petit délai pour laisser respirer le CPU
                await asyncio.sleep(0.01)

    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")
        print("Assure-toi que le jeu Space Invaders est bien lancé.")
    
    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(controleur_vision())
