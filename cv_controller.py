# /// script
# dependencies = ["websockets", "opencv-python", "numpy"]
# ///

import asyncio
import cv2
import numpy as np
import websockets

ADRESSE_SERVEUR = "ws://localhost:8765"

# ── Plage de couleur verte en espace HSV ──────────────────────
# On utilise HSV car c'est moins sensible aux changements de lumière que le RGB
VERT_MIN = np.array([35, 50, 50])
VERT_MAX = np.array([85, 255, 255])

# Taille minimale de l'objet détecté (évite les faux positifs / bruits)
TAILLE_MIN = 500  # en pixels²

# ── Zones de contrôle (en pourcentage de l'image) ─────────────
ZONE_GAUCHE  = 0.35   # objet < 35% de la largeur  → LEFT
ZONE_DROITE  = 0.65   # objet > 65% de la largeur  → RIGHT
SEUIL_TIR    = 0.35   # objet < 35% de la hauteur  → FIRE
SEUIL_ENTER  = 0.75   # objet > 75% de la hauteur  → ENTER


def analyser_image(image) -> str | None:
    """
    Analyser une image et retourner une commande ou None.

    Logique :
    - Objet vert en bas    (> 75%) → ENTER (démarrer la partie)
    - Objet vert en haut   (< 35%) → FIRE  (tirer)
    - Objet vert à gauche  (< 35%) → LEFT
    - Objet vert à droite  (> 65%) → RIGHT
    - Zone neutre / rien détecté   → None
    """
    hauteur, largeur = image.shape[:2]

    # Conversion BGR → HSV pour détecter la couleur verte
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masque : garde uniquement les pixels verts (crée l'image noir et blanc)
    masque = cv2.inRange(image_hsv, VERT_MIN, VERT_MAX)

    # Réduit le bruit (Erode enlève les petits points, Dilate regonfle l'objet)
    masque = cv2.erode(masque,  None, iterations=2)
    masque = cv2.dilate(masque, None, iterations=2)

    # --- AFFICHAGE DU MASQUE (L'écran noir de debug) ---
    cv2.imshow("Masque vert", masque)

    # Trouve les contours de l'objet
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Prend le plus grand contour détecté (ton objet principal)
    plus_grand = max(contours, key=cv2.contourArea)
    aire = cv2.contourArea(plus_grand)

    # Ignore si l'objet est trop petit (poussière ou reflet)
    if aire < TAILLE_MIN:
        return None

    # Calcule le centroïde (le point central de l'objet)
    moments = cv2.moments(plus_grand)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # Normalise les coordonnées entre 0.0 et 1.0 pour faciliter les calculs
    x = cx / largeur
    y = cy / hauteur

    # Dessine le contour et le point central (centroïde) sur l'image couleur
    cv2.drawContours(image, [plus_grand], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)

    
    # ── LOGIQUE DES COMMANDES (Priorités)  ──────────────────────
    # Priorité 1: objet en bas → ENTER
    if y > SEUIL_ENTER:
        return "ENTER"

    # Priorité 2 : objet en haut → FIRE
    if y < SEUIL_TIR:
        return "FIRE"

    # Priorité 3 : position horizontale → LEFT ou RIGHT
    if x < ZONE_GAUCHE:
        return "LEFT"
    elif x > ZONE_DROITE:
        return "RIGHT"

    return None


def dessiner_zones(image):
    """Dessine les zones de contrôle sur l'image pour visualisation."""
    hauteur, largeur = image.shape[:2]
    x_gauche = int(largeur * ZONE_GAUCHE)
    x_droite = int(largeur * ZONE_DROITE)
    y_tir    = int(hauteur * SEUIL_TIR)
    y_enter  = int(hauteur * SEUIL_ENTER)

    # Lignes de séparation (Blanches)
    cv2.line(image, (x_gauche, 0), (x_gauche, hauteur), (255, 255, 255), 1)
    cv2.line(image, (x_droite, 0), (x_droite, hauteur), (255, 255, 255), 1)
    cv2.line(image, (0, y_tir),    (largeur, y_tir),    (255, 255, 255), 1)
    cv2.line(image, (0, y_enter),  (largeur, y_enter),  (255, 255, 255), 1)

    # Étiquettes de l'interface (Positionnement et Couleurs)
    police = cv2.FONT_HERSHEY_SIMPLEX
    # TIR en haut
    cv2.putText(image, "TIR",    (largeur // 2 - 30,  y_tir - 10), police, 0.8, (0, 220, 255), 2)
    # GAUCHE / NEUTRE / DROITE au milieu
    cv2.putText(image, "GAUCHE", (x_gauche // 2 - 50,  hauteur // 2), police, 0.7, (0, 255, 0), 2)
    cv2.putText(image, "NEUTRE", (x_gauche + 20,       hauteur // 2), police, 0.6, (180, 180, 180), 1)
    cv2.putText(image, "DROITE", (x_droite + 20,       hauteur // 2), police, 0.7, (0, 255, 0), 2)
    # ENTER en bas
    cv2.putText(image, "ENTER",  (largeur // 2 - 40,  y_enter + 40), police, 0.8, (0, 255, 100), 2)

    return image


async def controleur_vision():
    """Boucle principale : capture la webcam et envoie les commandes au jeu."""
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        async with websockets.connect(ADRESSE_SERVEUR) as connexion:
            print("Connecté au jeu Space Invaders !")
            print("Utilisez votre objet vert pour piloter.")

            while True:
                succes, image = webcam.read()
                if not succes:
                    break

                # Miroir horizontal : pour que ta main droite soit à la droite de l'écran
                image = cv2.flip(image, 1)

                # Dessine les zones et récupère la commande
                dessiner_zones(image)
                commande = analyser_image(image)

                # ── ENVOI DE LA COMMANDE  ──────────────────────
                # On envoie la commande dès qu'elle existe pour un mouvement fluide.
                if commande:
                    await connexion.send(commande)

                # Affiche l'action en cours en haut à gauche (Cyan/Bleu)
                texte_action = commande or "..."
                cv2.putText(image, f"Commande : {texte_action}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 0), 3)

                # Affiche la fenêtre principale
                cv2.imshow("Controleur Vision - Objet Vert", image)
                
                # Quitter avec 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Petit temps de pause pour ne pas saturer le processeur
                await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Erreur de connexion : {e}")
        print("Vérifiez que le jeu Space Invaders est lancé sur le port 8765.")

    finally:
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(controleur_vision())
