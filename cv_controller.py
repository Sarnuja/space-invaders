# /// script
# dependencies = ["websockets", "opencv-python", "numpy"]
# ///

import asyncio
import cv2
import numpy as np
import websockets

ADRESSE_SERVEUR = "ws://localhost:8765"

# ── Plage de couleur verte en espace HSV ──────────────────────
# Plage permissive pour détecter un objet vert (ex: un bouchon, un stylo)
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
    # cv2.imshow("Masque vert (Debug)", masque) # Tu peux décommenter cette ligne pour voir le masque

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
    # Reproduit le rendu de ton image : contour vert et point central vert
    cv2.drawContours(image, [plus_grand], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)

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
    """Dessine les zones de contrôle et les étiquettes comme sur l'image exemple."""
    h, l = image.shape[:2]
    x_g, x_d = int(l * ZONE_GAUCHE), int(l * ZONE_DROITE)
    y_f, y_e = int(h * SEUIL_TIR), int(h * SEUIL_ENTER)

    # --- Dessiner les lignes de délimitation ---
    # Lignes verticales (pour GAUCHE/NEUTRE/DROITE)
    cv2.line(image, (x_g, 0), (x_g, h), (255, 255, 255), 1)
    cv2.line(image, (x_d, 0), (x_d, h), (255, 255, 255), 1)
    # Lignes horizontales (pour TIR/ENTER)
    cv2.line(image, (0, y_f), (l, y_f), (255, 255, 255), 1)
    cv2.line(image, (0, y_e), (l, y_e), (255, 255, 255), 1)

    # --- Paramètres du texte ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # --- Positionner et écrire les étiquettes ---
    # TIR (Haut, Centré horizontalement)
    cv2.putText(image, "TIR", (int(l*0.48), int(h*0.3)), font, font_scale, (0, 220, 255), font_thickness)
    
    # GAUCHE (Gauche, Centré verticalement)
    cv2.putText(image, "GAUCHE", (int(l*0.1), int(h*0.55)), font, font_scale, (0, 255, 0), font_thickness)
    
    # NEUTRE (Centre, Centré verticalement)
    cv2.putText(image, "NEUTRE", (int(l*0.4), int(h*0.55)), font, font_scale, (180, 180, 180), font_thickness)
    
    # DROITE (Droite, Centré verticalement)
    cv2.putText(image, "DROITE", (int(l*0.7), int(h*0.55)), font, font_scale, (0, 255, 0), font_thickness)
    
    # ENTER (Bas, Centré horizontalement)
    cv2.putText(image, "ENTER", (int(l*0.45), int(h*0.85)), font, font_scale, (0, 255, 0), font_thickness)


async def controleur_vision():
    """Boucle principale de capture et d'envoi WebSocket."""
    # On précise la taille de la capture pour être sûr (reproduit un affichage standard)
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        async with websockets.connect(ADRESSE_SERVEUR) as connexion:
            print("✅ Connecté au serveur Space Invaders !")
            
            while True:
                succes, image = webcam.read()
                if not succes:
                    break

                # Effet miroir pour que la droite soit la droite (crucial pour le contrôle)
                image = cv2.flip(image, 1)
                
                # Étape 1 : Dessiner l'interface graphique (grille et textes)
                dessiner_zones(image)
                
                # Étape 2 : Analyser l'image pour détecter l'objet vert
                commande = analyser_image(image)

                # Étape 3 : Afficher l'action détectée en haut à gauche, comme sur ton image
                action_texte = commande or "..."
                # Texte bleu clair en haut à gauche
                cv2.putText(image, f"Commande : {action_texte}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 220, 0), 3)

                # Étape 4 : ENVOI DE LA COMMANDE au jeu (s'il y en a une)
                if commande:
                    await connexion.send(commande)
                    # print(f"Envoi : {commande}") # Décommenter pour voir le log dans la console

                # Étape 5 : Afficher le résultat final
                cv2.imshow("Contrôleur Vision - Objet Vert", image)

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
