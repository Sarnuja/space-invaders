# /// script
# dependencies = ["websockets", "opencv-python"]
# ///

import asyncio
import cv2
import numpy as np
import websockets

ADRESSE_SERVEUR = "ws://localhost:8765"

# ── Plage de couleur verte en espace HSV ──────────────────────
VERT_MIN = np.array([35, 100, 100])
VERT_MAX = np.array([85, 255, 255])

# Taille minimale de l'objet détecté (évite les faux positifs)
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

   Retourne :
       "LEFT", "RIGHT", "FIRE", "ENTER" ou None
   """
   hauteur, largeur = image.shape[:2]

   # Conversion BGR → HSV pour détecter la couleur verte
   image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   # Masque : garde uniquement les pixels verts
   masque = cv2.inRange(image_hsv, VERT_MIN, VERT_MAX)

   # Réduit le bruit
   masque = cv2.erode(masque,  None, iterations=2)
   masque = cv2.dilate(masque, None, iterations=2)

   # Affiche le masque pour calibrer la détection
   cv2.imshow("Masque vert", masque)

   # Trouve les contours de l'objet
   contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   if not contours:
       return None

   # Prend le plus grand contour détecté
   plus_grand = max(contours, key=cv2.contourArea)
   aire = cv2.contourArea(plus_grand)

   # Ignore si trop petit (faux positif)
   if aire < TAILLE_MIN:
       return None

   # Calcule le centroïde (centre de l'objet)
   moments = cv2.moments(plus_grand)
   if moments["m00"] == 0:
       return None

   cx = int(moments["m10"] / moments["m00"])
   cy = int(moments["m01"] / moments["m00"])

   # Normalise entre 0.0 et 1.0
   x = cx / largeur
   y = cy / hauteur

   # Dessine le contour et le centroïde sur l'image
   cv2.drawContours(image, [plus_grand], -1, (0, 255, 0), 2)
   cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)

   # Priorité 1 : objet en bas → ENTER
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

   # Lignes de séparation
   cv2.line(image, (x_gauche, y_tir),  (x_gauche, y_enter), (255, 255, 255), 1)
   cv2.line(image, (x_droite, y_tir),  (x_droite, y_enter), (255, 255, 255), 1)
   cv2.line(image, (0, y_tir),         (largeur, y_tir),     (0, 200, 255),  1)
   cv2.line(image, (0, y_enter),       (largeur, y_enter),   (0, 255, 100),  1)

   # Étiquettes
   police = cv2.FONT_HERSHEY_SIMPLEX
   cv2.putText(image, "TIR",    (largeur // 2 - 20,   y_tir // 2 + 8),                  police, 0.8, (0, 220, 255),  2)
   cv2.putText(image, "GAUCHE", (x_gauche // 2 - 40,  y_tir + (y_enter - y_tir) // 2),  police, 0.7, (0, 255, 0),    2)
   cv2.putText(image, "NEUTRE", (x_gauche + 20,       y_tir + (y_enter - y_tir) // 2),  police, 0.6, (180, 180, 180),1)
   cv2.putText(image, "DROITE", (x_droite + 20,       y_tir + (y_enter - y_tir) // 2),  police, 0.7, (0, 255, 0),    2)
   cv2.putText(image, "ENTER",  (largeur // 2 - 30,   y_enter + (hauteur - y_enter) // 2), police, 0.8, (0, 255, 100), 2)

   return image


async def controleur_vision():
   """Boucle principale : capture la webcam et envoie les commandes au jeu."""
   webcam = cv2.VideoCapture(0)
   webcam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
   webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

   async with websockets.connect(ADRESSE_SERVEUR) as connexion:
       print("Connecté au jeu Space Invaders !")
       print("Contrôles avec objet vert :")
       print("  - Objet en bas    (> 75%)  → ENTER (démarrer)")
       print("  - Objet en haut   (< 35%)  → FIRE  (tirer)")
       print("  - Objet à gauche  (< 35%)  → LEFT")
       print("  - Objet à droite  (> 65%)  → RIGHT")
       print("  - Appuyer sur 'q' pour quitter\n")

       derniere_commande = None

       while True:
           succes, image = webcam.read()
           if not succes:
               break

           # Miroir horizontal
           image = cv2.flip(image, 1)

           # Dessine les zones
           dessiner_zones(image)

           # Analyse l'image
           commande = analyser_image(image)

           # Envoie la commande si elle change
           if commande and commande != derniere_commande:
               await connexion.send(commande)
               print(f"Commande envoyée : {commande}")
               derniere_commande = commande
           elif not commande:
               derniere_commande = None

           # Affiche la commande
           couleurs = {
               "LEFT":  (255, 180, 0),
               "RIGHT": (255, 180, 0),
               "FIRE":  (0, 220, 255),
               "ENTER": (0, 255, 100),
           }
           texte   = commande or "..."
           couleur = couleurs.get(texte, (180, 180, 180))
           cv2.putText(image, f"Commande : {texte}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, couleur, 2)

           cv2.imshow("Contrôleur Vision - Objet Vert", image)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break

   webcam.release()
   cv2.destroyAllWindows()


if __name__ == "__main__":
   asyncio.run(controleur_vision())
