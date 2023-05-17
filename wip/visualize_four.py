import cv2
import numpy as np
import yaml
from torch.utils.data import DataLoader

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.data.MNISTDataset import MNISTDataset


def show_4():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    key_points_extractor_1 = FeatureExtractingAlgorithm(nfeatures=200,
                                                        # (default = 0 = all) Small images, few features
                                                        nOctaveLayers=3,
                                                        # (default = 3) Default should be ok
                                                        contrastThreshold=0.04,
                                                        # (default = 0.04) Lower = Include kps with lower contrast
                                                        edgeThreshold=10,
                                                        # (default = 10) Higher = Include KPS with lower edge response
                                                        sigma=1.2)  # (default = 1.2) capture finer details in imgs
    key_points_extractor_2 = FeatureExtractingAlgorithm(nfeatures=200,
                                                        # (default = 0 = all) Small images, few features
                                                        nOctaveLayers=3,
                                                        # (default = 3) Default should be ok
                                                        contrastThreshold=0.04,
                                                        # (default = 0.04) Lower = Include kps with lower contrast
                                                        edgeThreshold=10,
                                                        # (default = 10) Higher = Include KPS with lower edge response
                                                        sigma=0.75)  # (default = 1.2) capture finer details in imgs

    key_points_extractor_3 = FeatureExtractingAlgorithm(nfeatures=200,
                                                        # (default = 0 = all) Small images, few features
                                                        nOctaveLayers=3,
                                                        # (default = 3) Default should be ok
                                                        contrastThreshold=0.04,
                                                        # (default = 0.04) Lower = Include kps with lower contrast
                                                        edgeThreshold=20,
                                                        # (default = 10) Higher = Include KPS with lower edge response
                                                        sigma=1.2)  # (default = 1.2) capture finer details in imgs

    key_points_extractor_4 = FeatureExtractingAlgorithm(nfeatures=200,
                                                        # (default = 0 = all) Small images, few features
                                                        nOctaveLayers=3,
                                                        # (default = 3) Default should be ok
                                                        contrastThreshold=0.04,
                                                        # (default = 0.04) Lower = Include kps with lower contrast
                                                        edgeThreshold=20,
                                                        # (default = 10) Higher = Include KPS with lower edge response
                                                        sigma=0.75)  # (default = 1.2) capture finer details in imgs
    # for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for img in imgs:
            img = (img.numpy().squeeze() * 255).astype(np.uint8)

            kp1, _ = key_points_extractor_1.run(img)
            kp2, _ = key_points_extractor_2.run(img)
            kp3, _ = key_points_extractor_3.run(img)
            kp4, _ = key_points_extractor_4.run(img)

            keypoints = [kp1, kp2, kp3, kp4]

            # Create a blank canvas to display the images
            canvas = np.zeros((28 * 2, 28 * 2, 3), dtype=np.uint8)

            # Loop through each digit and its keypoints
            for i in range(4):
                # Convert the image to uint8 and resize it
                # img = (digits[i].numpy().squeeze() * 255).astype(np.uint8)
                # img = cv2.resize(img, (28, 28))

                # Draw keypoints on the image
                img_kp = cv2.drawKeypoints(img, keypoints[i], None,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Compute the coordinates for placing the image on the canvas
                x = (i % 2) * 28
                y = (i // 2) * 28

                # Place the image with keypoints on the canvas
                canvas[y:y + 28, x:x + 28] = img_kp

            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            scale = 4  # Adjust this to change the size of the canvas
            resized_canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale))
            cv2.imshow('Canvas', resized_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    show_4()

# 1 step - corner detection. Poi ci sono degli step che sopprimono feature troppo vicine alle altre
# Anche globalmente, le feature stesse vengono clusterizzate e tengono feature distinte
# Ne trova tante o poche?

# Stiamo cercando di mettere assieme explainability in generale. Invece di lavorare su dato raw, dato di secondo livello
# Testo: magari fare lemmatization, named entity extraction...
# Idea: non avere stack interamente black box, ma avere una pre-elaborazione del dato che prende tutta la conoscenza di
# dominio certa e la butta nel dato iniziale per arricchirla.
# E.g. chicchi di caffè, se so che sono a punta = better -> estrattore di punte
# Se so che nei testi ogni volta che c'è una parola chiave, estraggo quella

# Se voglio che sia explainable by design, è folle lavorare sull'immagine stessa!
# Non possiamo certo clusterizzare i pixel
# Processiamo cose che hanno già una semantica. Abbiamo pensato con le features come cosa specifica, ma in generale
# l'idea di usare SIFT che è low level (could be good or bad) perché magari i pixel non sono giusti!

# L'informazione semantica dell'expert (che è SIFT) abbiamo assunto essere clusterizzabile e riassumibile in un
# vocabolario. Un certo numero di modi in cui appare. Nel caso delle feature visuali abbiamo detto che la clusterizzazione
# dei descrittori potrebbe essere un buon vocabolario. Questa è UNA via.

# Se non funziona, dobbiamo cambiare rappresentatori di basso livello. E.g. banane = colore best descriptor
# Cosa può fallire: descrittori magari non vanno bene.
# Magari il vocabolario non è costruibile

# Vocabolario serve!
# Non è detto che il vocabolario non possa ESSERE APPRESO!
# Altra cosa: cercare dalle classificazioni di estrarre il vocabolario e poi usarlo per la semantica
# Invece di usare PRIMA feature detection. Poi SUL CLASSIFICATIOn si può creare una rappresentazione dei pixel più significativi!
# E.g. costruire descrittore sui pixel delle heatmap; magari un intorno, allinearli

# Sempre come IDEA GENERALE
# Il vocabolario SERVE. Serve per passare dal locale al globale. Altrimenti non e' possibile portare la spiegazione
# a globale

# PROVIAMO DATASET DIVERSI. POCO RICCO!
# Dataset con scene pubbliche, comportamenti, cartelli stradali, ambienti artificiali... molto più ricco
# Riflettiamo anche sull'imparare i vocabolari DOPO la classificazione
# Ovvero, imparare un modo in cui le semantiche appaion o poi un algoritmo per produrle
# Dato un testo, dammi gli elementi del vocabolario presenti.

# E.g., SHAP on MNIST, prendi top 10. Clusterizzi i rettangoli 3x3 che A POSTERIORI diventano il vocabolario.


# Baseline experiment. Classification + SHAP + prendi 10 pixel piu' importanti (magari con criterio di non maximum noise supporesion)
# Nuclearizzi l'intorno (magari tecniche piu' smart di NMS). Trova i picchi e tieni solo il picco piu' completo
# Trovare 10 picchi piu' sigfnificativi (con threshold, magari son 5)

# Intensita in 3x3 o 5x5, normalizza a 1 sui numeri stupid, o min max normalization
# Non è rotational invariant but it's ok, 6 and 9 non lo sono. Scale invariant no ma chissene
# linearizzazione del 3x3 (9) e' sufficiente. Esempio dell'altro giro.

# Estrarre feature dall'attenzione e usarle come vocabolario!

# Nuova rete che usa le feature estratte dal vocabolario!!
# Potrei dare in pasto un immagine con 200 layer, ogni layer e' la convoluzione dell'immagine originale con il centroide del cluster
# 200 layer sono la conoscenza di dominio (1o corner)
# 200 se son 200 le parole di vocab

# Con le SIFT si puo' usare come kernel. Etrai per ogni pixel, fai prodotto coseno
# fra SIFT di quell'elemento del vocabolario e quella estratta in quel punto

# Pero' serve qualcuno (nel post hoc) che guarda cosa sono ste robe nel vocabolario)


# Non superare i 9x9. Numeri delle ottave. OCCHO! Riduci!! metti dei boundaries

# LASCIA STARE SIFT QUI. FAI CORNER EXTRACTION. CONTROLLA CHE I CORNER TI PIACCIANO.
# CORNER EXTRACTION MULTISCALA (ANCHE IN MEZZO ALL'OGGETTO)
# bei punti con bella variazione di intensità
# E CHE SIANO COERENTI (e.g. fra i sette)
# PUNTI STABILI!
# FEATURE: INTENSITA' IN BOX 3x3 e 5x5 (7x7) -> vettori linearizzati

# In backward, max attention intensity SHAP

# Clustering gerarchico: distanza fra vettori, radice quadrata di distanze e coseno

# E visualizza su tutto il dataset. Quando hai un vocabolario.
# Poi si puo' fare andare indietro con SHAP e calcola stesse feature, ALTRO VOCABOLARIO

# Gli do corner in pasto, SHAP mi dira' i corner piu' importanti.
# Usiamo gli score in backward; usando il secondo vocabolario

# La classificazione dei numeri la farei con i 200 layer della convoluzione di ogni parola del vocabolario