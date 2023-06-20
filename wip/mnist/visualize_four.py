import cv2
import numpy as np
import yaml
from torch.utils.data import DataLoader

from classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
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


def show_4_harris():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    # for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for img in imgs:
            # Create a blank canvas to display the images
            canvas = np.zeros((28 * 2, 28 * 2, 3), dtype=np.uint8)
            # ---------------------------------------------------
            img = img.numpy().squeeze()
            # ---------------------------------------------------
            corner_imgs = []
            block_size, ksize, k = 2, 3, 0.04
            corner_imgs.append(cv2.cornerHarris(img, block_size, ksize, k))
            block_size, ksize, k = 3, 3, 0.04
            corner_imgs.append(cv2.cornerHarris(img, block_size, ksize, k))
            block_size, ksize, k = 4, 3, 0.04
            corner_imgs.append(cv2.cornerHarris(img, block_size, ksize, k))
            block_size, ksize, k = 5, 3, 0.04
            corner_imgs.append(cv2.cornerHarris(img, block_size, ksize, k))

            # Loop through each digit and its keypoints
            for i, corner_img in enumerate(corner_imgs):
                threshold = 0.01  # Adjust this value to change the threshold
                # result is dilated for marking the corners, not important
                corner_img = cv2.dilate(corner_img, None)
                corner_img_thresholded = corner_img > threshold * corner_img.max()
                # ---------------------------------------------------
                color_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                color_img[corner_img_thresholded] = [0, 0, 255]  # Mark the corners in red (assuming a color image)
                # ---------------------------------------------------
                corner_img = np.uint8(corner_img)
                # find centroids
                ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_img)
                # define the criteria to stop and refine the corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                corners = cv2.cornerSubPix(img, np.float32(centroids), (5, 5), (-1, -1), criteria)
                res = np.hstack((centroids, corners))
                res = np.int0(res)
                color_img[res[:, 1], res[:, 0]] = [0, 0, 255]
                color_img[res[:, 3], res[:, 2]] = [0, 255, 0]
                # Compute the coordinates for placing the image on the canvas
                x = (i % 2) * 28
                y = (i // 2) * 28

                # Place the image with keypoints on the canvas
                canvas[y:y + 28, x:x + 28] = color_img

            # ---------------------------------------------------
            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            scale = 4  # Adjust this to change the size of the canvas
            resized_canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale))
            cv2.imshow('Canvas', resized_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def show_4_shitomasi():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    # for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for img in imgs:
            # Create a blank canvas to display the images
            canvas = np.zeros((28 * 2, 28 * 2, 3), dtype=np.uint8)
            # ---------------------------------------------------
            img = img.numpy().squeeze()
            # ---------------------------------------------------
            corner_imgs = []

            # Set the parameters for Shi-Tomasi Corner Detector
            max_corners = None  # Maximum number of corners to detect
            quality_level = 0.01  # Quality level threshold
            min_distance = 1  # Minimum distance between detected corners
            corner_imgs.append(np.int0(cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)))

            max_corners = None  # Maximum number of corners to detect
            quality_level = 0.01  # Quality level threshold
            min_distance = 2  # Minimum distance between detected corners
            corner_imgs.append(np.int0(cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)))

            max_corners = None  # Maximum number of corners to detect
            quality_level = 0.1  # Quality level threshold
            min_distance = 3  # Minimum distance between detected corners
            corner_imgs.append(np.int0(cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)))

            max_corners = None  # Maximum number of corners to detect
            quality_level = 0.01  # Quality level threshold
            min_distance = 4  # Minimum distance between detected corners
            corner_imgs.append(np.int0(cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)))
            # Loop through each digit and its keypoints
            for i, corner_img in enumerate(corner_imgs):
                # ---------------------------------------------------
                draw_in_square(canvas, corner_img, i, img)
            # ---------------------------------------------------
            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            scale = 4  # Adjust this to change the size of the canvas
            resized_canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale))
            cv2.imshow('Canvas', resized_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def show_multiscale_shitomasi():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    # for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for image in imgs:
            image = image.numpy().squeeze()
            max_corners = None  # Maximum number of corners to detect
            quality_level = 0.01  # Quality level threshold
            min_distance = 2  # Minimum distance between detected corners
            block_size = 3  # Size of the neighborhood considered for corner detection
            use_harris_detector = False  # Whether to use the Harris corner detector or not
            k = 0.04  # Free parameter for the Harris detector
            scale_factor = 1.2  # Scale factor between each level of the image pyramid (scale down)
            num_levels = 3  # Number of levels in the image pyramid

            # Create an image pyramid
            pyramid = [image]  # Default size
            for i in range(1, num_levels):
                scaled = cv2.resize(pyramid[i - 1], (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
                pyramid.append(scaled)

            # Perform corner detection at each level of the pyramid
            corners = []
            for level, img in enumerate(pyramid):
                # Apply corner detection
                level_corners = cv2.goodFeaturesToTrack(image=img, maxCorners=max_corners,
                                                        qualityLevel=quality_level,
                                                        minDistance=min_distance,
                                                        blockSize=block_size,
                                                        useHarrisDetector=use_harris_detector, k=k)
                if level_corners is not None:
                    level_corners = level_corners.reshape(-1, 2)  # Reshape corner coordinates
                    level_corners *= (scale_factor ** level)  # Scale the corners back to the original image size
                    corners.extend(level_corners)

            # Draw the detected corners on the image
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for corner in corners:
                x, y = corner.astype(int)
                cv2.circle(color_img, (x, y), 1,  (0, 0, 255), -1)

            cv2.namedWindow("Multiscale Corner Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Multiscale Corner Detection", 280, 280)  # Set the desired window size

            # Display the result
            cv2.imshow("Multiscale Corner Detection", color_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def show_4_multiscale_shitomasi():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_loader = DataLoader(MNISTDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=False,
                              num_workers=config["workers"])
    # for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for image in imgs:
            # Create a blank canvas to display the images
            canvas = np.zeros((28 * 2, 28 * 2, 3), dtype=np.uint8)
            # ---------------------------------------------------
            image = image.numpy().squeeze()
            # ---------------------------------------------------
            # corner_imgs = []
            max_corners = None  # Maximum number of corners to detect
            quality_level = 0.01  # Quality level threshold
            min_distance = 2  # Minimum distance between detected corners
            block_size = 3  # Size of the neighborhood considered for corner detection
            use_harris_detector = False  # Whether to use the Harris corner detector or not
            k = 0.04  # Free parameter for the Harris detector
            scale_factor = 1.2  # Scale factor between each level of the image pyramid (scale down)
            num_levels = 4  # Number of levels in the image pyramid

            # Create an image pyramid
            pyramid = [image]  # Default size
            for i in range(1, num_levels):
                scaled = cv2.resize(pyramid[i - 1], (0, 0), fx=1 / scale_factor, fy=1 / scale_factor)
                pyramid.append(scaled)

            # Perform corner detection at each level of the pyramid
            for level, img in enumerate(pyramid):
                # Apply corner detection
                level_corners = cv2.goodFeaturesToTrack(image=img, maxCorners=max_corners,
                                                        qualityLevel=quality_level,
                                                        minDistance=min_distance,
                                                        blockSize=block_size,
                                                        useHarrisDetector=use_harris_detector, k=k)
                if level_corners is not None:
                    level_corners = level_corners.reshape(-1, 2)  # Reshape corner coordinates
                    # level_corners *= (scale_factor ** level)  # Scale the corners back to the original image size
                    draw_in_square(canvas, level_corners, level, img)

            # Draw the detected corners on the image
            # color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # for corner in corners:
            #     x, y = corner.astype(int)
            #     cv2.circle(color_img, (x, y), 1,  (0, 0, 255), -1)
            #
            #
            # for i, corner_img in enumerate(corner_imgs):
            #     # ---------------------------------------------------
            #     draw_in_square(canvas, corner_img, i, img)

            # ---------------------------------------------------
            cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
            scale = 4  # Adjust this to change the size of the canvas
            resized_canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale))
            cv2.imshow('Canvas', resized_canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def draw_in_square(canvas, corner_img, i, img):
    color_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Compute the coordinates for placing the image on the canvas
    x = (i % 2) * 28
    y = (i // 2) * 28
    # Place the image with keypoints on the canvas
    padded_image = np.pad(color_img, ((0, 28-color_img.shape[0]), (0, 28-color_img.shape[1]), (0, 0)), mode='constant')
    canvas[y:y + 28, x:x + 28] = padded_image
    for corner in corner_img:
        _x, _y = corner.ravel()
        # x = (i % 2) * 28
        # y = (i // 2) * 28
        cv2.circle(canvas[y:y + 28, x:x + 28], (int(_x), int(_y)), 1, (0, 0, 255), -1)


if __name__ == "__main__":
    show_4()
    # show_4_shitomasi()
    # show_multiscale_shitomasi()
    # show_4_multiscale_shitomasi()
    # show_4_harris()

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
