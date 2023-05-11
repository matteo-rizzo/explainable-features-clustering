import cv2
import numpy as np
import yaml
from torch.utils.data import DataLoader

from classes.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm
from classes.data.MNISTDataset import MNISTDataset


def show_4():
    with open('../config/training/training_configuration.yaml', 'r') as f:
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
