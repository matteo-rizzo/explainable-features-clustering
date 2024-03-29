import time

import cv2
import numpy as np
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from src.classes.data import *
from src.classes.feature_extraction.FeatureExtractingAlgorithm import FeatureExtractingAlgorithm


def show_4():
    with open('config/training/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # train_loader = DataLoader(DescribableTexturesDataset(split="train", gray=True),
    train_loader = DataLoader(OxfordIIITPetDataset(train=True),
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config["workers"])
    # DEFAULT
    key_points_extractor_1 = FeatureExtractingAlgorithm(algorithm="SIFT")

    key_points_extractor_2 = FeatureExtractingAlgorithm(algorithm="SIFT",
                                                        contrastThreshold=0.02,
                                                        edgeThreshold=30,)
                                                        # nOctaveLayers=3,
                                                        # # (default = 3) Default should be ok
                                                        # contrastThreshold=0.01,
                                                        # # (default = 0.04) Lower = Include kps with lower contrast
                                                        # edgeThreshold=10,
                                                        # # (default = 10) Higher = Include KPS with lower edge response
                                                        # sigma=1.6)  # (default = 1.2) capture finer details in imgs
    key_points_extractor_3 = FeatureExtractingAlgorithm(algorithm="SIFT",
                                                        contrastThreshold=0.02,
                                                        edgeThreshold=50,)
                                                        # nOctaveLayers=3,
                                                        # # (default = 3) Default should be ok
                                                        # contrastThreshold=0.01,
                                                        # # (default = 0.04) Lower = Include kps with lower contrast
                                                        # edgeThreshold=20,
                                                        # # (default = 10) Higher = Include KPS with lower edge response
                                                        # sigma=1.6)  # (default = 1.2) capture finer details in imgs
    with open('config/feature_extraction/SIFT_config.yaml', 'r') as f:
        sift_config: dict = yaml.safe_load(f)
    key_points_extractor_4 = FeatureExtractingAlgorithm(algorithm="SIFT", **sift_config) # logger=logger,
    # key_points_extractor_4 = FeatureExtractingAlgorithm(algorithm="SIFT",
    #                                                     contrastThreshold=0.01,
    #                                                     edgeThreshold=50,)
                                                        # nOctaveLayers=3,
                                                        # # (default = 3) Default should be ok
                                                        # contrastThreshold=0.01,
                                                        # # (default = 0.04) Lower = Include kps with lower contrast
                                                        # edgeThreshold=30,
                                                        # # (default = 10) Higher = Include KPS with lower edge response
                                                        # sigma=1.6)  # (default = 1.2) capture finer details in imgs

    # for imgs, _ in train_loader:
    for imgs, _ in train_loader:
        for img in imgs:
            # img = (img.numpy().squeeze() * 255).astype(np.uint8).transpose((1, 2, 0))
            img = (img.numpy().squeeze() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            kp1, _ = key_points_extractor_1.run(img)
            kp2, _ = key_points_extractor_2.run(img)
            kp3, _ = key_points_extractor_3.run(img)
            kp4, _ = key_points_extractor_4.run(img)


            if len(kp1) < 50 or len(kp2) < 50 or len(kp3) < 50 or len(kp4) < 50:
                print(len(kp1), len(kp2), len(kp3), len(kp4))
                keypoints = [kp1, kp2, kp3, kp4]

                # Create a blank canvas to display the images
                canvas = np.zeros((224 * 2, 224 * 2, 3), dtype=np.uint8)

                # Loop through each digit and its keypoints
                for i in range(4):
                    # Draw keypoints on the image
                    img_kp = cv2.drawKeypoints(img, keypoints[i], None,
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    # Compute the coordinates for placing the image on the canvas
                    x = (i % 2) * 224
                    y = (i // 2) * 224

                    # Place the image with keypoints on the canvas
                    canvas[y:y + 224, x:x + 224] = img_kp

                    # Display the canvas using plt
                plt.imshow(canvas)
                plt.show()
                # time.sleep(1.0)
                # cv2.namedWindow('Canvas', cv2.WINDOW_NORMAL)
                # scale = 4  # Adjust this to change the size of the canvas
                # resized_canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale))
                # cv2.imshow('Canvas', resized_canvas)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


# def count():
#     with open('config/training/training_configuration.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#
#     train_loader = DataLoader(OxfordIIITPetDataset(train=True),
#                               batch_size=config["batch_size"],
#                               shuffle=True,
#                               num_workers=config["workers"])
#     key_points_extractor_1 = FeatureExtractingAlgorithm(algorithm="SIFT",
#                                                         nfeatures=0,
#                                                         # (default = 0 = all) Small images, few features
#                                                         nOctaveLayers=3,
#                                                         # (default = 3) Default should be ok
#                                                         contrastThreshold=0.04,
#                                                         # (default = 0.04) Lower = Include kps with lower contrast
#                                                         edgeThreshold=10,
#                                                         # (default = 10) Higher = Include KPS with lower edge response
#                                                         sigma=1.2)  # (default = 1.2) capture finer details in imgs
#     # for imgs, _ in train_loader:
#     n_kps = 0
#     n_imgs = 0
#     for imgs, _ in tqdm(train_loader):
#         for img in imgs:
#             img = (img.numpy().squeeze() * 255).astype(np.uint8).transpose((1, 2, 0))
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#             kp1, _ = key_points_extractor_1.run(img)
#
#             n_kps += len(kp1)
#             n_imgs += 1
#
#     print(f"images: {n_imgs} keypoints: {n_kps}  average: {n_kps / n_kps:.2f}")


if __name__ == "__main__":
    show_4()
    # count()
