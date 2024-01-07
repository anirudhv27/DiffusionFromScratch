import os

import matplotlib.pyplot as plt
from numpy import ndarray
from skimage import io
from torch.utils.data import DataLoader, Dataset


class CelebADataset(Dataset):
    """
    CelebA Dataset. Already centered and aligned.
    """

    def __init__(self, dir_path: str):
        """
        Initialize dataset to the directory with all images.
        """
        self.dir_path = dir_path
        self.len = 0
        for _ in os.scandir(dir_path):
            self.len += 1

        self.filename_num_chars = 6

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> ndarray:
        """
        Convert index to filename, return image as a tensor.
        """
        file_path = str(index + 1).rjust(self.filename_num_chars, "0") + ".jpg"
        img_arr: ndarray = io.imread(os.path.join(self.dir_path, file_path))
        return img_arr


if __name__ == "__main__":
    """
    Test Visualization.
    """
    curr_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(curr_dir, "../../data/img_align_celeba")
    celebs_dataset = CelebADataset(dataset_path)

    fig = plt.figure()

    for i, img in enumerate(celebs_dataset):
        print(i, img.shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title("Face #{}".format(i))
        ax.axis("off")
        plt.imshow(img)

        if i == 3:
            plt.show()
            break
