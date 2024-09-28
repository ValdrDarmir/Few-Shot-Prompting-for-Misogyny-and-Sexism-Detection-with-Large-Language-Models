# This script combines the wordcloud images into a single one

# Author: Niklas Donhauser
# Date: September 26, 2024

# import libraries
import matplotlib.pyplot as plt
from PIL import Image
import sys


# combine the images
def combineImages(file_name, name):
    # exchange 5, 2 by the amount of pictures 5 == rows, 2 == columns
    fig, axs = plt.subplots(5, 2, figsize=(10, 10), dpi=300)

    axs = axs.flatten()

    for i, png_file in enumerate(file_name):
        img = Image.open(png_file)

        axs[i].imshow(img)
        axs[i].axis('off')

    plt.tight_layout()

    plt.savefig(f'../../05_results/visuals/combined_{name}.png', format='png')
    plt.savefig(f'../../05_results/visuals/combined_{name}.pdf', format='pdf',
                dpi=300)

    plt.show()


def main():
    all_files = ["path/[image_name_1].png",
                 "path/[image_name_2].png"]
    combineImages(all_files, "both")


if __name__ == '__main__':
    sys.exit(main())
