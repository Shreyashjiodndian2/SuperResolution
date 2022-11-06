from . import config
from matplotlib.pyplot import subplots, savefig, title, xticks, yticks, show
from tensorflow.keras.preprocessing.image import array_to_img
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os


def zoomIntoImages(image, imageTitle):
    (fig, ax) = subplots()
    im = ax.imshow(array_to_img(image[::-1]), origin="lower")
    title(imageTitle)
    axins = zoomed_inset_axes(ax, 2, loc=1)
    axins.imshow(array_to_img(image[::-1]), origin="lower")

    (x1, x2, y1, y2) = 20, 40, 20, 40

    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    xticks(visible=False)
    yticks(visible=False)

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    imagePath = os.path.join(config.BASE_IMAGE_PATH, f"{imageTitle}.png")
    savefig(imagePath)
    show()
