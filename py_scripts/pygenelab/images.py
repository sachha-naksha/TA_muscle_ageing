# images.py 

"""
functions relating to modifying images
"""

# imports
import io
import math
from PIL import Image
from pathlib import Path
import matplotlib as mpl


# combine_two_figures
def combine_two_figures(fig1, fig2, layout="horizontal", save_path=None, dpi=200):
    """
    combine two matplotlib figures into one image
    """

    def fig_to_image(fig):
        # save figure into memory
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)

        # convert figure to image
        return Image.open(buffer).convert("RGB")

    # convert both figures to images
    img1 = fig_to_image(fig1)
    img2 = fig_to_image(fig2)

    # combine figures side by side
    if layout == "horizontal":
        new_width = img1.width + img2.width
        new_height = max(img1.height, img2.height)

        combined_img = Image.new("RGB", (new_width, new_height), "white")
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.width, 0))

    # combine figures top to bottom
    elif layout == "vertical":
        new_width = max(img1.width, img2.width)
        new_height = img1.height + img2.height

        combined_img = Image.new("RGB", (new_width, new_height), "white")
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (0, img1.height))

    else:
        raise ValueError("layout must be either 'horizontal' or 'vertical'")

    # save combined image if needed
    if save_path is not None:
        combined_img.save(save_path)

    # return combined image
    return combined_img


# combine_multiple_figures
def combine_multiple_figures(figures, layout="horizontal", n_cols=None, save_path=None, dpi=200):
    """
    combine multiple matplotlib figures into one image

    layout:
    horizontal,
    vertical,
    grid (use n_cols to specify the number of columns in the grid)
    """

    if len(figures) == 0:
        raise ValueError("figures list is empty")

    def fig_to_image(fig):
        # save figure into memory
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)

        # convert figure to image
        return Image.open(buffer).convert("RGB")

    # convert all figures to images
    images = [fig_to_image(fig) for fig in figures]

    # combine figures in one row
    if layout == "horizontal":
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        combined_img = Image.new("RGB", (total_width, max_height), "white")

        x_offset = 0
        for img in images:
            combined_img.paste(img, (x_offset, 0))
            x_offset += img.width

    # combine figures in one column
    elif layout == "vertical":
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        combined_img = Image.new("RGB", (max_width, total_height), "white")

        y_offset = 0
        for img in images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height

    # combine figures in a grid
    elif layout == "grid":
        if n_cols is None:
            n_cols = 2

        n_rows = math.ceil(len(images) / n_cols)

        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        total_width = n_cols * max_width
        total_height = n_rows * max_height

        combined_img = Image.new("RGB", (total_width, total_height), "white")

        for idx, img in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols

            x_offset = col * max_width
            y_offset = row * max_height

            combined_img.paste(img, (x_offset, y_offset))

    else:
        raise ValueError("layout must be 'horizontal', 'vertical', or 'grid'")

    # save combined image if needed
    if save_path is not None:
        combined_img.save(save_path)

    # return combined image
    return combined_img


def save_fig_as_png(
    fig,
    output_path,
    dpi=300,
    bbox_inches="tight",
    transparent=False
):
    """
    save one matplotlib figure as a png file
    """

    # save_mpl_fig_png
    # api:
    # save_mpl_fig_png(
    #     fig=fig,
    #     output_path="volcano_plot.png",
    #     dpi=300,
    # )

    # convert output path to Path object
    output_path = Path(output_path)

    # create parent folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # make sure file extension is png
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")

    # save figure
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent
    )

    # return saved path
    return output_path


# save_editable_svg
def save_editable_svg(
    fig,
    output_path,
    bbox_inches="tight",
    transparent=True,
    dpi=300
):
    """
    save a matplotlib figure as an editable svg file

    text will remain editable in illustrator / inkscape when possible
    """

    # save_editable_svg
    # api:
    # save_editable_svg(
    #     fig=fig,
    #     output_path="figure.svg",
    # )

    # convert path to pathlib path
    output_path = Path(output_path)

    # make sure parent folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # keep text as editable text instead of paths
    mpl.rcParams["svg.fonttype"] = "none"

    # save figure as svg
    fig.savefig(
        output_path,
        format="svg",
        bbox_inches=bbox_inches,
        transparent=transparent,
        dpi=dpi
    )

    # return saved path
    return output_path
