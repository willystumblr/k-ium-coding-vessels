from PIL import Image, ImageEnhance, ImageOps

def preprocess(img: Image) -> Image:
    """
    Image preprocessing function; when given the image, it is resized, inverted, sharpened,
    and contrasted.
        - `img`: Image instance
    """

    inverted = ImageOps.invert(img)
    #display(inverted)


    sharpened = ImageEnhance.Sharpness(inverted)
    sharpened = sharpened.enhance(2)
    #display(sharpened)

    iandsandocnt = ImageEnhance.Contrast(sharpened)
    iandsandocnt = iandsandocnt.enhance(1.25)
    #display(iandsandocnt)

    return iandsandocnt

def crop(img: Image) -> Image:
    width, height = img.size   # Get dimensions
    new_width = height

    left = (width - new_width)/2
    right = (width + new_width)/2


    # Crop the center of the image
    img = img.crop((left, 0, right, height))
    return img