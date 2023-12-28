from PIL import Image

def resizeimg(img):
    target_size = (512, 512)
    resized_img = img.resize(target_size, Image.LANCZOS)
    return resized_img