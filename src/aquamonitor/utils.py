from PIL import Image

def stack_images(img_list, orientation="vertical"):
    if orientation == "vertical":
        max_width = max([img.width for img in img_list])
        height = sum([img.height for img in img_list])
        new_img = Image.new("RGB", (max_width, height), "white")
        height_offset = 0
        for i, img in enumerate(img_list):
            new_img.paste(img, (0, height_offset))
            height_offset += img.height
        return new_img
    if orientation == "horizontal":
        max_height = max([img.height for img in img_list])
        width = sum([img.width for img in img_list])
        new_img = Image.new("RGB", (width, max_height), "white")
        width_offset = 0
        for i, img in enumerate(img_list):
            new_img.paste(img, (width_offset, 0))
            width_offset += img.width
        return new_img