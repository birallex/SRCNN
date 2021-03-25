from PIL import Image
import os

def process_folder(folder_name, new_width, new_height):
    new_folder_name = folder_name + "_resized/"
    os.mkdir(new_folder_name)
    for img_name in os.listdir(folder_name):
        img = Image.open(folder_name + "/" + img_name)
        img = img.resize((new_width, new_height))
        img.save(new_folder_name + img_name)
    print(folder_name + " was resized!")

if __name__ == "__main__":
    process_folder("ft_image", 672, 496)
