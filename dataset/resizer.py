from PIL import Image
import os

def process_folder_current_resolution(folder_name, new_width, new_height, prefix):
    new_folder_name = folder_name + prefix + "/"
    os.mkdir(new_folder_name)
    for img_name in os.listdir(folder_name):
        img = Image.open(folder_name + "/" + img_name)
        img = img.resize((new_width, new_height))
        img.save(new_folder_name + img_name)
    print(folder_name + " was resized!")


def process_folder(folder_name, resize_koeff, prefix):
    new_folder_name = folder_name + prefix + "/"
    os.mkdir(new_folder_name)
    for img_name in os.listdir(folder_name):
        img = Image.open(folder_name + "/" + img_name)
        new_width, new_height = img.size[0] // resize_koeff, img.size[1] // resize_koeff
        img = img.resize((new_width, new_height))
        img.save(new_folder_name + img_name)
    print(folder_name + " was resized!")

if __name__ == "__main__":
    process_folder_current_resolution("photos", 672, 496, "_resized_current")
    process_folder("photos", 2, "_resized")