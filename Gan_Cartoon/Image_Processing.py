from PIL import Image
import os

image_dir = "E:\\Dataset\\anime_face"
image_output_dir = "E:\\Dataset\\anime_face_resize"
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
files = os.listdir(image_dir)
for file in files:
    image_path = os.path.join(image_dir,file)
    image = Image.open(image_path,'r')
    image = image.resize((48,48))
    output_path = os.path.join(image_output_dir,file)
    image.save(output_path)


