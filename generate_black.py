import os
from PIL import Image
import numpy as np

folder = 'masks_black'
if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(200):
    image = np.zeros((256,256), dtype=np.uint8)
    img = Image.fromarray(image)
    name = os.path.join(folder, f'{i:04}.png')
    img.save(name)

print(f'Generated 200 images at {folder}')