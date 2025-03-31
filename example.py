from hill import AdvHill
from image import read_image
import numpy as np
img, _ = read_image("docs/images/face.png")
# print(img)



hill = AdvHill(img=img)
hill._generate_key(img.shape[0])
print(hill.A)
print(np.dot(hill.A, hill.A))