import os
import PIL
from PIL import Image

dirList = os.listdir('./')

for dirName in dirList:
    if not os.path.isdir(os.path.join('./', dirName)):
        continue

    entries = os.listdir(dirName)

    for entry in entries:
        filePath = dirName + '/' + entry

        img = Image.open(filePath)
        width, height = img.size

        print('width = {}, height = {}'.format(width, height))

        if height <= 500 and width <= 500:
            continue

        if height < 1000 and width < 1000:
            continue

        ratio = 1 / 2
        if height > 1500 or width > 1500:
            ratio = 1 / 3

        newHeight = int(height * ratio)
        newWidth = int(width * ratio)
        img = img.resize((newWidth, newHeight), Image.ANTIALIAS)
        img.save(filePath)
