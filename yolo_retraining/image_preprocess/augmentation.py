import cv2
import os


def rotate(image, angle=90, scale=1.0):
    '''
    Rotate the image
    :param image: image to be processed
    :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
    :param scale: Isotropic scale factor.
    '''
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image, M, (w, h))
    return image


def flip(image, vflip=False, hflip=False):
    '''
    Flip the image
    :param image: image to be processed
    :param vflip: whether to flip the image vertically
    :param hflip: whether to flip the image horizontally
    '''
    if hflip or vflip:
        if hflip and vflip:
            c = -1
        else:
            c = 0 if vflip else 1
        image = cv2.flip(image, flipCode=c)
    return image


def image_augment(file_path, save_path, index):
    '''
    Create the new image with imge augmentation

    :param file_path: the path of the original image
    :param save_path: the path to store the new image
    :param index: the index of augmented images
    '''
    img = cv2.imread(file_path)

    img_hflip  = flip(img, vflip=False, hflip=True)
    img_rotate = rotate(img)
    img_rotate_30 = rotate(img, angle=30, scale=0.8)
    img_rotate_60 = rotate(img, angle=60, scale=0.8)

    def writeImage(path, i, target_img):
        cv2.imwrite('{0}/{1}{2}.jpg'.format(path, path, index), target_img)
        return index + 1

    index = writeImage(save_path, index, img_hflip)
    index = writeImage(save_path, index, img_rotate)
    index = writeImage(save_path, index, img_rotate_30)
    index = writeImage(save_path, index, img_rotate_60)

    return index


def augmentationLoop(name):
    fileList = os.listdir('./{}'.format(name))
    startIndex = 1
    endIndex = 0

    # use for loop to iterate file list
    for fn in fileList:
        if fn == '.DS_Store':
            continue
        if os.path.isdir(os.path.join('./{}',format(name), fn)):
            continue

        endIndex += 1
    aug_start = endIndex + 1

    for fn in fileList:
        file_path = '{}/{}'.format(name, fn)
        if not fn.startswith(name):
            continue
        print(file_path)
        aug_start = image_augment(file_path, name, aug_start)

def main():
    dirList = os.listdir('./')

    # loop for directory list
    for dirName in dirList:
        # check if the current file is file (not directory)
        if not os.path.isdir(os.path.join('./', dirName)):
            continue
        # check if the current file is a dot file (or dot folder such as .dist)
        if dirName.startswith('.'):
            continue

        augmentationLoop(dirName)

if __name__ == '__main__':
    main()
