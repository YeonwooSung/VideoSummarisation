import os

basedir = './'

dirList = os.listdir(basedir)
index = 1

for fn in dirList:
    if not os.path.isdir(os.path.join(basedir, fn)):
        continue

    os.rename(os.path.join(basedir, fn), os.path.join(basedir, '00{}'.format(index)))
    index += 1
