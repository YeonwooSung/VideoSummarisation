import os

dirList = os.listdir('./')

for dirName in dirList:
    if not os.path.isdir(os.path.join('./', dirName)):
        continue

    entries = os.listdir(dirName)

    for entry in entries:
        newFileName = entry.lower().replace('raw', '')
        os.rename('{0}/{1}'.format(dirName,entry), '{0}/{1}'.format(dirName, newFileName))
