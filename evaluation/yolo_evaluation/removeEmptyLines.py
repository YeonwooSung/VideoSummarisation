def removeEmptyLines(input_filePath, output_filePath):
    with open(input_filePath) as infile, open(output_filePath, 'w') as outfile:
        for line in infile:
            if not line.strip():
                continue  # skip the empty line
            outfile.write(line)  # non-empty line. Write it to output


def addCommaToLines(intput_filePath, output_filePath):
    f = open(intput_filePath, 'r')
    f_output = open(output_filePath, "w")

    for x in f:
        line = x.strip()
        if ',' in line:
            f_output.write('{}\n\r'.format(line))
        else:
            line = line.replace('.jpg', '.jpg,')
            print(line)
            f_output.write('{}\n\r'.format(line))
    f.close()
    f_output.close()

if __name__ == '__main__':
    removeEmptyLines('./output.csv', './detected.csv')
    #addCommaToLines('./detected.csv', './output.csv')
