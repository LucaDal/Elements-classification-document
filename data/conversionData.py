import os


def convert_space(text):
    new_content = []
    flag_space = False
    start = False
    for line in text:
        space_count = 1
        new_line = ''
        print(len(line))
        for index in range(len(line)):
            if line[index] != ' ':
                new_line += line[index]
                flag_space = False
                start = True
            else:
                if start and 11 > space_count > 0 and not flag_space :
                    new_line += ','
                    flag_space = True
                    space_count += 1
            if line[index] == ' ' and line[index + 1] == '.':
                new_line += '0'
        start = False
        flag_space = False
        new_content.append(new_line)
    return new_content


with open('page-blocks.data') as inFile:
    content = inFile.readlines()
    new = convert_space(content)
    with open('page_blocks_data.arff', 'w') as newFile:
        newFile.writelines(new)
