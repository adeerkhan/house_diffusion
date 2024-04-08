import os

def get_image_number(name):
    return int(name.split(".")[0])

def write_filenames_to_txt(directory, txt_file):
    json_names = os.listdir(directory)
    json_names = sorted(json_names, key=get_image_number)
    with open(txt_file, 'w') as f:
        for filename in json_names:
            f.write(filename + '\n')


write_filenames_to_txt("rplan", "list.txt")