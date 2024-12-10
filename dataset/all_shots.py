import os
import random


def save_files(random_files):
    # Write the list of random files to the output file
    with open(output_file, 'w') as f:
        for file in random_files:
            f.write(file + '\t')

    print(f"List of {number_of_files} random files saved to {output_file}")

def save_random_files_list(folder_path1, folder_path2, number_of_files, output_file):
    # Get a list of all files in the specified folder
    files = [f for f in os.listdir(folder_path1) if os.path.isfile(os.path.join(folder_path1, f))]

    # Check if the number of files requested is not more than the available files
    if number_of_files > len(files):
        print(f"Only {len(files)} files available, returning all.")
        number_of_files = len(files)

    # Randomly select the specified number of files
    random_files1 = random.sample(files, number_of_files)


    files = [f for f in os.listdir(folder_path2) if os.path.isfile(os.path.join(folder_path2, f))]

    # Check if the number of files requested is not more than the available files
    if number_of_files > len(files):
        print(f"Only {len(files)} files available, returning all.")
        number_of_files = len(files)

    # Randomly select the specified number of files
    random_files2 = random.sample(files, number_of_files)
    random_files = ['a-0:'] + random_files2 + ['\n'] + ['n-0:'] + random_files1
    save_files(random_files)


# Example usage:
task = "Skin_ISIC2019"
folder_path1 = f"/Users/bcw3zj/PycharmProjects/MVFA-AD/data/{task}_AD/valid/good/img"
folder_path2 = f"/Users/bcw3zj/PycharmProjects/MVFA-AD/data/{task}_AD/valid/Ungood/img"
save_path = f"/Users/bcw3zj/PycharmProjects/MVFA-AD/dataset/fewshot_seed/{task}/"
number_of_files = min(len(os.listdir(folder_path1)), len(os.listdir(folder_path2)))  # Specify the number of random files you want
# number_of_files = 16
output_file = save_path + f"{number_of_files}-shot.txt"  # Output file name

save_random_files_list(folder_path1, folder_path2, number_of_files, output_file)
