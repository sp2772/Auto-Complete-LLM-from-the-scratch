import os
import lzma
from tqdm import tqdm


def xz_files_in_dir(directory):
    filelist= [filename for filename in os.listdir(directory) if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename))]
    return filelist

if __name__ =="__main__":
    folder_path = "C:/MyEverything/PythonProjects/gpt-course/dataset"
    output_file_train = "train_split.txt"
    output_file_val = "val_split.txt"
    vocab_file = "vocab.txt"
    
    files = xz_files_in_dir(folder_path)
    total_files = len(files)
    print("Total number of files found:", total_files)
    split_index = int(total_files * 0.9)  # 90% for training
    files_train = files[:split_index]
    files_val = files[split_index:]
    
    vocab =set()
    
    with open(output_file_train,'w',encoding="utf-8") as outfile:
        for filename in tqdm(files_train,total=len(files_train)):
            file_path = os.path.join(folder_path,filename)
            with lzma.open(file_path,'rt',encoding="utf-8") as infile:
                text= infile.read()
                outfile.write(text)
                characters=set(text)
                vocab.update(characters)
    
    with open(output_file_val,'w',encoding="utf-8") as outfile:
        for filename in tqdm(files_val,total=len(files_val)):
            file_path = os.path.join(folder_path,filename)
            with lzma.open(file_path,"rt",encoding="utf-8") as infile:
                text = infile.read()
                outfile.write(text)
                characters = set(text)
                vocab.update(characters)
    
    with open(vocab_file,"w",encoding="utf-8") as vfile:
        for char in vocab:
            vfile.write(char+'\n')
                