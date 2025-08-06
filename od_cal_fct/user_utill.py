# Import necessary modules.
import os, shutil
import pickle
import gzip

# Define function to list up all files in the directory.
def fileListCreator(strDirPath:str = "./", lv_flt= False, ext_flt= "csv"):
    
    absDirPath = os.path.abspath(strDirPath) # Convert to absolute path.

    listDir_allFiles = [] # List for paths for all types of files. 
    listNameDupFlt = [] # List to filter out duplicated file.

    # os.walk will return tuple of file path combinations including all sub-folders
    for (path, _, files) in os.walk(absDirPath):        
        for name in files:
            if name not in listNameDupFlt: # Duplicated one won't be appended in target list. 
                listNameDupFlt.append(name)
                # In below list, all paths for files (not directory) will be added.
                absfilePath = os.path.join(path, name).replace("\\","/")
                listDir_allFiles.append(absfilePath)
            else:
                continue
    
    # Filter out with extention input
    if lv_flt:
        listDir_ext_flt = list(filter(lambda x: ext_flt.replace(".","") == x.split(".")[-1], listDir_allFiles))
    else:
        listDir_ext_flt = listDir_allFiles    
    
    return listDir_ext_flt

# Define function to delete all files in the directory.
def delAllInDir(strDirPath:str):
    
    for filename in os.listdir(strDirPath):        
        file_path = os.path.join(strDirPath, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Function to save pickle file           
def savePklFile(strPathPklFile:str, varToBeStored):
    
    with gzip.open(strPathPklFile, "wb") as f:
        pickle.dump(varToBeStored,f)
        
# Function to load pickle file
def loadPklFile(strPathPklFile:str):
    
    with gzip.open(strPathPklFile, "rb") as f:
        loadedFile = pickle.load(f)
    
    return loadedFile