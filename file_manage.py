# file management function

# import all packages
import os
import shutil

# 
def del_files_folder(folder_path, del_folder=False):

  # iterate all files
  for filename in os.listdir(folder_path):
      file_path = os.path.join(folder_path, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)  # del files
          elif os.path.isdir(file_path):
            if del_folder == True:
                shutil.rmtree(file_path)  # del subfolders
            else:
                pass
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

  print("All files in the folder have been deleted.")
