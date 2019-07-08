
import os
import shutil
#this path will move all image with same label to same folder in rvl-cdip dataset
for file_name  in  os.listdir("./labels_only/labels") :
 f_train=open("./labels_only/labels/"+file_name)
 lines=f_train.readlines()
 if  not  os.path.exists("./datasets/"+file_name.split(".")[0]) :
  os.mkdir("./datasets/"+file_name.split(".")[0])
 for  line  in  lines :
  file,label = line.split(" ")
  
  print file
  if not os.path.exists("./datasets/" + file_name.split(".")[0] + "/" + label.strip()) :
   os.mkdir("./datasets/" + file_name.split(".")[0] + "/" + label.strip())
  shutil.move("./rvl-cdip/images/" + file,"./datasets/" + file_name.split(".")[0] + "/"+label.strip() + "/" + file.split("/")[-1])

map_2_data={"ADVE":"4","Email":"2","Form":"1","Letter":"0","Memo":"15","News":"9","Note":"3","Report":"5","Resume":"14","Scientific":"6"}


## this path will remove all image of rvl-cdip contained in tobaco3482
for key , value in map_2_data.items() :
 arr_tobaco = os.listdir("Tobacco3482/"+key)
 arr_rvl = os.listdir("datasets/train/"+value)
 for s in arr_tobaco :
  for t in arr_rvl :
   if s.split(".")[0].strip() == t.split(".")[0].strip() :
    os.remove("datasets/train/"+value+"/"+t)
