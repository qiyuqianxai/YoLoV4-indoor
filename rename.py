import os
oldpath=r"C:\Users\wanghao\Desktop\新建文件夹"
filelist=os.listdir(path)
count=0
for file in filelist:
    olddir=os.path.join(path,file)
    if os.path.isdir(olddir):
        continue
    filename=os.path.splitext(file)[0]
    filetype=".jpg"
    newdir=os.path.join(path,str(count).zfill(3)+filetype)
    count+=1
    print(newdir)
    os.rename(olddir,newdir)