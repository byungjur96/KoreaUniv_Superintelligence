import os, glob, shutil

img_path_list = sorted(glob.glob("/Dataset/Quilt-1M/images/*.jpg"))
# 65,013
train_pths = img_path_list[:3000]
test_pths  = img_path_list[-300:]

train_dir = "/Dataset/Quilt-1M/train"
test_dir  = "/Dataset/Quilt-1M/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for train_pth in train_pths:
    fn = os.path.basename(train_pth)
    destination = os.path.join(train_dir, fn)
    shutil.copyfile(train_pth, destination)

for test_pth in test_pths:
    fn = os.path.basename(test_pth)
    destination = os.path.join(test_dir, fn)
    shutil.copyfile(test_pth, destination)

# shutil.copyfile()