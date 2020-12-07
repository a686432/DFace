import re

f = open("/data/jdq/AFLW2000-3D/file_path_list_AFLW2000_align.txt", "r")
alllines = f.readlines()
f.close()
f = open("/data/jdq/AFLW2000-3D/file_path_list_AFLW2000_align.txt", "w+")
for eachline in alllines:
    a = re.sub("data2/lmd_jdq", "data/jdq", eachline)
    f.writelines(a)
f.close()
