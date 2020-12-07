import numpy as np
import torch
from torch.autograd import Variable

# from .get_nets import PNet, RNet, ONet
# from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
# from first_stage import run_first_stage
import os, sys
from src import detect_faces
from PIL import Image, ImageDraw
import cv2
from src import matlab_cp2tform
import scipy.io as sio
from glob import glob


def find_bb_ldv(root="/data3/imgc/", targetfile="./output.txt", out_dir="../"):
    with open(targetfile, "w") as f:
        dirs = os.listdir(root)
        for idx, dir in enumerate(dirs):
            sys.stdout.write(str(idx + 1) + "/" + str(len(dirs)) + "\n")
            individual_dir = os.path.join(root + dir)
            individual_dir_o = os.path.join(out_dir + dir)
            frames = os.listdir(individual_dir)
            if not os.path.exists(individual_dir_o):
                os.makedirs(individual_dir_o)
            for frame in frames:
                frame_dir = os.path.join(individual_dir, frame)
                frame_dir_o = os.path.join(individual_dir_o, frame)
                imgs = os.listdir(frame_dir)
                if not os.path.exists(frame_dir_o):
                    os.makedirs(frame_dir_o)
                for img in imgs:
                    imgname = os.path.join(frame_dir, img)
                    crop_img_name = os.path.join(frame_dir_o, img)
                    if not os.path.exists(crop_img_name):
                        try:
                            image = Image.open(imgname)
                            bounding_boxes, landmarks = detect_faces(image)
                            nolandmard = 0
                            inx = 0
                            if len(landmarks) > 1:
                                minIoU = -1
                                for i, bb in enumerate(bounding_boxes):
                                    x1a, y1a, x2a, y2a, _ = bounding_boxes[i]
                                    IoU = (x2a - x1a) * (y2a - y1a)
                                    if IoU > minIoU:
                                        minIoU = IoU
                                        inx = i
                            elif len(landmarks) == 0:
                                nolandmard = 1
                        except:
                            nolandmard = 1
                        if not nolandmard:
                            # f.write(dir + "/" + img+","+str(bounding_boxes[inx][0])+","+str(bounding_boxes[inx][1])+","+str(bounding_boxes[inx][2])+","+str(bounding_boxes[inx][3])+",")
                            # for i in range(5):
                            #     f.write(str(landmarks[inx][i])+","+str(landmarks[inx][5+i]))
                            # f.write("\n")
                            lm = []
                            for i in range(5):
                                lm.append([landmarks[inx][i], landmarks[inx][5 + i]])
                            image = np.asarray(image)
                            image = alignment(image, lm)
                            crop_img = image[:, :, ::-1]
                            # if not os.path.exists(out_dir + dir):
                            #     os.makedirs(out_dir + dir)
                            # print(dir)
                            cv2.imwrite(crop_img_name, crop_img)


def find_bb_ld(root="/data3/imgc/", targetfile="./output.txt", out_dir="../"):
    with open(targetfile, "w") as f:
        dirs = os.listdir(root)
        for idx, dir in enumerate(dirs):
            sys.stdout.write(str(idx + 1) + "/" + str(len(dirs)) + "\n")
            imgs = os.listdir(root + dir)
            for img in imgs:
                imgname = root + dir + "/" + img
                crop_img_name = out_dir + dir + "/" + img
                if not os.path.exists(crop_img_name):
                    try:
                        image = Image.open(imgname)
                        bounding_boxes, landmarks = detect_faces(image)
                        nolandmard = 0
                        inx = 0
                        if len(landmarks) > 1:
                            minIoU = -1
                            for i, bb in enumerate(bounding_boxes):
                                x1a, y1a, x2a, y2a, _ = bounding_boxes[i]
                                IoU = (x2a - x1a) * (y2a - y1a)
                                if IoU > minIoU:
                                    minIoU = IoU
                                    inx = i
                        elif len(landmarks) == 0:
                            nolandmard = 1
                    except:
                        nolandmard = 1
                    if not nolandmard:
                        # f.write(dir + "/" + img+","+str(bounding_boxes[inx][0])+","+str(bounding_boxes[inx][1])+","+str(bounding_boxes[inx][2])+","+str(bounding_boxes[inx][3])+",")
                        # for i in range(5):
                        #     f.write(str(landmarks[inx][i])+","+str(landmarks[inx][5+i]))
                        # f.write("\n")
                        lm = []
                        for i in range(5):
                            lm.append([landmarks[inx][i], landmarks[inx][5 + i]])
                        image = np.asarray(image)
                        image = alignment(image, lm)
                        crop_img = image[:, :, ::-1]
                        if not os.path.exists(out_dir + dir):
                            os.makedirs(out_dir + dir)
                            # print(dir)
                        cv2.imwrite(crop_img_name, crop_img)


def find_bb_ld_ytf(
    imgname="/home/diqong/Documents/Zhejiang University Email system_files/2.jpg",
    crop_img_name="/home/diqong/Documents/Zhejiang University Email system_files/3.jpg",
):

    try:
        imgname = imgname.replace("\n", "")
        # imgname=imgname.replace("image00002.jpg","image00032.jpg")
        # print imgname
        image = Image.open(imgname)
        bounding_boxes, landmarks = detect_faces(image)

        # print bounding_boxes, landmarks, imgname
        # exit()

        nolandmard = 0
        inx = 0
        if len(landmarks) >= 1:
            minIoU = -1
            for i, bb in enumerate(bounding_boxes):
                x1a, y1a, x2a, y2a, _ = bounding_boxes[i]
                IoU = (x2a - x1a) * (y2a - y1a)
                if IoU > minIoU:
                    minIoU = IoU
                    inx = i
            print(imgname)
        elif len(landmarks) == 0:
            nolandmard = 1
            print("-" * 10, imgname)
    except:
        nolandmard = 1

    if not nolandmard:
        # f.write(dir + "/" + img+","+str(bounding_boxes[inx][0])+","+str(bounding_boxes[inx][1])+","+str(bounding_boxes[inx][2])+","+str(bounding_boxes[inx][3])+",")
        # for i in range(5):
        #     f.write(str(landmarks[inx][i])+","+str(landmarks[inx][5+i]))
        # f.write("\n")
        # print imgname
        lm = []
        for i in range(5):
            lm.append([landmarks[inx][i], landmarks[inx][5 + i]])
        image = np.asarray(image)

        image = alignment(image, lm)

        crop_img = image[:, :, ::-1]
        outdir = crop_img_name[: crop_img_name.rfind("/")]
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            print(outdir)
        cv2.imwrite(crop_img_name, crop_img)


def find_bb_ld_img(
    imgname="/home/diqong/Documents/Zhejiang University Email system_files/2.jpg",
    crop_img_name="/home/diqong/Documents/Zhejiang University Email system_files/3.jpg",
):
    """
    imgname=imgname.replace("image00002.jpg","image00032.jpg")
    #print imgname
    image=Image.open(imgname)

    drawObject = ImageDraw.Draw(image)
    matObj = sio.loadmat(imgname.replace('jpg', 'mat'))
    print(imgname)
    lms = matObj['pt3d_68'][:2]
    print(lms)
    for i in range(lms.shape[1]):
        print(i)
        drawObject.ellipse((lms[0][i]-1,lms[1][i]-1,lms[0][i]+1,lms[1][i]+1),fill = "red")

    image.save('/data2/lmd_jdq/AFLW2000-3D/test.jpg')
    print('OK')
    fuck
    exit(1)
    """

    if os.path.exists(imgname.replace(".jpg", ".npy")):
        pass
    # try:
    if True:
        imgname = imgname.replace("\n", "")
        # imgname=imgname.replace("image00002.jpg","image00032.jpg")
        # print imgname
        image = Image.open(imgname)
        bounding_boxes, landmarks = detect_faces(image)

        # print bounding_boxes, landmarks, imgname
        # exit()

        nolandmard = 0
        inx = 0
        if len(landmarks) >= 1:
            minIoU = -1
            for i, bb in enumerate(bounding_boxes):
                x1a, y1a, x2a, y2a, _ = bounding_boxes[i]
                IoU = (x2a - x1a) * (y2a - y1a)
                if IoU > minIoU:
                    minIoU = IoU
                    inx = i
            print(imgname)
            np.save(imgname.replace(".jpg", ".npy"), bounding_boxes[inx])
        elif len(landmarks) == 0:
            nolandmard = 1
            print("-" * 10, imgname)
            return 0
    # except:
    #    nolandmard=1
    """
    re-align AFLW2000-3D
    start
    """
    return 0
    matObj = sio.loadmat(imgname.replace("jpg", "mat"))
    lms = matObj["pt3d_68"][:2]

    lm = []
    lm.append((lms[:, 36] + lms[:, 39]) / 2)
    lm.append((lms[:, 42] + lms[:, 45]) / 2)
    lm.append(lms[:, 30])
    lm.append(lms[:, 48])
    lm.append(lms[:, 54])

    nolandmard = False

    """
    end
    """

    if not nolandmard:
        # f.write(dir + "/" + img+","+str(bounding_boxes[inx][0])+","+str(bounding_boxes[inx][1])+","+str(bounding_boxes[inx][2])+","+str(bounding_boxes[inx][3])+",")
        # for i in range(5):
        #     f.write(str(landmarks[inx][i])+","+str(landmarks[inx][5+i]))
        # f.write("\n")
        # print imgname
        # lm=[]
        # for i in range(5):
        #    lm.append([ landmarks[inx][i], landmarks[inx][5+i] ])
        image = np.asarray(image)

        matObj = sio.loadmat(imgname.replace("jpg", "mat"))

        image, matObj_align = alignment(image, lm, matObj)

        crop_img = image[:, :, ::-1]
        # if not os.path.exists(out_dir + dir):
        #     os.makedirs(out_dir + dir)
        # print(dir)
        cv2.imwrite(crop_img_name, crop_img)
        sio.savemat(crop_img_name.replace("jpg", "mat"), matObj_align)


def alignment(src_img, src_pts, matObj=None):
    of = 0
    a = 112.0 / 112  # height
    b = 96.0 / 96  # width
    ref_pts = [
        [30.2946 * b + of, 51.6963 * a + of],
        [65.5318 * b + of, 51.5014 * a + of],
        [48.0252 * b + of, 71.7366 * a + of],
        [33.5493 * b + of, 92.3655 * a + of],
        [62.7299 * b + of, 92.2041 * a + of],
    ]
    crop_size = (96 + of * 2, 112 + of * 2)
    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = matlab_cp2tform.get_similarity_transform_for_cv2(s, r)
    # [r, r, tx]
    # [r, r, ty]

    # tfm

    # print(tfm)
    # exit
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    if matObj is not None:

        ori_lms = np.array(matObj["pt3d_68"][:2])

        # print(lms.shape)

        one = np.ones(68).reshape(1, 68)
        lms = np.r_[ori_lms, one]
        lms = np.matmul(tfm, lms)

        matObj["pt2d_68"] = lms
        # print(lms)
        # fuck
        # print(matObj['pt2d'])

        # exit(0)
        return face_img, matObj
    else:
        return face_img


def alignment_3ddfa(src_img, src_pts, trans):
    """
    trans: 12
    """
    of = 0
    a = 112 / 112
    b = 96 / 96
    ref_pts = [
        [30.2946 * b + of, 51.6963 * a + of],
        [65.5318 * b + of, 51.5014 * a + of],
        [48.0252 * b + of, 71.7366 * a + of],
        [33.5493 * b + of, 92.3655 * a + of],
        [62.7299 * b + of, 92.2041 * a + of],
    ]
    crop_size = (96 + of * 2, 112 + of * 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = matlab_cp2tform.get_similarity_transform_for_cv2(s, r)
    # [r, r, tx]
    # [r, r, ty]

    # tfm

    # print(tfm)
    # exit
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    m = tfm[:2, :2]
    m = np.matmul(m, m.transpose(1, 0))
    print(np.sqrt(m[0, 0]))
    exit()

    ori_lms = np.array(matObj["pt3d_68"][:2])

    # print(lms.shape)

    one = np.ones(68).reshape(1, 68)
    lms = np.r_[ori_lms, one]
    lms = np.matmul(tfm, lms)

    matObj["pt2d_68"] = lms
    # print(lms)
    # fuck
    # print(matObj['pt2d'])

    # exit(0)
    return face_img, matObj


def face_alignment(root="../", out_dir="../", filename="./output.txt"):
    with open(filename) as f:
        s = f.readline()
        while len(s) != 0:

            sa = s.split(",")
            id_name = sa[0].split("/")
            imgname = sa[0]
            full_img_name = root + imgname

            # x1,y1,x2,y2=int(sa[1]),int(sa[2]),int(sa[3]),int(sa[4])
            lm = []
            for i in range(5):
                lm.append([float(sa[i + 5]), sa[10 + i]])
            img = Image.open(full_img_name)
            img = np.asarray(img)
            img = alignment(img, lm)
            crop_img = img[:, :, ::-1]
            crop_img_name = out_dir + sa[0]

            if not os.path.exists(out_dir + id_name[0]):
                os.makedirs(out_dir + id_name[0])
                print(id_name[0])
            cv2.imwrite(crop_img_name, crop_img)
            s = f.readline()


def main():

    # filelists="/data2/lmd_jdq/AFLW2000-3D/file_path_list_AFLW2000.txt"
    # filelists="/ssd-data/lmd/file_path_list_agedb.txt"
    # filelists="/ssd-data/lmd/eval_dbs/file_path_list_cfp.txt"
    # filelists="/ssd-data/lmd/eval_dbs/file_path_list_lfw.txt"
    # filelists="/ssd-data/lmd/file_path_list_lfw_ori.txt"
    # filelists="/data1/lmd2/file_path_list_ytf_ori.txt"

    # #find_bb_ld_ytf('/data1/lmd2/ytf_ori/Aaron_Eckhart/0/0.555.jpg', '/data1/lmd2/ytf_112x112/Aaron_Eckhart/0/0.555.jpg')
    # #exit()
    # #exit(0)
    # #print(1)
    # import concurrent.futures
    # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    #     with open(filelists,"r") as f:
    #         #imgnames=f.readlines()
    #         #print(len(imgnames))
    #         count = 0
    #         for imgname in f:
    #             imgname = imgname[:-1]

    #             #sys.stdout.write(str(count)+"\n")
    #             file_output=imgname.replace('ytf_ori', 'ytf_112x112/')
    #             #print imgname, file_output
    #             executor.submit(find_bb_ld_ytf, imgname, file_output)
    #             #find_bb_ld_img(imgname=imgname,crop_img_name=file_output)
    #             count += 1
    # #find_bb_ldv(root="/media/diqong/2e0efd0a-80d6-43d4-99e6-e1fbf0527b67/diqiong/database/aligned_images_DB/",out_dir="/media/diqong/b3a57ec9-13c5-4b82-aea5-40a469de3177/ytf/")

    # #face_alignment(root="/media/diqong/b3a57ec9-13c5-4b82-aea5-40a469de3177/imgc/",out_dir="/media/diqong/b3a57ec9-13c5-4b82-aea5-40a469de3177/imgc2/")
    """
    img = Image.open('/ssd-data/lmd/train_aug_120x120/LFPWFlip_LFPW_image_train_0859_12_2.jpg')
    img = np.asarray(img)
    lms = [[36.1884, 86.3004],
        [41.7595, 62.9591],
        [62.9924, 95.5168],
        [75.9926, 81.8416],
        [78.8219, 68.2653]]
    alignment_3ddfa(img, lms, None)
    """


def align_test():
    inputdir = "../test"
    outdir = "../test_align"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    types = ("*.jpg", "*.png")
    image_path_list = []
    for files in types:
        image_path_list.extend(glob(os.path.join(inputdir, files)))
    total_num = len(image_path_list)
    for i, imgname in enumerate(image_path_list):
        # imgname=imgname.replace("\n","")
        # imgname=imgname.replace("image00002.jpg","image00032.jpg")
        # print imgname
        image = Image.open(imgname).convert("RGB")
        print(np.asarray(image))
        # print(image.shape)
        # exit()
        bounding_boxes, landmarks = detect_faces(image)

        # print bounding_boxes, landmarks, imgname
        # exit()

        nolandmard = 0
        inx = 0
        if len(landmarks) >= 1:
            minIoU = -1
            for i, bb in enumerate(bounding_boxes):
                x1a, y1a, x2a, y2a, _ = bounding_boxes[i]
                IoU = (x2a - x1a) * (y2a - y1a)
                if IoU > minIoU:
                    minIoU = IoU
                    inx = i
        elif len(landmarks) == 0:
            nolandmard = 1
            print("-" * 10, imgname)

        if not nolandmard:
            # f.write(dir + "/" + img+","+str(bounding_boxes[inx][0])+","+str(bounding_boxes[inx][1])+","+str(bounding_boxes[inx][2])+","+str(bounding_boxes[inx][3])+",")
            # for i in range(5):
            #     f.write(str(landmarks[inx][i])+","+str(landmarks[inx][5+i]))
            # f.write("\n")
            # print imgname
            lm = []
            for i in range(5):
                lm.append([landmarks[inx][i], landmarks[inx][5 + i]])
            image = np.asarray(image)

            image = alignment(image, lm)

            crop_img = image[:, :, ::-1]
            crop_img_name = os.path.join(outdir, os.path.basename(imgname))
            print(crop_img_name)
            # outdir = crop_img_name[:crop_img_name.rfind('/')]
            # if not os.path.exists(outdir):
            #     os.makedirs(outdir)
            #     print(outdir)
            cv2.imwrite(crop_img_name, crop_img)


if __name__ == "__main__":

    align_test()
    # lm = []
    # for i in range(5):
    #     lm.append([landmarks[inx][i], landmarks[inx][5+i]])
