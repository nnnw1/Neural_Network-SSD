import numpy as np
import os

threshold = 0.5

gtdir = "data/train/annotations/"
output_dir = "output/training/"

gt_txtnames = os.listdir(gtdir)
output_txtnames = os.listdir(output_dir)

train_test_split = 0.9
split = int(len(gt_txtnames) * train_test_split)
start_img = 0
end_img = len(gt_txtnames)
# start_img = 0
# end_img = split

gt_txtnames = gt_txtnames[start_img:end_img]
output_txtnames = output_txtnames[start_img:end_img]

length = len(gt_txtnames)


TP = np.zeros([length, 3])
FP = np.zeros([length, 3])
FN_TP = np.zeros([length, 3])


def calculate_iou(pred_box, ann_box):
    pred_w = pred_box[2]
    pred_h = pred_box[3]
    pred_xmin = pred_box[0]
    pred_ymin = pred_box[1]
    predBox = [pred_xmin, pred_ymin, pred_w, pred_h]

    ann_w = ann_box[2]
    ann_h = ann_box[3]
    ann_xmin = ann_box[0]
    ann_ymin = ann_box[1]
    annBox = [ann_xmin, ann_ymin, ann_w, ann_h]

    inter_box_top_left = [max(annBox[0], predBox[0]), max(annBox[1], predBox[1])]
    inter_box_bottom_right = [min(annBox[0]+annBox[2], predBox[0]+predBox[2]), min(annBox[1]+annBox[3], predBox[1]+predBox[3])]
    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]
    intersection = inter_box_w * inter_box_h
    union = annBox[2] * annBox[3] + predBox[2] * predBox[3] - intersection
    
    iou = intersection / union

    return iou

sum = np.array([0, 0, 0])

for idx in range(length):
    gt_txt_name = gtdir + gt_txtnames[idx]
    output_txt_name = output_dir + output_txtnames[idx]

    file1 = open(gt_txt_name, "r")
    file2 = open(output_txt_name, "r")

    lines1 = file1.readlines()
    lines2 = file2.readlines()

    length1 = len(lines1)
    length2 = len(lines2)

    gt_class = np.zeros(length1, np.int8)
    gt_box = np.zeros([length1, 4], np.float32)
    count = 0
    for line in lines1:
        params = line.split()
        class_id, x, y, w, h = params
        class_id = int(class_id)
        x = (float(x))
        y = (float(y))
        w = (float(w))
        h = (float(h))
        
        gt_class[count] = class_id
        gt_box[count, :] = [x, y, w, h]

        FN_TP[idx, class_id] += 1

        count += 1

        sum[class_id] += 1
    
    for li2 in range(length2):
        params = lines2[li2].split()
        class_id, x, y, w, h = params
        class_id = int(class_id)
        x = (float(x))
        y = (float(y))
        w = (float(w))
        h = (float(h))
        
        currentBox = [x, y, w, h]

        cnt = False
        for li1 in range(length1):
            if class_id == gt_class[li1]:
                iou = calculate_iou(currentBox, gt_box[li1])
                if iou > threshold:
                    cnt = True
        if cnt:
            TP[idx, class_id] += 1
        else:
            FP[idx, class_id] += 1

    file1.close()
    file2.close()

sum_TP = np.sum(TP, axis=0)
sum_FP = np.sum(FP, axis=0)
sum_FN_TP = np.sum(FN_TP, axis=0)
sum_TP = np.maximum(sum_TP, 1e-8)
sum_FP = np.maximum(sum_FP, 1e-8)
sum_FN_TP = np.maximum(sum_FN_TP, 1e-8)                  
print(sum)
print(sum_FN_TP)
print(sum_TP)
print(sum_FP)

precision = sum_TP / (sum_TP + sum_FP)
recall = sum_TP / sum_FN_TP
print("precision:")
print(precision)
print("recall:")
print(recall)

for c_id in range(3):
    F1score = 2*precision[c_id]*recall[c_id]/np.maximum(precision[c_id]+recall[c_id],1e-8)
    print("F1 score, class-{}:".format(c_id))
    print(F1score)


