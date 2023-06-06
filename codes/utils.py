import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes
    

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, img_name):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]

    height, width, channels = image.shape
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)

                #image1: draw ground truth bounding boxes on image1
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = ann_box[i, 0]
                dy = ann_box[i, 1]
                dw = ann_box[i, 2]
                dh = ann_box[i, 3]
                gx = pw * dx + px
                gy = ph * dy + py
                gw = pw * np.exp(dw)
                gh = ph * np.exp(dh)
                x1 = int((gx - gw/2) * width)
                y1 = int((gy - gh/2) * height)
                x2 = int((gx + gw/2) * width)
                y2 = int((gy + gh/2) * height)
                start_point = (x1, y1)
                end_point = (x2, y2)
                color = colors[j]
                image1 = cv2.rectangle(image1, start_point, end_point, color, thickness=2)

                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                x1 = int(boxs_default[i, 4] * width)
                y1 = int(boxs_default[i, 5] * height)
                x2 = int(boxs_default[i, 6] * width)
                y2 = int(boxs_default[i, 7] * height)
                start_point = (x1, y1)
                end_point = (x2, y2)
                # color = colors[j]
                image2 = cv2.rectangle(image2, start_point, end_point, color, thickness=2)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                
                #image3: draw network-predicted bounding boxes on image3
                px = boxs_default[i, 0]
                py = boxs_default[i, 1]
                pw = boxs_default[i, 2]
                ph = boxs_default[i, 3]
                dx = pred_box[i, 0]
                dy = pred_box[i, 1]
                dw = pred_box[i, 2]
                dh = pred_box[i, 3]
                pred_x = pw * dx + px
                pred_y = ph * dy + py
                pred_w = pw * np.exp(dw)
                pred_h = ph * np.exp(dh)
                x1 = int((pred_x - pred_w/2) * width)
                y1 = int((pred_y - pred_h/2) * height)
                x2 = int((pred_x + pred_w/2) * width)
                y2 = int((pred_y + pred_h/2) * height)
                start_point = (x1, y1)
                end_point = (x2, y2)
                color = colors[j]
                image3 = cv2.rectangle(image3, start_point, end_point, color, thickness=2)

                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                x1 = int(boxs_default[i, 4] * width)
                y1 = int(boxs_default[i, 5] * height)
                x2 = int(boxs_default[i, 6] * width)
                y2 = int(boxs_default[i, 7] * height)
                start_point = (x1, y1)
                end_point = (x2, y2)
                # color = colors[j]
                image4 = cv2.rectangle(image4, start_point, end_point, color, thickness=2)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)

    # if windowname == "test":
    #     path = "output_images/before_nms/" + img_name[0] + ".jpg"
    #     cv2.imwrite(path, image)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.

def getBox(pred_box, boxes_default):
    box4 = np.zeros_like(pred_box)
    box8 = np.zeros_like(boxes_default)

    dx = pred_box[:, 0]
    dy = pred_box[:, 1]
    dw = pred_box[:, 2]
    dh = pred_box[:, 3]
    px = boxes_default[:, 0]
    py = boxes_default[:, 1]
    pw = boxes_default[:, 2]
    ph = boxes_default[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)
    box4[:, 0] = gx
    box4[:, 1] = gy
    box4[:, 2] = gw
    box4[:, 3] = gh
    box8[:, 4] = gx - gw/2
    box8[:, 5] = gy - gh/2
    box8[:, 6] = gx + gw/2
    box8[:, 7] = gy + gh/2
    
    return box4, box8


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.1, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    
    #TODO: non maximum suppression
    result_box = np.zeros_like(box_)
    result_confidence = np.zeros_like(confidence_)
    # before_box = box_
    # before_confidence = confidence_
    # pred_box4_pre, pred_box8_pre = getBox(box_, boxs_default)
    for class_num in range(0, 3):
        cur_class = confidence_[:, class_num]
        max_idx = np.argmax(cur_class)
        max_prob = cur_class[max_idx]
        max_box = box_[max_idx]

        while max_prob > threshold:
            # cur_class[max_idx] = 0
            result_box[max_idx, :] = max_box
            result_confidence[max_idx, class_num] = confidence_[max_idx][class_num]
            confidence_[max_idx, class_num] = 0
            box_[max_idx, :] = 0

            pred_box4_pre, pred_box8_pre = getBox(box_, boxs_default)
            # pred_box4_aft, pred_box8_aft = getBox(np.array([max_box]), boxs_default)
            

            dx = max_box[0]
            dy = max_box[1]
            dw = max_box[2]
            dh = max_box[3]
            px = boxs_default[max_idx, 0]
            py = boxs_default[max_idx, 1]
            pw = boxs_default[max_idx, 2]
            ph = boxs_default[max_idx, 3]
            gx = pw * dx + px
            gy = ph * dy + py
            gw = pw * np.exp(dw)
            gh = ph * np.exp(dh)

            x_min = gx - gw / 2
            y_min = gy - gh / 2
            x_max = gx + gw / 2
            y_max = gy + gh / 2

            ious = iou(pred_box8_pre, x_min, y_min, x_max, y_max)

            cur_class[max_idx] = 0
            del_idx = np.where(ious > overlap)[0]
            confidence_[del_idx, :] = 0
            box_[del_idx, :] = 0
            max_idx = np.argmax(cur_class)
            max_prob = cur_class[max_idx]
            max_box = box_[max_idx]
    
    return result_confidence, result_box

def intersection_over_union(pred_conf, pred, ann_conf, ann, thres, boxs_default, class_id):
    def calculate_iou(pred_box, ann_box):
        pred_w = pred_box[2]
        pred_h = pred_box[3]
        pred_xmin = pred_box[0] - pred_w / 2
        pred_ymin = pred_box[1] - pred_h / 2
        predBox = [pred_xmin, pred_ymin, pred_w, pred_h]

        ann_w = ann_box[2]
        ann_h = ann_box[3]
        ann_xmin = ann_box[0] - ann_w / 2
        ann_ymin = ann_box[1] - ann_h / 2
        annBox = [ann_xmin, ann_ymin, ann_w, ann_h]

        inter_box_top_left = [max(annBox[0], predBox[0]), max(annBox[1], predBox[1])]
        inter_box_bottom_right = [min(annBox[0]+annBox[2], predBox[0]+predBox[2]), min(annBox[1]+annBox[3], predBox[1]+predBox[3])]
        inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
        inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]
        intersection = inter_box_w * inter_box_h
        union = annBox[2] * annBox[3] + predBox[2] * predBox[3] - intersection
    
        iou = intersection / union

        return iou


    pred_box4, pred_box8 = getBox(pred, boxs_default)
    ann_box4, ann_box8 = getBox(ann, boxs_default)
    idx_ann = np.where(ann_conf[:, class_id] == 1)[0]
    ann_box4 = ann_box4[idx_ann]
    idx_pred = np.where(pred_conf[:, class_id] > thres)[0]
    pred_box4 = pred_box4[idx_pred]


    pred_box4 = np.unique(pred_box4, axis=0)
    # pred_box8 = np.unique(pred_box8, axis=0)
    ann_box4 = np.unique(ann_box4, axis=0)
    # ann_box8 = np.unique(ann_box8, axis=0)

    TP = 0
    FP = 0
    for idx1 in range(len(pred_box4)):
        cnt = False
        for idx2 in range(len(ann_box4)):
            if pred_box4[idx1].any() and ann_box4[idx2].any():
                iou = calculate_iou(pred_box4[idx1], ann_box4[idx2])
                if iou > thres:
                    cnt = True
        if cnt:
            TP += 1
        else:
            FP += 1
    return TP, FP

#implement a function to accumulate precision and recall to compute mAP or F1.
def update_precision_recall(_pred_confidence, _pred_box, _ann_confidence, _ann_box, _boxs_default,TP, FP, FN_TP, class_stat, thres=0.5):
    # TP = np.zeros([precision.shape[0], 3])
    # FP = np.zeros([precision.shape[0], 3])
    # FN = np.zeros([precision.shape[0], 3])
    for idx in range(len(_pred_box)):
        pred_box = _pred_box[idx]
        ann_box = _ann_box[idx]
        ann_confidence = _ann_confidence[idx]
        pred_confidence = _pred_confidence[idx]
        boxs_default = _boxs_default
        # print(class_stat)
        # print(type(class_stat))
        for class_num in range(0, 3):
            each_class_id = np.where(ann_confidence[: ,class_num] == 1)[0]
            cur_ann_box = ann_box[each_class_id]
            cur_ann_box = np.unique(cur_ann_box, axis=0)
            # print(cur_ann_box)
            # for box in cur_ann_box:
            #     if box.any():
            #         FN_TP[idx][class_num] += 1
            FN_TP[idx][class_num] = class_stat[0][class_num]
            # print(FN_TP[idx][class_num])
            TP[idx][class_num], FP[idx][class_num] = intersection_over_union(pred_confidence, pred_box, ann_confidence, ann_box, thres, boxs_default, class_num)
            # print(TP[idx][class_num], FP[idx][class_num])
    return TP, FP, FN_TP


def output(output_path, img_name, boxs_default, pred_confidence, pred_box, ori_height, ori_width):
    # print(pred_box[0][0])
    px = boxs_default[:, 0]
    py = boxs_default[:, 1]
    pw = boxs_default[:, 2]
    ph = boxs_default[:, 3]
    dx = pred_box[:, 0]
    dy = pred_box[:, 1]
    dw = pred_box[:, 2]
    dh = pred_box[:, 3]
    pred_x = pw * dx + px
    pred_y = ph * dy + py
    pred_w = pw * np.exp(dw)
    pred_h = ph * np.exp(dh)
    # x1 = int((pred_x - pred_w/2) * width)
    # y1 = int((pred_y - pred_h/2) * height)
    # x2 = int((pred_x + pred_w/2) * width)
    # y2 = int((pred_y + pred_h/2) * height)

    # pred_pts, _ = getBox(pred_box, boxs_default)
    # w = pred_pts[:, 2]
    # h = pred_pts[:, 3]
    # x_min = pred_pts[:, 0] - w / 2
    # y_min = pred_pts[:, 1] - h / 2
    # print(w)
    ori_height = ori_height.numpy()
    ori_width = ori_width.numpy()
    x_min = (pred_x - pred_w/2) * ori_width
    y_min = (pred_y - pred_h/2) * ori_height
    w = pred_w * ori_width
    h = pred_h * ori_height

    lines = []

    for i in range(len(pred_confidence)):
        for j in range(3):
            if pred_confidence[i,j]>0.5:
                class_id = j
                output_x_min = x_min[i]
                output_y_min = y_min[i]
                output_w = w[i]
                output_h = h[i]

                line = [class_id, output_x_min, output_y_min, output_w, output_h]
                lines.append(line)
    
    lines = np.asarray(lines)
    path = output_path + img_name[0] + ".txt"
    if len(lines):
        fmt = '%d', '%1.2f', '%1.2f', '%1.2f', '%1.2f'
        np.savetxt(path, lines, fmt)
    else:
        np.savetxt(path, lines)
    
def compareTXT(_pred_confidence, _pred_box, _ann_confidence, _ann_box, _boxs_default,TP, FP, FN_TP, class_stat, thres=0.5):
    # TP = np.zeros([precision.shape[0], 3])
    # FP = np.zeros([precision.shape[0], 3])
    # FN = np.zeros([precision.shape[0], 3])
    for idx in range(len(_pred_box)):
        pred_box = _pred_box[idx]
        ann_box = _ann_box[idx]
        ann_confidence = _ann_confidence[idx]
        pred_confidence = _pred_confidence[idx]
        boxs_default = _boxs_default
        # print(class_stat)
        # print(type(class_stat))
        for class_num in range(0, 3):
            each_class_id = np.where(ann_confidence[: ,class_num] == 1)[0]
            cur_ann_box = ann_box[each_class_id]
            cur_ann_box = np.unique(cur_ann_box, axis=0)
            # print(cur_ann_box)
            # for box in cur_ann_box:
            #     if box.any():
            #         FN_TP[idx][class_num] += 1
            FN_TP[idx][class_num] = class_stat[0][class_num]
            # print(FN_TP[idx][class_num])
            TP[idx][class_num], FP[idx][class_num] = intersection_over_union(pred_confidence, pred_box, ann_confidence, ann_box, thres, boxs_default, class_num)
            print(TP[idx][class_num], FP[idx][class_num])
    return TP, FP, FN_TP






