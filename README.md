# Report: Assignment 3
## 1.Prediction of model on images from the test set
### 1.1 Results on image with one cat

![Alt](./before_nms/00217.jpg "1img")
![](./after_nms/00217.jpg)

### 1.2 Results on image with one dog  

![](./before_nms/00401.jpg)
![](./after_nms/00401.jpg)

### 1.3 Results on image with one person

![](./before_nms/00003.jpg)
![](./after_nms/00003.jpg)

### 1.4 Results on image with two persons

![](./before_nms/00110.jpg)
![](./after_nms/00110.jpg)

## 2.Hyperparameter settings
The hyperparameter settings are as follows. The learning rate is '1e-4', epoch number is '100', batch size is '32' and optimizer is 'Adam'.

![](./hyper1.png)
![](./hyper2.png)

## 3.Traning/Validation loss graphs
The below two pictures show different traning/validation loss graphs when applying softmax to confidence and not, respectively. The loss is pretty low without applying softmax, but the predicted results are not satisfying. The loss is relatively greater when applying softmax, but the predicted results are much better, which is why the submmitted results are the ones with softmax applied to confidence.

![](./loss_graph_1.png)
![](./loss_graph_2.png)

## 4. The F1 score calculated on the validation set (with precision and recall)
Separate.py is used to calculte precision, recall and F1 score on the dataset.
Although only the F1 score on validation set is required, the precision, recall and F1 scores on train set and whole dataset (including both train and validation set) are also shown as follows:

![](./val_F1.png)
![](./train_F1.png)
![](./train_val_F1.png)