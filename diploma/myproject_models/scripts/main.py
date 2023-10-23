import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from torchvision.utils import draw_bounding_boxes, draw_keypoints
import torch.nn.functional as F

class Model:
  def nn_model(self):
    model =  torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    return model
  
class FitImage:
  def __init__(self, image, model):
    self.image = image
    self.model = model
    transform = torchvision.transforms.Compose([T.ToTensor()])
    self.model_image_tensor = transform(self.image)
    model.eval()
    self.out = self.model([self.model_image_tensor])
    self.out = self.out[0]
    self.connect_skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

  def out_bboxes(self):
    boxes = np.zeros((1, 4))
    for i in range(len(self.out['boxes'])):
      if self.out['scores'][i] > 0.9:
        boxes = np.vstack((boxes, self.out['boxes'][i].detach().numpy()))
    boxes_tensor = torch.tensor(np.delete(boxes, 0, 0))
    return boxes_tensor

  def out_key_points(self):
    key_points = np.zeros((1, 17, 2))
    for i in range(len(self.out['boxes'])):
      if self.out['scores'][i] > 0.9:
        key_points[i] = self.out['keypoints'][i][:,:2].detach().numpy()
    key_points_tensor = torch.tensor(key_points)
    return key_points_tensor

  def out_keypoints_scores(self):
    out_kepoints_scores = np.zeros((1, 17))
    for i in range(len(self.out['boxes'])):
      if self.out['scores'][i] > 0.9:
        out_kepoints_scores = np.vstack((out_kepoints_scores, self.out['keypoints_scores'][i].detach().numpy()))
    out_kepoints_scores_tensor = torch.tensor(np.delete(out_kepoints_scores, 0, 0))
    return out_kepoints_scores_tensor

  def draw_bboxes(self, out_bboxes):
    image_255 = self.model_image_tensor * 255
    image = draw_bounding_boxes(image_255.to(torch.uint8), out_bboxes, colors = tuple(i*255//out_bboxes.shape[0] for i in range(out_bboxes.shape[0])), width=5)
    return image.permute(1, 2, 0)

  def draw_keypoints(self, out_key_points):
    image_255 = self.model_image_tensor * 255
    image = draw_keypoints(image_255.to(torch.uint8), out_key_points, colors = tuple(i*255//out_key_points.shape[0] for i in range(out_key_points.shape[0])), radius=5)
    return image.permute(1, 2, 0)

  def draw_skeleton(self, out_key_points):
    image_255 = self.model_image_tensor * 255
    model_skeleton = draw_keypoints(image_255.to(torch.uint8), out_key_points, connectivity=self.connect_skeleton, \
                                    colors = tuple(i*255//out_key_points.shape[0] for i in range(out_key_points.shape[0])), radius=5, width=3)
    return model_skeleton.permute(1, 2, 0)

  def crop_image(self, bboxes):
    list_image_crop = []
    for bbox in bboxes:
      image_crop = T.functional.crop(
          self.model_image_tensor, int(bbox[1]), int(bbox[0]), int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])
          )
      list_image_crop.append(image_crop)
    return list_image_crop

  def crop_key_points(self, crop_image, out_bboxes, out_key_points):

    self.data = zip(out_key_points, out_bboxes, crop_image)

    for key_point, bbox, image in self.data:

      key_points_crop_x = key_point[:,0] - bbox[0]
      key_points_crop_y = key_point[:,1] - bbox[1]
      key_points_crop = torch.ones((1, 17, 3))
      key_points_crop[0, :, 0] = key_points_crop_x
      key_points_crop[0, :, 1] = key_points_crop_y

      h2 = image.shape[1] / 2
      w2 = image.shape[2] / 2

      key_points_crop = key_points_crop[0][:, :2]

      key_points_crop_center = torch.zeros(key_points_crop.shape)

      key_points_crop_center[:, 0] = key_points_crop[:,0] - h2
      key_points_crop_center[:, 1] = key_points_crop[:,1] - w2


    return key_points_crop_center

class ScoreFit:
  def __init__(self, image_true, image_train):
    self.image_true = image_true
    self.image_train = image_train

  def aff_transform(self):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
    transform_unpad = lambda x: unpad(np.dot(pad(x), A))
    Y = pad(self.image_true.detach().numpy())
    X = pad(self.image_train.detach().numpy())
    A, res, rank, s = np.linalg.lstsq(X, Y)
    A[np.abs(A) < 1e-10] = 0
    key_points_transform = transform_unpad(self.image_train.detach().numpy())
    return key_points_transform

  def cos_distance(self, image_trans):

    cos_x = F.cosine_similarity(torch.tensor(image_trans), self.image_true.detach(), dim=1)
    cos_y = F.cosine_similarity(torch.tensor(image_trans), self.image_true.detach(), dim=0)
    return cos_x, cos_y

  def weight_distance(self, train_conf):

    self.train_conf = train_conf.detach().numpy()

    sum1 = 1 / np.sum(self.train_conf)
    sum2 = 0

    for i in range(len(self.image_train)):
        
        sum2 += train_conf[i] * abs(self.image_train[i] - self.image_true[i])
        weighted_dist = sum1 * sum2

    return weighted_dist


if __name__ == '__main__':


    m = Model()
    model = m.nn_model()

    count = 1
    videoFile1 = "/content/output_video.avi"
    videoFile2 = "/content/lena1.mp4"
    cap1 = cv2.VideoCapture(videoFile1)   # загрузка видео
    cap2 = cv2.VideoCapture(videoFile2)
    frameRate1 = cap1.get(5) # частота кадров
    frameRate2 = cap2.get(5)

    frameRate = frameRate1 if frameRate1 <= frameRate2 else frameRate2

    while(cap1.isOpened() and cap2.isOpened()):
        frameId1 = cap1.get(1) # номер текущего кадра
        ret1, frame1 = cap1.read()

        frameId2 = cap2.get(1) # номер текущего кадра
        ret2, frame2 = cap2.read()

        if (ret1 != True) or (ret2 != True):
            break
        else:

            image1 = FitImage(frame1, model)
            out_bboxes1 = image1.out_bboxes()
            out_key_points1 = image1.out_key_points()
            ckpt1 = image1.crop_key_points(image1.crop_image(out_bboxes1), out_bboxes1, out_key_points1)

            image2 = FitImage(frame2, model)
            out_bboxes2 = image2.out_bboxes()
            out_key_points2 = image2.out_key_points()
            ckpt2 = image2.crop_key_points(image2.crop_image(out_bboxes2), out_bboxes2, out_key_points2)

            image_out1 = image1.draw_skeleton(out_key_points1)
            image_out2 = image2.draw_skeleton(out_key_points2)

            score_fit = ScoreFit(ckpt1, ckpt2)

            cos_x, cos_y = score_fit.cos_distance(score_fit.aff_transform())

            cos_res =(np.mean(cos_x.detach().numpy()) + np.mean(cos_y.detach().numpy())) / 2

            weight_distance = score_fit.weight_distance(image2.out_keypoints_scores()[0])
            weight_distance_res = np.mean(weight_distance.detach().numpy())

            image = np.zeros((1480, 1280, 3))

            image[:720, :, :] = image_out1.detach().numpy()
            image[720:1440, :, :] = image_out2.detach().numpy()


            text = f"Cosinus: {cos_res}    Distance: {weight_distance_res}"
            image = cv2.putText(image, text, (20,1470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



            filename ="video_out/frame%d.jpg" % count;count+=1
            cv2.imwrite(filename, image)

    cap1.release()
    cap2.release()

    frameSize = (1280, 720)
    out = cv2.VideoWriter('video_out.avi',cv2.VideoWriter_fourcc(*'DIVX'), frameRate, frameSize)
    for i in range(1, count):
        img = cv2.imread("video_out/frame%d.jpg" % i)
        resized = cv2.resize(img, frameSize, interpolation = cv2.INTER_AREA)
        out.write(resized)
    out.release()