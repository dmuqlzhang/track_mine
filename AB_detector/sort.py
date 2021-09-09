"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import cv2
import time
import argparse
from filterpy.kalman import KalmanFilter  # 卡尔曼滤波
from AB_detector.blur_detector import BlurDetector


@jit
def iou(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
                bb_gt[3] - bb_gt[1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])  # w*h w/h  但是self.kf.x 维度是dim_x=7，那么7维度分别对应：
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] +
                         w / 2., x[1] + h / 2.]).reshape((1, 4))  # [1,4]
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] +
                         w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, class_name, ori_img, frame_id):
        """
        Initialises a tracker using initial bounding box.
        bbox：x1, y1, x2, y2, conf, label-class_name = bbox[0:6]
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  #
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [
            0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [
            0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 将 float >1，从x1y1x2y2 转成xysr；0-4是xysr坐标，4-6是？？？
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count  # 卡尔曼滤波器id
        KalmanBoxTracker.count += 1  # 1个tracker 1个滤波器
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.class_name = class_name
        x1, y1, x2, y2 = bbox[0:4]
        x1 = [0 if x1 < 0 else x1][0]
        y1 = [0 if y1 < 0 else y1][0]
        self.best_shot = ori_img[int(y1):int(y2), int(x1):int(x2)]
        self.confidence = bbox[4]
        self.frame_id = frame_id
        bd = BlurDetector()
        self.blur_score = bd.get_blurness(self.best_shot)
        snap_gray = cv2.cvtColor(self.best_shot, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(snap_gray, bins=16)
        self.highlight_ratio = hist[15] * 1.0 / ((y2 - y1) * (x2 - x1))  # 高于某个值的像素数量 必须大于某个比例
        self.label = bbox[5]  # 非机动车 机动车 人，大类
        self.padding = 0

    # 这里较原版的多了update_shot，并且多了frame_id，主要是为了从后面的操作
    def update_shot(self, bbox, ori_img, frame_id):
        # bbox float 不是归一化的数值
        x1, y1, x2, y2 = bbox[0:4]
        confidence = bbox[4]
        x1 = [0 if x1 < 0 else x1][0]
        y1 = [0 if y1 < 0 else y1][0]
        shot = ori_img[int(y1):int(y2), int(x1):int(x2)]  # 由此可知，bbox数值是0-height 0-width
        snap_gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(snap_gray, bins=16)
        highlight_ratio = hist[15] * 1.0 / ((y2 - y1) * (x2 - x1))
        if highlight_ratio < 0.15:
            bd = BlurDetector()  # 计算模糊度
            blur_score = bd.get_blurness(shot)
            if confidence - blur_score > self.confidence - self.blur_score:
                self.confidence = confidence
                self.blur_score = blur_score
                self.best_shot = shot  # bset shot最好时刻的截图，在满足条件时继续更新信息，包含置信度-模糊度-其所在frame_id-抠图
                self.frame_id = frame_id

    def update(self, bbox, ori_img, frame_id):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []  # 这里暂时不清楚用途 ###############
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.update_shot(bbox, ori_img, frame_id)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        预测就是返回边界框估计
        [x1,y1,x2,y2]
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]  # 只返回最新的最后1个，就是预测估计坐标

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


# 核心 这里是IoU_match 将detections 和 tracks 匹配
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    detections: [x1, y1, x2, y2, cf, cl]
    trackers: trk[:] = [pos[0], pos[1], pos[2], pos[3], cls, 0]  pos->predict

    Returns 3 lists of
        matches, unmatched_detections and unmatched_trackers
    """
    # 初始化时，没有tracker，将所有检测框分配1个id
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)  # M-detections * N-trackers

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            # 只有大类相同时 才进行判断当前测量框和卡尔曼预测框的iou
            if det[5] == trk[4]:
                iou_matrix[d, t] = iou(det, trk)
    # 匈牙利算法，核心算法################
    matched_indices = linear_assignment(-iou_matrix)  # 匈牙利算法(指派)求的是值最小，但是我们要值最大，因此取相反数

    # matched_indices = np.squeeze(np.dstack(matched_indices))
    matched_indices = np.stack(matched_indices, axis=1)  # [N,2] 每行是匹配结果 0列代表detections，1列代表trackers
    # print(matched_indices.shape)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        # 小于阈值的也要筛选掉，将其按照各自的划分到对应区域
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))  # 保证N,2形状
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=70, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age  # 跟丢 最大帧数
        self.min_hits = min_hits  # 命中次数，抽帧需要减少【也可能增大】，连续帧可不变，
        self.trackers = []
        self.frame_count = 0

    def update(self, results, best_shot,  abandon_dict):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        object_dict = abandon_dict
        frame_id = results['frame_id']
        ori_img = results['image']
        bbox_xyxyc = []
        class_names = []  # 一般为 机动车 非机动车 人
        detect_id = 0  # 每帧的detect_id都重新从0开始
        for key in results.keys():
            if key == 'image' or key == 'frame_id':
                continue
            for target in results[
                key]:  # 按照大类进行划分框，进行匹配，不是小类！！！原始的sort并未考虑这个，主要是因为其主要为person进行跟踪，所以这里改进一下！但是每类的框都在1个[]里面，但有cl进行标记
                class_names.append(key)
                cf = target['conf']
                cl = object_dict[key]  # 机动车为1
                x1, y1, x2, y2 = target['axis']
                xyxyc = np.array([x1, y1, x2, y2, cf, cl])  # 坐标4个+置信度+大类别  [N,6]
                bbox_xyxyc.append(xyxyc)
                target['detect_id'] = detect_id  # 为每个大类(此处为机动车)内每个对象赋予id，后面用来和track_id匹配
                detect_id += 1
        bbox_xyxyc = np.array(bbox_xyxyc)
        # 检测框按概率从大到小排序
        # indices = np.argsort(-bbox_xyxyc[:, 4])
        # class_names = np.array(class_names)[indices]
        # bbox_xyxyc = bbox_xyxyc[indices]
        if len(bbox_xyxyc) == 0:
            return results, best_shot
        #################################################################################
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))  # tracker不会无限增加，因为后面会对象消失; trackers里面包含多个卡尔曼滤波器，难道是1个对象1个滤波器？
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # 预测后的估计坐标[x1,y1,x2,y2]  预测后的坐标会有无效值么？
            cls = self.trackers[t].label  # 大类标签0-1-2 非机动车 机动车 人
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cls, 0]  # 这是卡尔曼滤波预测的
            '''
            1. 卡尔曼滤波预测的结果有NaN？？？  这里不是算法关键，后面的pop才是核心
            2. 卡尔曼滤波的predict get_state 有什么区别？？？
            3. 一些参数不懂 time_since_update hit_streak【命中次数】
            '''
            if np.any(np.isnan(pos)):  # pos有nan？？？？？？？？
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 去掉行上有无效值的行，只保留有效值的行;即使全部是空值，那么依旧是[0,6]的shape
        for t in reversed(to_del):
            self.trackers.pop(t)
        # 一开始，trks肯定为0，所有的框均被分为unmatched_dets，然后其被当成新的tracker
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(bbox_xyxyc,
                                                                                   trks)  # detect_id顺序增加 和 bbox_xyxyc顺序np.arange是一样的，因此matched的0列是detect_id的部分

        if len(matched) > 0:
            for key in results.keys():
                if key == 'image' or key == 'frame_id':
                    continue
                for target in results[key]:  # 每类的track_id都对应各自大类的
                    detect_id = target['detect_id']
                    track_id = matched[np.where(matched[:, 0] == detect_id)[0], 1]  # 匹配上的tracke_id，逐个for遍历，因此只需要1个
                    if len(track_id):
                        # 每次输入结果，track_id detect_id都是-1，每帧的结果都需要这2个参数，需要将这2个传播出去的；但是注意这里传出的track_id是滤波器的id，从0开始，并不一定和track_id是相同的吧？？？
                        target['track_id'] = self.trackers[int(
                            track_id)].id  # 滤波器id，1个track1个id，每次的track_id(卡尔曼滤波器的id)都是传出去每帧对应的结果；卡尔曼滤波器的id个数是不断增长的，直到超出int范围\
                        # 然后从0开始；匹配时track_id 的索引每次都是从0开始，每帧有几个正在跟踪的和新建的，都并不是实际的id，这么做便于操作
                        target['is_matched'] = True
                        #################################################################################
        # update matched trackers with assigned detections
        # 首帧这里 和 前面 都不会 进行计算，直接将首帧 进行初始化unmatched detections，输出是匹配的+未匹配的检测框(用于初始化)+跟丢的预测框
        # 但是首帧 是未匹配的 检测框
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:  # 说明已有的tracker还在跟，没有消失，这里就能匹配上
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(bbox_xyxyc[d, :][0], ori_img, frame_id)  # 更新卡尔曼滤波核 matched tracks
            else:  # t in unmatched_trks
                pos = trk.predict()[0]
                if pos[0] > results['image'].shape[1] or pos[1] > results['image'].shape[0] or pos[2] < 0 or pos[
                    3] < 0 or trk.padding > 3:
                    continue
                pos[0] = max(pos[0], 0)
                pos[1] = max(pos[1], 0)
                pos[2] = min(pos[2], results['image'].shape[1])
                pos[3] = min(pos[3], results['image'].shape[0])
                # 这里 可能是 想用未匹配上的状态预测框 弥补 检测框？？？ 先采用key过滤，那么需要在detector.py里面初始化 is_matched
                result = {'axis': [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])],
                          'attribute': {},
                          'conf': trk.confidence,
                          'detect_id': -1,  # 这里用于跟踪使用，跟丢时就重置
                          'track_id': trk.id,
                          'is_deleted': False,
                          'is_matched': False}
                trk.padding += 1
                results[trk.class_name].append(result)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(bbox_xyxyc[i, :], class_names[i], ori_img, frame_id)  # 每个卡尔曼滤波器 都要有对应大类别名字，用来标记
            self.trackers.append(trk)  # 将matched unmatched_detections跟踪器合并
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # 这里并不懂！
            d = trk.get_state()[0]  # 返回当前状态,它与predict的区别？？1个是当前状态，另1个是下次预测的状态？[xysr]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet  跟踪整个过程到结束，返回需要的信息
            if trk.time_since_update > self.max_age:
                target = {'frame_id': trk.frame_id,
                          'track_id': trk.id,
                          'best_shot': trk.best_shot,
                          'confidence': trk.confidence,
                          'class_name': trk.class_name,
                          'attribute': {}}
                best_shot.append(target)
                self.trackers.pop(i)
        if len(ret) > 0:
            return results, best_shot
        return None, best_shot


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument(
        '--display',
        dest='display',
        help='Display online tracker output (slow) [False]',
        action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    sequences = [
        'PETS09-S2L1',
        'TUD-Campus',
        'TUD-Stadtmitte',
        'ETH-Bahnhof',
        'ETH-Sunnyday',
        'ETH-Pedcross2',
        'KITTI-13',
        'KITTI-17',
        'ADL-Rundle-6',
        'ADL-Rundle-8',
        'Venice-2']
    args = parse_args()
    display = args.display
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if display:
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n'
                  '(https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n'
                  '$ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()

    if not os.path.exists('output'):
        os.makedirs('output')

    for seq in sequences:
        mot_tracker = Sort()  # create instance of the SORT tracker
        seq_dets = np.loadtxt(
            'data/%s/det.txt' %
            seq, delimiter=',')  # load detections
        with open('output/%s.txt' % seq, 'w') as out_file:
            print("Processing %s." % seq)
            for frame in range(int(seq_dets[:, 0].max())):
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
                # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if display:
                    ax1 = fig.add_subplot(111, aspect='equal')
                    fn = 'mot_benchmark/%s/%s/img1/%06d.jpg' % (
                        phase, seq, frame)
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                        (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle(
                            (d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                        ax1.set_adjustable('box-forced')

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" %
          (total_time, total_frames, total_frames / total_time))
    if display:
        print("Note: to get real runtime results run without the option: --display")
