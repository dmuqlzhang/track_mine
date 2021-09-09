import cv2
import torch
import torch.backends.cudnn as cudnn
from models.utils import LoadImages

from AB_detector.utils import get_roi_contours, compute_iou_shp, plot_one_box, compute_iou_match, count_overtime_trackid
from AB_detector.detection import Detector
from AB_detector.sort_my import Sort
from numpy import random

import numpy as np



import sys
#sys.path.insert(0, '../..')
from abandon_config import configs

class ABDetector:
    def __init__(self, device):
        self.detector = Detector(device)
        #self.manager = Manager()

        self.tracker = Sort()
        #self.tracker.set_manager(self.manager)

        self.detector.set_tracker(self.tracker)

    def process(self, opt):
        output_path = opt.output
        source = opt.source
        webcam = None

        imgsz = self.detector.imgsz
        model = self.detector.model
        device = self.detector.device
        half = self.detector.half


        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(1000000)]
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        vid_path, vid_writer = None, None
        #
        # 统计遗留物在roi内停留的时间，超出时间一直报警
        frame_counter = {}
        time_counter = {}#注意需要根据跳帧数去计算时间
        track_id_coordinates = {}
        track_id_class ={}
        track_id_iou_records = {}
        #
        # 用于记录 缓慢移动 or 停止的次数，用于判断


        for path, img, im0s, vid_cap in dataset:
            ##
            #print(webcam, path, img, im0s, names, dataset.frame)
            if dataset.frame <=  configs.start_frame or dataset.frame>=configs.end_frame:
                continue

            if dataset.frame % configs.skip_frames != 0:
                pass
            else:
                ##获得跟踪结果
                result_sort = self.detector.detect(webcam, path, img, im0s, names, dataset.frame)
                #
                print('video %g/%g  %s: ' % (dataset.frame + 1, dataset.nframes, path), end='\n')
                #
                #print(result_sort, dataset.frame)
                if result_sort is None:#跟踪为空，则跳过
                    continue
                else:
                    #合并遗留物列表的所有跟踪结果
                    detected_abandon = []
                    for i in configs.abandon_list:
                        detected_abandon.extend(result_sort[names[int(i)]])###获得所有遗留物列表的跟踪结果
                    #print(detected_abandon)
                    #对当前帧的所有跟踪结果进行处理，包括1判断是否进入禁止区域（比较与禁止区域的iou阈值） 2.是否处于静止状态（比较与上一帧的iou阈值）
                    unnormal_flag = False
                    for subs_abandon in detected_abandon:
                        #unnormal_flag = False#针对每一个ID，而不是每一帧
                        is_matched = subs_abandon['is_matched']
                        #print(is_matched, subs_abandon)
                        if not is_matched:
                            continue
                        coordinates_subs_abandon = subs_abandon['axis']#坐标
                        track_id_subs_abandon = subs_abandon['track_id']#track id
                        x1, y1, x2, y2 = coordinates_subs_abandon[0:4]  # float


                        if track_id_subs_abandon not in time_counter:
                            # 相减少1
                            time_counter[track_id_subs_abandon] = 1
                            track_id_iou_records[track_id_subs_abandon] = [0]
                            track_id_class[track_id_subs_abandon] = list(subs_abandon['attribute'])[0]
                        else:
                            time_counter[track_id_subs_abandon] += 1

                        ##比较与历史上一抽帧的iou
                        if track_id_subs_abandon not in track_id_coordinates:  # 首次出现的 不计算 iou匹配
                            # 首次出现的ID可能是首帧，也可能是跟丢的情形
                            # 对于跟丢的情况，当dataset.frame>1时，肯定是跟丢的情况，记录此时的情况，包括画图、以及计算此时与上一帧所有目标中最大的IOU的目标相关联
                            if dataset.frame > 2:
                                print('happen new id normal')
                                iou_change_id = 0
                                unnormal = {}
                                for last_frame_id, last_frame_coor in track_id_coordinates.items():
                                    iou_change_id_tmp = compute_iou_match((int(x1), int(y1), int(x2), int(y2)), last_frame_coor)
                                    if iou_change_id_tmp>iou_change_id:
                                        iou_change_id = iou_change_id_tmp
                                        unnormal['new_id'] = track_id_subs_abandon
                                        unnormal['new_coor'] = (int(x1), int(y1), int(x2), int(y2))
                                        unnormal['lost_id'] = last_frame_id
                                        unnormal['last_coor'] = last_frame_coor
                                        unnormal['match_iou'] = iou_change_id
                                if iou_change_id>0.1:#画出异常图
                                    plot_one_box(unnormal['new_coor'], im0s, color=(0, 255, 0),label_location='down',
                                                 label=f"unnormal id:{unnormal['new_id']} LM id {unnormal['lost_id']}",
                                                 line_thickness=3)
                                    plot_one_box(unnormal['last_coor'], im0s, color=(0, 0, 255),label_location='up',
                                                 label=f"unnormal LM id:{unnormal['lost_id']}",
                                                 line_thickness=3)
                                    cv2.imwrite(f'lost id {dataset.frame}_{configs.skip_frames}s.jpg', im0s)


                            #

                            #更新新的id
                            track_id_coordinates[track_id_subs_abandon] = (int(x1), int(y1), int(x2), int(y2))


                        else:
                            # 计算当前帧 和 前一帧的 iou_match 匹配
                            pre_coordinate = track_id_coordinates[track_id_subs_abandon]
                            track_id_coordinates[track_id_subs_abandon] = (int(x1), int(y1), int(x2), int(y2))

                            iou_match = compute_iou_match(pre_coordinate, (x1, y1, x2, y2))  # 这里是否也需要int？？

                            track_id_iou_records[track_id_subs_abandon].append(iou_match)
                            iou_match_2 = round(iou_match, 2)
                            #print(iou_match_2)
                            abandon_coordinates_i = (x1, y1, x2, y2)
                            if iou_match>configs.iou_history_frame_threshold:
                                #print(iou_match, list(subs_abandon['attribute'])[0], configs.label_list.index(list(subs_abandon['attribute'])[0]), track_id_subs_abandon)
                                plot_one_box(abandon_coordinates_i, im0s, color=colors[track_id_subs_abandon],
                                             label=f"{list(subs_abandon['attribute'])[0]} {iou_match_2} id:{track_id_subs_abandon}",
                                             line_thickness=3)
                                # plot_one_box(abandon_coordinates_i, im0s, color=colors[0],
                                #              label=f"type: {list(subs_abandon['attribute'])[0]} status:normal",
                                #              line_thickness=None)
                            else:
                                unnormal_flag = True
                                plot_one_box(abandon_coordinates_i, im0s, color=(0,0,255),label_location='down',
                                             label=f"{list(subs_abandon['attribute'])[0]} {iou_match_2} id:{track_id_subs_abandon} warning",
                                             line_thickness=3)
                        # if unnormal_flag:
                        #     import os
                        #     unnormal_snapshot_path = opt.output.split('.')[0]
                        #     os.makedirs(unnormal_snapshot_path, exist_ok=True)
                        #     cv2.imwrite(os.path.join(unnormal_snapshot_path, f'less iou: {iou_match} {dataset.frame}_{configs.skip_frames}s.jpg'), im0s)
                    if unnormal_flag:
                        import os
                        unnormal_snapshot_path = opt.output.split('.')[0]
                        os.makedirs(unnormal_snapshot_path, exist_ok=True)
                        iou_match = round(iou_match, 2)
                        cv2.imwrite(os.path.join(unnormal_snapshot_path, f'less iou: {configs.skip_frames} {dataset.frame}_{iou_match}s.jpg'), im0s)

                    #cv2.imshow("result", output)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                    if (not webcam) and (result_sort!=''):
                        if vid_path != output_path:
                            vid_path = output_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0s)
        #print(track_id_iou_records)
        for id, iou_list in track_id_iou_records.items():
            print(track_id_class[id], len([i for i in iou_list if i > configs.iou_history_frame_threshold]), '/', len(iou_list)-1)
        result_save ={}
        result_save['id_class'] = track_id_class
        result_save['id_iou'] = track_id_iou_records
        with open(f'iou_result_{opt.output.split("/")[-1].split(".")[0]}_{configs.skip_frames}s.txt','w') as f:
            f.write(str(result_save))
        f.close()