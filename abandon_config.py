
class DefaultConfigs(object):
    has_GUI = False
    # 当has_GUI=False时，就需要将roi区域读取进来
    save_as_video = True # 画出roi区域框 blue
    abandon_list = [0,1,2]#背包 鼠标
    label_list = ['nornmal close', 'open', 'Half-Occlusion']
    #
    #"背包"，"雨伞"，"手提包"，棒球手套, “瓶”,“杯”,“香蕉”,苹果,“55蛋糕”,笔记本电脑，鼠标，遥控器，键盘，67手机,“书”，“钟”，“花瓶”，“剪刀”,“79牙刷”
    #


    # weights = '/home/austin/docker_project/yolov5_0924/runs/exp153/weights/best.pt'
    weights = "/home-85/austin/project/yolov5/runs/train/jinsha_hw_v8_m/weights/best.pt"
    iou_history_frame_threshold = 0.85#与物体历史帧的iou阈值，判断是否移动
    skip_frames = 1 #跳帧检测 每帧是10秒，60帧就是600s
    start_frame = 0
    end_frame = 2000



configs = DefaultConfigs()

