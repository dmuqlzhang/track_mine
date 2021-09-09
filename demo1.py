import argparse
from AB_detector.detector import ABDetector



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/home/austin/docker_project/model_example/WineDetectionModel/images/vis.avi", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--model', type=str, default='/home/austin/docker_project/yolov5_0924/runs/exp152/best/weights/best.pt', help='source', choices=['yolov5x', 'paddle_676'])
    parser.add_argument('--device', type=str, default='1', help='device')
    parser.add_argument('--output', type=str, default='vis_10s.mp4', help='source')
    args = parser.parse_args()
    detector = ABDetector(args.device)
    detector.process(args)
