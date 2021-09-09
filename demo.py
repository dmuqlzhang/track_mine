import argparse
from AB_detector.detector import ABDetector



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="/home-85/austin/project/mineDtection/images/cam3.avi", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--model', type=str, default="/home-85/austin/project/yolov5/runs/train/jinsha_hw_v8_m/weights/best.pt", help='source', choices=['yolov5x', 'paddle_676'])
    parser.add_argument('--device', type=str, default='2', help='device')
    parser.add_argument('--output', type=str, default='cam3_out.mp4', help='source')
    args = parser.parse_args()
    detector = ABDetector(args.device)
    detector.process(args)
