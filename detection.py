import cv2 as cv
from ultralytics import YOLO
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="stream")
    parser.add_argument("--vid_path", type=str)
    parser.add_argument("--cam_no", type=int, default=0)
    parser.add_argument("--show", type=bool, default=True)
    parser.add_argument("--save", type=bool, default=False)
    return parser.parse_args()

def predictImage(img, model, thresh):
    scale = 640/img.shape[1]
    img = cv.resize(img, None, fx=scale, fy=scale)

    results = model(img)

    def drawBox(results, img, thresh):
        for i in results:
            conf = i.boxes.conf.cpu().tolist()
            xyxy = i.boxes.xyxy.cpu().tolist()
            num = len(conf)
            for j in range(num):
                if conf[j] > thresh:
                    [x1, y1, x2, y2] = [int(k) for k in xyxy[j]]
                    cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv.putText(img, "ruoc "+str(round(conf[j], 2)), (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    drawBox(results, img, thresh)
    return img

def predictVideo(model, vid_path, thresh, show=True, save=False):
    vid = cv.VideoCapture(vid_path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    save_dir = "result.mp4"
    out = cv.VideoWriter(save_dir, fourcc, 20.0, (640,480))
    while True:
        ret, frame = vid.read()
        if frame is None:
            break
        frame = predictImage(frame, model, thresh)
        if show:
            cv.imshow('A', frame)
        if save:
            out.write(cv.resize(frame, (640, 480)))
        if cv.waitKey(20) & 0xff == ord('d'):
            break

    vid.release()
    out.release()
    cv.destroyAllWindows()

def predictStream(model, cam_no, thresh, show=True):
    vid = cv.VideoCapture(cam_no)
    save_dir = "result.mp4"
    while True:
        ret, frame = vid.read()
        if frame is None:
            break
        frame = predictImage(frame, model, thresh)
        if show:
            cv.imshow('A', frame)
        if cv.waitKey(20) & 0xff == ord('d'):
            break

    vid.release()
    cv.destroyAllWindows()

def main():
    args = get_args()
    model = YOLO('3d_print_70_epoch_best.pt')
    mode = args["--mode"]
    vid_path = args["--vid_path"]
    cam_no = args["--cam_no"]
    show = args["--show"]
    save = args["save"]
    if(mode == "stream"):
        predictStream(model, cam_no, 0.7, show)
    elif(mode == "video"):
        predictVideo(model, vid_path, 0.7, show, save)
    else:
        raise NotImplemented

# model = YOLO('3dprint.pt')
# img = cv.imread('benchy_fail.png')
# img = predictImage(img, model, 0.55)

# cv.imshow('A', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

if __name__ == "__main__":
    main()