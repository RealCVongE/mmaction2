# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import time
from collections import defaultdict, deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from ultralytics import YOLO

from mmaction.apis import init_recognizer
from mmaction.utils import get_str_type
PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]
def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (g)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))
def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    while True:
        msg = 'Waiting for action ...'
        _, frame = camera.read()
        
        track_history = defaultdict(lambda: ())
        boxes=[]
        r = yolo.track(frame ,persist=True)[0]
        if r.boxes.id!=None:
            track_ids = r.boxes.id.int().cpu().tolist()
            for box,keypoint, track_id in zip(r.boxes.xyxy, r.keypoints, track_ids):
                track = track_history[track_id]
                track=(keypoint,box)
            frame_queue.append(copy.deepcopy(track_history))
        else:
            frame_queue.append(copy.deepcopy(defaultdict(lambda: [])))
        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            max_num=5
            # add spatio-temporal action detection results
            for track_id ,ann in results.items():
                ann.sort(key=lambda x: -x[2])
                results.append(
                (ann[0][0].cpu().numpy().astype(np.int64), [x[1] for x in ann], [x[2]
                                                        for x in ann]))
                for  k in results:
                    box = k[0]
                    label = k[1]
                    if not len(label):
                        continue
                    score = k[2]
                    st, ed = tuple(box[:2]), tuple(box[2:])

                    for k, lb in enumerate(label):
                        if k >= max_num:
                            break
                        text = abbrev(lb)
                        text = ': '.join([text, f'{score[k]:.3f}'])
                        location = (0 + st[0], 18 + k * 18 + st[1])
                        textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                                THICKNESS)[0]
                        textwidth = textsize[0]
                        diag0 = (location[0] + textwidth, location[1] - 14)
                        diag1 = (location[0], location[1] + 2)
                        
                        text_info[location] = text
                        # cv2.rectangle(frame, diag0, diag1, PLATEBLUE[k + 1], -1)
                        cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                    FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            camera.release()
            cv2.destroyAllWindows()
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = copy.deepcopy(frame_queue)
        for  i in cur_windows:
            for track_id, item in i.items():
                (keypoint,boxe)=item
                
        for track_id, item in new_track_history.items():
            for  k in item:
                (keypoints,boxes) =k
                fake_anno = dict(
                    frame_dict='',
                    label=-1,
                    img_shape=(self.h, self.w),
                    origin_shape=(self.h, self.w),
                    start_index=0,
                    modality='Pose',
                    total_frames=len(task.frames))
                num_person = 1

                num_keypoint = 17
                keypoint = np.zeros(
                    (num_person, NUM_FRAME, num_keypoint, 2))  # M T V 2
                keypoint_score = np.zeros(
                    (num_person, NUM_FRAME, num_keypoint)) 
                for  j, pose in enumerate(keypoints):
                    keypoint_temp= np.squeeze( pose.xy.cpu().numpy())
                    keypoint_score_temp=np.squeeze( pose.conf.cpu().numpy())
                    keypoint[0, j] = keypoint_temp
                    keypoint_score[0, j] = keypoint_score_temp
                fake_anno['keypoint'] = keypoint
                fake_anno['keypoint_score'] = keypoint_score

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = pseudo_collate([cur_data])

        # Forward the model
        with torch.no_grad():
            result = model.test_step(cur_data)[0]
        scores = result.pred_score.tolist()
        scores = np.array(scores)
        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            score_tuples = tuple(zip(label, scores_avg))
            score_sorted = sorted(
                score_tuples, key=itemgetter(1), reverse=True)
            results = score_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

            if inference_fps > 0:
                # add a limiter for actual inference fps <= inference_fps
                sleep_time = 1 / inference_fps - (time.time() - cur_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                cur_time = time.time()


def main():
    global average_size, threshold, drawing_fps, inference_fps, \
        device, model, camera, data, label, sample_length, \
        test_pipeline, frame_queue, result_queue ,yolo ,img_shape
    img_shape=None
    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps

    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    yolo =YOLO('checkpoints/yolov8x-pose-p6.pt')
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    camera = cv2.VideoCapture(args.camera_id)
    _, frame = camera.read()
    data = dict(frame_dict='',
                            label=-1,
                            img_shape=frame.shape[:2],
                            origin_shape=frame.shape[:2],
                            start_index=0,
                            modality='Pose',
                            total_frames=len(frame_queue))

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.test_pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in get_str_type(step['type']):
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if get_str_type(step['type']) in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
