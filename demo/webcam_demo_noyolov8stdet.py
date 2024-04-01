# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import time
from collections import defaultdict, deque
from operator import itemgetter
from threading import Thread
import multiprocessing as mp
import mmengine

import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from ultralytics import YOLO
from mmdet.apis import inference_detector, init_detector
from mmpose.apis.inference import init_model ,inference_topdown
from mmpose.structures import PoseDataSample, merge_data_samples
from mmengine.structures import InstanceData
from mmengine import Config, DictAction

from mmaction.apis import init_recognizer
from mmaction.apis.inference import inference_recognizer
from mmaction.utils import get_str_type
PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (g)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (0, 0, 0)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]
NUM_KEYPOINT=17
NUM_FRAME=30
def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1

    square_l = max(width, height)
    new_width = new_height = square_l * ratio

    new_x1 = max(0, int(center_x - new_width / 2))
    new_x2 = min(int(center_x + new_width / 2), w)
    new_y1 = max(0, int(center_y - new_height / 2))
    new_y2 = min(int(center_y + new_height / 2), h)
    return (new_x1, new_y1, new_x2, new_y2)


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}
def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name
def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    intersect = w * h
    union = s1 + s2 - intersect
    iou = intersect / union

    return iou

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--label-map',
        default='tools/data/kinetics/label_map_k400.txt',
        help='label map file for action recognition')
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
        '--output-file',
        type=str,
        default=None,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')
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


def show_results(result_queue,result_queue_stdet):
    print('Press "Esc", "q" or "Q" to exit')
    # 비디오 저장을 위한 설정
    if args.output_file!=None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output_file, fourcc, 30.0, (640, 480))
        fps = camera.get(cv2.CAP_PROP_FPS)
        duration_in_seconds=300
        total_frames = fps * duration_in_seconds
    current_frame = 0
    text_info = {}
    bbox_info = None
    bbox_info2 = None
    cur_time = time.time()
    while True:
        msg = 'Waiting for action ...'
        
        _, frame = camera.read()
        results2=None
        frame_queue.append(np.array(frame))
        if len(pose_deque)!= 0:
            results = pose_deque[0]
            if(results.boxes.id!=None):
                if result_queue_stdet.qsize() != 0:
                    results2 = result_queue_stdet.get()
                    bbox_info = results2[0]
                    track_id2,skeleton_prediction,box  = results2[0]
                track_ids = results.boxes.id.int().cpu().tolist()
                boxes = results.boxes.xyxy.cpu().numpy().astype(np.int64)
                bbox_info2 =zip(boxes,track_ids)
                for box, track_id  in zip(boxes,track_ids):
                    (startX, startY, endX, endY) = box
                    # if score < threshold:
                    #     break
                    if results2!=None and (track_id2==track_id):
                        cv2.putText(frame, "label", (startX+1, startY), FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, str(track_id), (startX, startY), FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)
        elif bbox_info2!=None:
            (boxes,track_ids) = bbox_info2
            for box, track_id  in zip(boxes,track_ids):
                (startX, startY, endX, endY) = box
                # if score < threshold:
                #     break
                if(bbox_info!=None):
                    track_id2,skeleton_prediction,box  = bbox_info
                    if  (track_id2==track_id):
                        cv2.putText(frame, "label", (startX+1, startY), FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
                    
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, str(track_id), (startX, startY), FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        if result_queue.qsize() != 0:
            text_info = {}
            results = result_queue.get()
            for i, result in enumerate(results):
                selected_label, score = result
                # if score < threshold:
                #     break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score * 100, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)
        if out!=None:
            out.write(frame)
            current_frame += 1
            if(current_frame==total_frames):
                current_frame=0
        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            camera.release()
            out.release()
            cv2.destroyAllWindows()
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()
def inference_pose(pose_queue,pose_queue_stdet):
    model = init_detector(config= "/home/bigdeal/mnt2/workspace/mmaction2/checkpoints/yolox_x_8x8_300e_coco.py", checkpoint="/home/bigdeal/mnt2/workspace/mmaction2/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth", device="cuda:0")
    model2 = init_model("checkpoints/yoloxpose_l_8xb32-300e_coco-640.py", "checkpoints/yoloxpose_l_8xb32-300e_coco-640-de0f8dee_20230829.pth", "cuda:0")
    
    pose_deque=deque(maxlen=30)
    bbox_deque=deque(maxlen=30)
    predict_step=0
    while True:
        if len(frame_queue) != 0:
            results = []
            data_samples = []
            for frame_path in list(frame_queue):
                det_data_sample= inference_detector(model, frame_path)
                pred_instance = det_data_sample.pred_instances.cpu().numpy()
                bboxes = pred_instance.bboxes
                scores = pred_instance.scores
                # We only keep human detection bboxs with score larger
                # than `det_score_thr` and category id equal to `det_cat_id`.
                valid_idx = np.logical_and(pred_instance.labels == 0,
                                            pred_instance.scores > 0.9)
                bboxes = bboxes[valid_idx]
                scores = scores[valid_idx]
                results.append(bboxes)
                data_samples.append(det_data_sample)
            bbox_deque.append(results[0])
            results2 = []
            data_samples2 = []
            for f, d in list(zip(list(frame_queue), results)):
                pose_data_samples = inference_topdown(model2, f, d[..., :4], bbox_format='xyxy')
                pose_data_sample = merge_data_samples(pose_data_samples)
                pose_data_sample.dataset_meta = model.dataset_meta
                # make fake pred_instances
                if not hasattr(pose_data_sample, 'pred_instances'):
                    num_keypoints = model.dataset_meta['num_keypoints']
                    pred_instances_data = dict(
                        keypoints=np.empty(shape=(0, num_keypoints, 2)),
                        keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                        bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                        bbox_scores=np.empty(shape=(0), dtype=np.float32))
                    pose_data_sample.pred_instances = InstanceData(
                        **pred_instances_data)

                poses = pose_data_sample.pred_instances.to_dict()
                results2.append(poses)
                data_samples2.append(pose_data_sample)
            pose_deque.append(results2[0])

        if len(pose_deque)==30 and predict_step==8:
            pose_queue_stdet.put((copy.deepcopy(pose_deque),copy.deepcopy(bbox_deque)))
            predict_step=0
        elif predict_step==8:
            predict_step=0
        predict_step+=1
def get_items_from_queue(queue):
    """Queue에서 num_items 개수만큼 아이템을 꺼내어 리스트로 반환"""
    keypoint=[]
    keypoint_score=[]
    r = queue.get()
    for item in r:
        keypoint.append( item.keypoints.xy.cpu().numpy())
        keypoint_score.append( item.keypoints.conf.cpu().numpy())
    return keypoint ,keypoint_score
def get_items_from_queue_stdet(queue):
    """Queue에서 num_items 개수만큼 아이템을 꺼내어 리스트로 반환"""
    track_history=defaultdict(lambda:[])
    item = queue.get()
    
    for r in item:
        if r.boxes.id!=None:
            track_ids = r.boxes.id.int().cpu().tolist()
            for box,keypoint, track_id in zip(r.boxes.xyxy, r.keypoints, track_ids):
                track = track_history[track_id]
                track.append((keypoint.xy.cpu().numpy(),keypoint.conf.cpu().numpy(),box)) 
                if len(track) > NUM_FRAME:  # retain 90 tracks for 90 frames
                    track.pop(0)

    return track_history
def inference(pose_queue,queue,result_queue):
    data = queue.get()  # 첫 번째 아이템 (dict)
    args = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    cfg = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    label_map = load_label_map(args["label_map_stdet"])
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config = mmengine.Config.fromfile(args["config"])
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset
    model = init_recognizer(skeleton_config, args["checkpoint"], device=args["device"])
    while True:
        keypoint_score= []
        while len(keypoint_score) == 0:
            if pose_queue.qsize() >= NUM_FRAME:
                keypoint, keypoint_score= get_items_from_queue(pose_queue)
        # frame_queue.append((np.array(r[0].keypoints.xy.cpu().numpy()),np.array(r[0].keypoints.conf.cpu().numpy())))
                num_person = max([len(x) for x in keypoint])
                combined_keypoint = np.zeros((NUM_FRAME, num_person, NUM_KEYPOINT, 2),
                        dtype=np.float16)
                combined_keypoint_score = np.zeros((NUM_FRAME, num_person, NUM_KEYPOINT),
                              dtype=np.float16)
                for f_idx, frm_pose in enumerate(zip(keypoint,keypoint_score)):
                    keypoint_m,keypoint_score_m = frm_pose 
                    frm_num_persons = keypoint_m.shape[0]
                    for p_idx in range(frm_num_persons):
                        combined_keypoint[f_idx, p_idx] = keypoint_m[p_idx]
                        combined_keypoint_score[f_idx, p_idx] = keypoint_score_m[p_idx]


        cur_data = data.copy()
        cur_data['keypoint'] = combined_keypoint.transpose((1, 0, 2, 3))
        cur_data['keypoint_score'] = combined_keypoint_score.transpose((1, 0, 2))

        output = inference_recognizer(model, cur_data)
        max_pred_index = output.pred_score.argmax().item()
        action_label = label_map[max_pred_index]
        action_score = output.pred_score[max_pred_index]
        my_list = []
        my_tuple= (action_label,action_score.cpu().numpy() )
        my_list.append(my_tuple)
        result_queue.put(copy.deepcopy(my_list))
def inference_stdet(pose_queue_stdet,queue,result_queue_stdet):
    data = queue.get()  # 첫 번째 아이템 (dict)
    args = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    cfg = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    label_map = load_label_map(args["label_map_stdet"])

    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config = mmengine.Config.fromfile(args["config"])
    skeleton_config.model.cls_head.num_classes = num_class # for K400 dataset
    model = init_recognizer(skeleton_config, args["checkpoint"], device=args["device"])
    while True:
        keypoint_score= []
        while len(keypoint_score) == 0:
            if pose_queue_stdet.qsize() >= NUM_FRAME:
                pose_results ,bbox_results = pose_queue_stdet.get()
        # frame_queue.append((np.array(r[0].keypoints.xy.cpu().numpy()),np.array(r[0].keypoints.conf.cpu().numpy())))
                
                combined_keypoint = np.zeros(( 1,NUM_FRAME, NUM_KEYPOINT, 2),
                        dtype=np.float16)
                combined_keypoint_score = np.zeros(( 1,NUM_FRAME, NUM_KEYPOINT),
                              dtype=np.float16)

                skeleton_prediction = []
                for i in range(bbox_results[0].shape[0]):  # num_person
                    skeleton_prediction.append([])


                    num_person = 1

                    num_keypoint = 17

                    # pose matching
                    person_bbox = bbox_results[0][i][:4]
                    area = expand_bbox(person_bbox, data["img_shape"][0], data["img_shape"][1])

                    for j, poses in enumerate(pose_results):  # num_frame
                        max_iou = float('-inf')
                        index = -1
                        if len(poses['keypoints']) == 0:
                            continue
                        for k, bbox in enumerate(poses['bboxes']):
                            iou = cal_iou(bbox, area)
                            if max_iou < iou:
                                index = k
                                max_iou = iou
                        combined_keypoint[0, j] = poses['keypoints'][index]
                        combined_keypoint_score[0, j] = poses['keypoint_scores'][index]

                    cur_data = data.copy()
                    cur_data['total_frames'] = combined_keypoint_score.shape[0]
                    cur_data['keypoint'] = combined_keypoint.transpose((1, 0, 2, 3))
                    cur_data['keypoint_score'] = combined_keypoint_score.transpose((1, 0, 2))

                    output = inference_recognizer(model, cur_data)
                    # for multi-label recognition
                    score = output.pred_score.tolist()
                    for k in range(len(score)):  # 81
                        if k not in label_map:
                            continue
                        if score[k] > args["action_score_thr"]:
                            skeleton_prediction[i].append((label_map[k], score[k]))

                    print(f" {skeleton_prediction}")
                    # my_tuple= (track_id,skeleton_prediction,box.cpu().numpy() )
                    # my_list.append(my_tuple)
                # result_queue_stdet.put(copy.deepcopy(my_list))


def main():
    global average_size, threshold, drawing_fps, inference_fps, \
        device, model, camera, data, label, sample_length, \
        args, frame_queue, result_queue ,img_shape ,pose_queue,out,pose_deque
    img_shape=None
    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps
    stdet_label_map = load_label_map(args.label_map_stdet)

    device = torch.device(args.device)
    out=None
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build the recognizer from a config file and checkpoint file/url
    camera = cv2.VideoCapture("demo/test_video_structuralize.mp4")
    w = round(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_length=30
    data = dict(frame_dict='',
                            label=-1,
                            img_shape=(h,w),
                            origin_shape=(h,w),
                            start_index=0,
                            modality='Pose',
                            total_frames=sample_length)

    with open(args.label_map, 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    # cfg = model.cfg
    # sample_length = 0
    # pipeline = cfg.test_pipeline
    # pipeline_ = pipeline.copy()
    # for step in pipeline:
    #     if 'SampleFrames' in get_str_type(step['type']):
    #         sample_length = step['clip_len'] * step['num_clips']
    #         data['num_clips'] = step['num_clips']
    #         data['clip_len'] = step['clip_len']
    #         pipeline_.remove(step)
    #     if get_str_type(step['type']) in EXCLUED_STEPS:
    #         # remove step to decode frames
    #         pipeline_.remove(step)
    # test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:
        mp.set_start_method('spawn', force=True)
        frame_queue = deque(maxlen=1)
        pose_deque=deque(maxlen=30)
        
        # pose_queue = deque(maxlen=sample_length)
        result_queue =  mp.Queue()
        result_queue_stdet =  mp.Queue()
        pose_queue = mp.Queue()
        pose_queue_stdet = mp.Queue()
        queue = mp.Queue()
        queue.put(data)  # dict 전달
        queue.put(vars(args))
        queue.put(cfg)
        queue2 = mp.Queue()
        queue2.put(data)  # dict 전달
        queue2.put(vars(args))
        queue2.put(cfg)
        pw = Thread(target=show_results, args=(result_queue,result_queue_stdet), daemon=True)
        ps = Thread(target=inference_pose, args=(pose_queue,pose_queue_stdet,), daemon=True)
        # pr = Thread(target=inference, args=(), daemon=True)
        # pr = mp.Process(target=inference, args=(pose_queue,queue,result_queue,))
        pr2 = mp.Process(target=inference_stdet, args=(pose_queue_stdet,queue2,result_queue_stdet,))
        pw.start()
        ps.start()
        # pr.start()
        pr2.start()
        pw.join()
        # pr.join()
        pr2.join()
        ps.join()
    except KeyboardInterrupt:
        if out!=None:
            out.release()
        pass


if __name__ == '__main__':
    main()
