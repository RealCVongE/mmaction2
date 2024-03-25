# Copyright (c) OpenMMLab. All rights reserved.
"""Webcam Spatio-Temporal Action Detection Demo.

Some codes are based on https://github.com/facebookresearch/SlowFast
"""
from collections import defaultdict
from mmengine.utils import track_iter_progress
from mmdet.structures import DetDataSample
from mmpose.structures import PoseDataSample, merge_data_samples
from typing import List, Optional, Tuple, Union
import mmengine
from ultralytics import YOLO
from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
import argparse
import atexit
import copy
import logging
import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from mmaction.registry import VISUALIZERS
import torch
import torch.nn as nn
import cv2
import mmcv
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.structures import InstanceData
from mmpose.apis import inference_topdown, init_model
from mmaction.structures import ActionDataSample

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1
NUM_FRAME=30
PREDICT_STEP=8
def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]
def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


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
def dense_timestamps(timestamps, n):
    """Make it nx frames."""
    if len(timestamps)==1:
            new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
    else:
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
    return new_frame_inds.astype(np.int64)

def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


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

def parse_args():

    parser = argparse.ArgumentParser(
        description='MMAction2 webcam spatio-temporal detection demo')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d'
        '/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--label-map-stdet',
        default='tools/data/ava/label_map.txt',
        help='label map file for spatio-temporal action detection')
    parser.add_argument(
        '--config',
        default=(
            'configs/detection/slowonly/'
            'slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'
        ),
        help='spatio temporal detection config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of human action score')
    parser.add_argument(
        '--det-config',
        default='checkpoints/yolox_x_8x8_300e_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='checkpoints/yoloxpose_l_8xb32-300e_coco-640.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('checkpoints/yoloxpose_l_8xb32-300e_coco-640-de0f8dee_20230829.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--input-video',
        default='0',
        type=str,
        help='webcam id or input video file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/ava/label_map.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--output-fps',
        default=15,
        type=int,
        help='the fps of demo video output')
    parser.add_argument(
        '--out-filename',
        default=None,
        type=str,
        help='the filename of output video')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show results with cv2.imshow')
    parser.add_argument(
        '--display-height',
        type=int,
        default=0,
        help='Image height for human detector and draw frames.')
    parser.add_argument(
        '--display-width',
        type=int,
        default=0,
        help='Image width for human detector and draw frames.')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a prediction per n frames')
    parser.add_argument(
        '--clip-vis-length',
        default=8,
        type=int,
        help='Number of draw frames per clip.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")

    args = parser.parse_args()
    return args


class TaskInfo:
    """Wapper for a clip.

    Transmit data around three threads.

    1) Read Thread: Create task and put task into read queue. Init `frames`,
        `processed_frames`, `img_shape`, `ratio`, `clip_vis_length`.
    2) Main Thread: Get data from read queue, predict human bboxes and stdet
        action labels, draw predictions and put task into display queue. Init
        `display_bboxes`, `stdet_bboxes` and `action_preds`, update `frames`.
    3) Display Thread: Get data from display queue, show/write frames and
        delete task.
    """

    def __init__(self):
        self.id = -1

        # raw frames, used as human detector input, draw predictions input
        # and output, display input
        self.frames = None

        # stdet params
        self.processed_frames = None  # model inputs
        self.frames_inds = None  # select frames from processed frames
        self.img_shape = None  # model inputs, processed frame shape
        # `action_preds` is `list[list[tuple]]`. The outer brackets indicate
        # different bboxes and the intter brackets indicate different action
        # results for the same bbox. tuple contains `class_name` and `score`.
        self.action_preds = None  # stdet results
        self.stdet_results=None
        # human bboxes with the format (xmin, ymin, xmax, ymax)
        self.display_bboxes = None  # bboxes coords for self.frames
        self.stdet_bboxes = None  # bboxes coords for self.processed_frames
        self.stdet_poses = None  # bboxes coords for self.processed_frames
        self.stdet_poses_data_samples = None  # bboxes coords for self.processed_frames
        self.ratio = None  # processed_frames.shape[1::-1]/frames.shape[1::-1]
        self.track_history = None
        self.frame_buffer = None
        self.keypoint_score = None
        self.boxes = None
        # for each clip, draw predictions on clip_vis_length frames
        self.clip_vis_length = -1
        self.timestamps = None
    def add_frames(self, idx, frames, processed_frames):
        """Add the clip and corresponding id.

        Args:
            idx (int): the current index of the clip.
            frames (list[ndarray]): list of images in "BGR" format.
            processed_frames (list[ndarray]): list of resize and normed images
                in "BGR" format.
        """
        self.frames = frames
        self.processed_frames = processed_frames
        self.id = idx
        self.img_shape = processed_frames[0].shape[:2]
    def add_item(self,track_history,frame_buffer ,boxes):
        self.track_history = track_history
        self.frame_buffer = frame_buffer
        self.boxes = boxes
    def add_bboxes(self, display_bboxes):
        """Add correspondding bounding boxes."""
        self.display_bboxes = display_bboxes
        
        self.stdet_bboxes = display_bboxes.copy()
        for i in range(len(self.frames)):
            det2 = self.stdet_bboxes[i]
            det = self.display_bboxes[i]
            det[:, ::2] *= self.ratio[0]
            det[:, 1::2]*= self.ratio[1]
            self.stdet_bboxes[i] = torch.from_numpy(det2[:, :4]).to("cuda")
            self.display_bboxes[i] = det[:, :4]
        
    def add_poses(self, preds , data_samples):
        """Add the corresponding action predictions."""
        self.stdet_poses = preds.copy()
        self.stdet_poses_data_samples = data_samples.copy()
        
    def add_action_preds(self, timestamps,stdet_preds):
        """Add the corresponding action predictions."""
        self.action_preds = stdet_preds
        self.timestamps = timestamps

    def get_model_inputs(self, device):
        """Convert preprocessed images to MMAction2 STDet model inputs."""
        cur_frames = [self.processed_frames[idx] for idx in self.frames_inds]
        input_array = np.stack(cur_frames).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(device)
        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=self.stdet_bboxes)
        datasample.set_metainfo(dict(img_shape=self.img_shape))

        return dict(
            inputs=input_tensor, data_samples=[datasample], mode='predict')


class StdetPredictor:
    """Wrapper for MMAction2 spatio-temporal action models.

    Args:
        config (str): Path to stdet config.
        ckpt (str): Path to stdet checkpoint.
        device (str): CPU/CUDA device option.
        score_thr (float): The threshold of human action score.
        label_map_path (str): Path to label map file. The format for each line
            is `{class_id}: {class_name}`.
    """

    def __init__(self, config, checkpoint, device, score_thr, label_map_stdet,display_height,
        display_width,predict_stepsize,clip_helper):
        self.action_score_thr = score_thr
        self.config =config
        # load model
        # self.config.model.backbone.pretrained = None
        # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        # load_checkpoint(model, checkpoint, map_location='cpu')
        # model.to(device)
        # model.eval()
        self.device = device
        self.label_map_stdet = label_map_stdet
        self.stdet_label_map = load_label_map(label_map_stdet)
        self.num_class = max(self.stdet_label_map.keys()) + 1  # for AVA dataset (81)
        self.config.model.cls_head.num_classes = self.num_class
        self.model = init_recognizer(config, checkpoint, device=device)
        self.predict_stepsize = predict_stepsize
        self.cliper = clip_helper
        # self.new_w, self.new_h = mmcv.rescale_size((self.w, self.h), (256, np.Inf))
        self.clip_len, self.frame_interval = 30, 1
    def skeleton_based_stdet(self, label_map, task:TaskInfo):
        self.clip_len= len(task.boxes)
        window_size = self.clip_len * self.frame_interval
        timestamps = np.arange(window_size // 2, NUM_FRAME + 1 - window_size // 2,PREDICT_STEP )
        skeleton_predictions = []
        for timestamp in timestamps:
            start_frame = timestamp - (self.clip_len // 2 - 1) * self.frame_interval
            frame_inds = start_frame + np.arange(0, window_size, self.frame_interval)
            frame_inds = list(frame_inds - 1)
            num_frame = len(frame_inds)  # 30
            new_track_history = defaultdict(task.track_history.default_factory)
            for key in task.track_history.keys():
                if not len(task.track_history[key])>len(frame_inds):
                    continue
                new_track_history[key] = [task.track_history[key][ind] for ind in frame_inds]

            skeleton_prediction = defaultdict(lambda:[])
            for track_id, item in new_track_history.items():
                for  k in item:
                    (keypoints,boxes) =k
                    fake_anno = dict(
                        frame_dict='',
                        label=-1,
                        img_shape=(self.cliper.h, self.cliper.w),
                        origin_shape=(self.cliper.h, self.cliper.w),
                        start_index=0,
                        modality='Pose',
                        total_frames=len(task.frame_buffer))
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

                    output = inference_recognizer(self.model, fake_anno)
                    # for multi-label recognition
                    score = output.pred_score.tolist()
                    score.sort(key=lambda x: -x)
                    # count=0
                    for k in range(len(score)):  # 81
                        if k not in label_map:
                            continue
                        # if count>4:
                            # break
                        skeleton_prediction[track_id].append((boxes,label_map[k], score[k]))
                        break
                        # count+=1
            skeleton_predictions.append(skeleton_prediction)
        return  timestamps, skeleton_predictions
    def predict(self, task):
        """Spatio-temporval Action Detection model inference."""
        # No need to do inference if no one in keyframe
        if len(task.boxes) == 0:
            return task

        timestamps, stdet_preds = self.skeleton_based_stdet( self.stdet_label_map,
                                                       task)
        # stdet_results = []
        # for timestamp in timestamps:
        #     human_detection = task["boxes"][timestamp - 1]
        #     stdet_results.append(
        #         pack_result(human_detection, prediction, self.cliper.new_h, self.cliper.new_w))        


        task.timestamps=timestamps
        task.stdet_results=stdet_preds

        return task


class ClipHelper:
    """Multithrading utils to manage the lifecycle of task."""

    def __init__(self,
                 config,
                 display_height=0,
                 display_width=0,
                 input_video=0,
                 predict_stepsize=40,
                 output_fps=25,
                 clip_vis_length=8,
                 out_filename="output.mp4",
                 show=True,
                 stdet_input_shortside=256):
        # stdet sampling strategy
        val_pipeline = config.val_pipeline
        self.stdet_input_shortside = stdet_input_shortside
        clip_len, frame_interval = 30,1
        self.window_size = clip_len * frame_interval
        self.output_fps = output_fps
        self.show = show
        self.video_writer = None
        if out_filename is not None:
            self.video_writer = self.get_output_video_writer(out_filename)
        # asserts
        assert (out_filename or show), \
            'out_filename and show cannot both be None'
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'
        assert clip_vis_length <= predict_stepsize
        assert 0 < predict_stepsize <= self.window_size

        self.model2 = YOLO('checkpoints/yolov8x-pose-p6.pt')  # Load an official Pose model
        self.display_height=display_height
        self.display_width=display_width

        self.results2 = self.model2.track(source="rtsp://dudnjs:aiffelmodu@192.168.219.102:554/stream1" ,stream=True,show=False,persist=True)
        # start_time = time.time()
        self.h, self.w = next(self.results2).orig_shape
        self.stdet_input_size = mmcv.rescale_size(
        (self.w, self.h), (self.stdet_input_shortside, np.Inf))

        self.new_w, self.new_h = mmcv.rescale_size((self.w, self.h), (256, np.Inf))
        self.predict_stepsize = predict_stepsize
    
        # output/display params
        if self.display_height > 0 and self.display_width > 0:
            self.display_size = (self.display_width, self.display_height)
        elif self.display_height > 0 or self.display_width > 0:
            self.display_size = mmcv.rescale_size(
                (self.w, self.h), (np.Inf, max(self.display_height, self.display_width)))
        else:
            self.display_size = (self.w, self.h)
        self.ratio = tuple(
            n / o for n, o in zip(self.stdet_input_size, self.display_size))


        display_start_idx = self.window_size // 2 - self.predict_stepsize // 2
        self.display_inds = [
            display_start_idx + i for i in range(self.predict_stepsize)
        ]

        self.clip_vis_length = clip_vis_length
        self.predict_stepsize = predict_stepsize
        self.buffer_size = self.window_size - self.predict_stepsize
        frame_start = self.window_size // 2 - (clip_len // 2) * frame_interval
        self.frames_inds = [
            frame_start + frame_interval * i for i in range(clip_len)
        ]
        self.buffer = []
        self.processed_buffer = []

        # display multi-theading params
        self.display_id = -1  # task["id"] for display queue
        self.display_queue = queue.Queue()
        # self.display_lock = threading.Lock()
        # self.output_lock = threading.Lock()

        # read multi-theading params
        self.read_id = -1  # task["id"] for read queue
        # self.read_id_lock = threading.Lock()
        self.read_queue = queue.Queue()
        # self.read_lock = threading.Lock()
        self.not_end = True  # cap.read() flag

        # program state
        self.stopped = False
        
        atexit.register(self.clean)
        self.frame_buffer = []
        self.poses = []
        self.track_history = None
        self.keypoint_score = []
        self.boxes = []
         
    def read_fn(self,read_queue):
        """Main function for read thread.

        Contains three steps:

        1) Read and preprocess (resize + norm) frames from source.
        2) Create task by frames from previous step and buffer.
        3) Put task into read queue.
        """
        was_read = True
        start_time = time.time()
        poses=[]
        boxes=[]
        frame_buffer=[]
        for r in self.results2:
            track_history = defaultdict(lambda: [])

            # get data from read queue
            # inference_start = time.time()
            keypoints2 = r.keypoints.xy.cpu()
            res_plotted = r.plot()
            frame_buffer.append(res_plotted)
            boxes2 = r.boxes.xywh.cpu()
            task = TaskInfo()
            
            # self.keypoint.append(keypoints.xy.cpu().numpy())
            poses.append(r.keypoints)
                        # update read state
            track_ids=None
            if r.boxes.id!=None:
                track_ids = r.boxes.id.int().cpu().tolist()
                for box,keypoint, track_id in zip(r.boxes.xyxy, r.keypoints, track_ids):
                    track = track_history[track_id]
                    track.append((keypoint,box)) 
                    if len(track) > NUM_FRAME:  # retain 90 tracks for 90 frames
                        track.pop(0)
                    boxes.append(r.boxes.xyxy)
            if len(frame_buffer) == NUM_FRAME:
                # Prepare a task with the accumulated frames
                # frame_buffer = np.array(self.frame_buffer)
                # keypoint_score = np.array(self.keypoint_score)
                # keypoint_score = np.transpose(keypoint_score, axes=(1,0, 2 ))
                # keypoint = np.transpose(keypoint, axes=(1, 0, 2,3))
                task.add_item(track_history,frame_buffer  ,boxes)
                # task = {
                #     "track_history": track_history ,
                #     'frame_buffer': frame_buffer,
                #     'keypoint_score': keypoint_score,
                #     'boxes': boxes
                # }

                # Clear the buffer for the next set of frames

                # Update read state and put the task into the read queue
                # with self.read_id_lock:
                #     self.read_id += 1
                #     self.not_end = was_read
                # task["id"]=copy.deepcopy(self.read_id)
                read_queue.put((was_read, copy.deepcopy(task)))
                for track_id, keypoint in track_history.items():
                    track = track_history[track_id]
                    if len(track) > 0: 
                        track.clear()
                poses.clear()
                frame_buffer.clear()
                # keypoint_score.clear()
                boxes.clear()

        # If the loop ends but there are frames left in the buffer, put them as a task as well
            cur_time = time.time()
            # logger.debug(
            #     f'Read thread: {1000*(cur_time - start_time):.0f} ms, '
            #     f'{read_frame_cnt / (cur_time - before_read):.0f} fps')
            start_time = cur_time
                    

    def display_fn(self,display_queue):
        """Main function for display thread.

        Read data from display queue and display predictions.
        """
        while not self.stopped:

            self.display_id += 1
            task = display_queue.get()
            if task==None:
                continue
 
            for frame_id in range(NUM_FRAME):
                frame = task.frame_buffer[frame_id]
                if self.show:
                    # cv2.imshow('Demo', frame)
                    # cv2.waitKey(int(1000 / self.output_fps))
                    pass
                if self.video_writer:
                    self.video_writer.write(frame)

    def __iter__(self):
        return self

    def __next__(self):
        """Get data from read queue.

        This function is part of the main thread.
        """
        
        if self.read_queue.qsize() == 0:
            time.sleep(0.02)
            return not self.stopped, None

        was_read, task = self.read_queue.get()
        if not was_read:
            # If we reach the end of the video, there aren't enough frames
            # in the task.processed_frames, so no need to model inference
            # and draw predictions. Put task into display queue.
            with self.read_id_lock:
                read_id = self.read_id
            with self.display_lock:
                self.display_queue[read_id] = was_read, copy.deepcopy(task)

            # main thread doesn't need to handle this task again
            task = None
        return was_read, task

    def start(self):
        """Start read thread and display thread."""
        self.read_thread = threading.Thread(
            target=self.read_fn, args=(self.read_queue,), name='VidRead-Thread', daemon=True)
        self.read_thread.start()
        self.display_thread = threading.Thread(
            target=self.display_fn,
            args=(self.display_queue,),
            name='VidDisplay-Thread',
            daemon=True)
        self.display_thread.start()

        return self

    def clean(self):
        """Close all threads and release all resources."""
        self.stopped = True
        # self.read_lock.acquire()
        # self.read_lock.release()
        # self.output_lock.acquire()
        cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()
        # self.output_lock.release()

    def join(self):
        """Waiting for the finalization of read and display thread."""
        self.read_thread.join()
        self.display_thread.join()

    def display(self, task):
        """Add the visualized task to the display queue.

        Args:
            task (TaskInfo object): task object that contain the necessary
            information for prediction visualization.
        """
        # with self.display_lock:
        self.display_queue.put(copy.deepcopy(task) )

    def get_output_video_writer(self, path):
        """Return a video writer object.

        Args:
            path (str): path to the output video file.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=float(self.output_fps),
            frameSize=self.display_size,
            isColor=True)


class BaseVisualizer(metaclass=ABCMeta):
    """Base class for visualization tools."""

    def __init__(self, max_labels_per_bbox):
        self.max_labels_per_bbox = max_labels_per_bbox

    def draw_predictions(self, task):
        """Visualize stdet predictions on raw frames."""
        # read bboxes from task

        # bboxes = task.display_bboxes.cpu().numpy()
        # bboxes = task.stdet_bboxes
        # dense_n = int(predict_stepsize / output_stepsize)
        dense_n =8
        # output_timestamps = dense_timestamps(task.timestamps, dense_n)
        # output_timestamps = np.arange(30)
        # frames = [
        #     # cv2.imread(task["frame_buffer"][timestamp - 1])
        #     task["frame_buffer"][timestamp - 1]
        #     for timestamp in output_timestamps
        # ]        
        
        # pose_datasample = [
        #     task.stdet_poses_data_samples[timestamp - 1] for timestamp in output_timestamps
        # ]
      

        # # draw predictions and update task
        # keyframe_idx = len(task["frame_buffer"]) // 2
        # draw_range = [
        #     keyframe_idx - task.clip_vis_length // 2,
        #     keyframe_idx + (task.clip_vis_length - 1) // 2
        # ]
        # assert draw_range[0] >= 0 and draw_range[1] < len(task["frame_buffer"])
        # task["frame_buffer"] = self.draw_clip_range(task["frame_buffer"], task.action_preds,
        #                                    bboxes, draw_range)
        task.frame_buffer  = self.draw_one_image(task.frame_buffer, task.stdet_results)
        # task["frame_buffer"]  = self.draw_one_image2(frames, pose_datasample)
        return task
    def draw_one_image2(self, frames, pose_data_samples):
        """Draw predictions on one image."""
        scale_ratio = np.array([self.clip.w, self.clip.h, self.clip.w, self.clip.h])
        max_num=5
        assert max_num + 1 <= len(PLATEBLUE)

        frames_ = copy.deepcopy(frames)
        # frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
        anno = None
    # add pose results
        if pose_data_samples:
            pose_config = mmengine.Config.fromfile(self.pose_config)
            visualizer = VISUALIZERS.build(pose_config.visualizer)
            visualizer.set_dataset_meta(pose_data_samples[0].dataset_meta)
            for i, (d, f) in enumerate(zip(pose_data_samples, frames_)):
                visualizer.add_datasample(
                    'result',
                    f,
                    data_sample=d,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=True,
                    show=False,
                    wait_time=0,
                    out_file=None,
                    kpt_thr=0.3)
                frames_[i] = visualizer.get_image()
        return frames_

    def draw_clip_range(self, frames, preds, bboxes, draw_range):
        """Draw a range of frames with the same bboxes and predictions."""
        # no predictions to be draw
        if bboxes is None or len(bboxes) == 0:
            return frames

        # draw frames in `draw_range`
        left_frames = frames[:draw_range[0]]
        right_frames = frames[draw_range[1] + 1:]
        draw_frames = frames[draw_range[0]:draw_range[1] + 1]

        # get labels(texts) and draw predictions
        draw_frames = [
            self.draw_one_image(frames, bboxes, preds) for frame in draw_frames
        ]

        return list(left_frames) + draw_frames + list(right_frames)

    @abstractmethod
    def draw_one_image(self, frame, bboxes, preds):
        """Draw bboxes and corresponding texts on one frame."""

    @staticmethod
    def abbrev(name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name


class DefaultVisualizer(BaseVisualizer):
    """Tools to visualize predictions.

    Args:
        max_labels_per_bbox (int): Max number of labels to visualize for a
            person box. Default: 5.
        plate (str): The color plate used for visualization. Two recommended
            plates are blue plate `03045e-023e8a-0077b6-0096c7-00b4d8-48cae4`
            and green plate `004b23-006400-007200-008000-38b000-70e000`. These
            plates are generated by https://coolors.co/.
            Default: '03045e-023e8aclip_helper-0077b6-0096c7-00b4d8-48cae4'.
        text_fontface (int): Fontface from OpenCV for texts.
            Default: cv2.FONT_HERSHEY_DUPLEX.
        text_fontscale (float): Fontscale from OpenCV for texts.
            Default: 0.5.
        text_fontcolor (tuple): fontface from OpenCV for texts.
            Default: (255, 255, 255).
        text_thickness (int): Thickness from OpenCV for texts.
            Default: 1.
        text_linetype (int): LInetype from OpenCV for texts.
            Default: 1.
    """

    def __init__(
            self,
            clip_helper,
            pose_config,
            max_labels_per_bbox=5,
            plate='03045e-023e8a-0077b6-0096c7-00b4d8-48cae4',
            text_fontface=cv2.FONT_HERSHEY_DUPLEX,
            text_fontscale=0.5,
            text_fontcolor=(255, 255, 255),  # white
            text_thickness=1,
            text_linetype=1,
            ):
        super().__init__(max_labels_per_bbox=max_labels_per_bbox)
        self.text_fontface = text_fontface
        self.text_fontscale = text_fontscale
        self.text_fontcolor = text_fontcolor
        self.text_thickness = text_thickness
        self.text_linetype = text_linetype
        self.pose_config= pose_config
        def hex2color(h):
            """Convert the 6-digit hex string to tuple of 3 int value (g)"""
            return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))
        self.clip= clip_helper
        plate = plate.split('-')
        self.plate = [hex2color(h) for h in plate]
 
    def draw_one_image(self, frames, annotations):
        """Draw predictions on one image."""
        # scale_ratio = np.array([self.clip.w, self.clip.h, self.clip.w, self.clip.h])
        max_num=5
        assert max_num + 1 <= len(PLATEBLUE)
        nf, na = len(frames), len(annotations)
        assert nf % na == 0
        nfpa = len(frames) // len(annotations)
        frames_ = copy.deepcopy(frames)
        # frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
        anno = None
    # add pose results
        for i in range(na):
            anno = annotations[i]
            if anno is None :
                continue
            for j in range(nfpa):
                ind = i * nfpa + j
                frame = frames_[ind]

                # add action result for whole video

                # add spatio-temporal action detection results
                results = []
                for track_id ,ann in anno.items():
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
                        # box = (box * scale_ratio).astype(np.int64)
                        st, ed = tuple(box[:2]), tuple(box[2:])
                        # if not pose_data_samples:
                        #     cv2.rectangle(frame, st, ed, PLATEBLUE[0], 2)

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
                            cv2.rectangle(frame, diag0, diag1, PLATEBLUE[k + 1], -1)
                            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                        FONTCOLOR, THICKNESS, LINETYPE)
        return frames_


def main(args):
    # init human detector
    human_detector = MmdetHumanDetector(args.det_config, args.det_checkpoint,
                                        args.device, args.det_score_thr)
    pose_estimator = MmdetPoseEstimator(args.pose_config, args.pose_checkpoint,
                                        args.device, args.det_score_thr)

    # init action detector
    config = Config.fromfile(args.skeleton_config)

    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass
        # init clip helper
    clip_helper = ClipHelper(
        config=config,
        display_height=args.display_height,
        display_width=args.display_width,
        input_video=args.input_video,
        predict_stepsize=args.predict_stepsize,
        output_fps=args.output_fps,
        clip_vis_length=args.clip_vis_length,
        out_filename=args.out_filename,
        show=args.show)
    stdet_predictor = StdetPredictor(
        config=config,
        checkpoint=args.skeleton_stdet_checkpoint,
        device=args.device,
        score_thr=args.action_score_thr,
        label_map_stdet=args.label_map ,
        display_height=args.display_height,
        display_width=args.display_width,
        predict_stepsize=args.predict_stepsize,
        clip_helper= clip_helper
        )


    # init visualizer
    vis = DefaultVisualizer(clip_helper=clip_helper,pose_config=args.pose_config)
    # start read and display thread
    clip_helper.start()
    # Tracking with default tracker

    try:
        # Main thread main function contains:
        # 1) get data from read queue
        # 2) get human bboxes and stdet predictions
        # 3) draw stdet predictions and update task
        # 4) put task into display queue
        for able_to_read, task in clip_helper:
            # get data from read queue

            if not able_to_read:
                # read thread is dead and all tasks are processed
                break

            if task is None:
                # when no data in read queue, wait
                time.sleep(0.01)
                continue
            # get human bboxes
            # human_detector.predict(task)
            # pose_estimator.predict(task)
            # get stdet predictions
            task =stdet_predictor.predict(task)

            # draw stdet predictions in raw frames
            task = vis.draw_predictions(task)
            # logger.info(f'Stdet Results: {task.action_preds}')

            clip_helper.display(task)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # logger.debug('Main thread inference time '
            #              f'{1000*(time.time() - inference_start):.0f} ms')

        # wait for display thread
        clip_helper.join()
    except KeyboardInterrupt:
        pass
    finally:
        # close read & display thread, release all resources
        clip_helper.clean()


if __name__ == '__main__':
    main(parse_args())
