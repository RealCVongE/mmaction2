# Copyright (c) OpenMMLab. All rights reserved.
import sys
sys.path.append('/home/bigdeal/mnt2/workspace/mmaction2/video_feature')
sys.path.append('/home/bigdeal/mnt2/workspace/mmaction2/HR-pro')
import argparse
import copy as cp
import tempfile
import warnings
import os.path as osp
import os
import cv2
import mmcv
import mmengine
import numpy as np
from typing import Optional
import torch
from mmengine import DictAction
from mmengine.structures import InstanceData
from omegaconf import OmegaConf

from mmaction.apis import (detection_inference, inference_recognizer,
                           inference_skeleton, init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.structures import ActionDataSample
from video_feature.video_extract import ExtractI3D  
from HR_pro import optimization 

try:
    from mmdet.apis import init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn('Failed to import `init_detector` form `mmdet.apis`. '
                  'These apis are required in skeleton-based applications! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]
PLATEGREEN = '004b23-006400-007200-008000-38b000-70e000'
PLATEGREEN = PLATEGREEN.split('-')
PLATEGREEN = [hex2color(h) for h in PLATEGREEN]


def visualize(args,
              frames,
              annotations,
              pose_data_samples,
              action_result,
              plate=PLATEBLUE,
              max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted spatio-temporal
            detection results.
        pose_data_samples (list[list[PoseDataSample]): The pose results.
        action_result (str): The predicted action recognition results.
        pose_model (nn.Module): The constructed pose model.
        plate (str): The plate used for visualization. Default: PLATEBLUE.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    frames_ = cp.deepcopy(frames)
    frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])

    # add pose results
    if pose_data_samples:
        pose_config = mmengine.Config.fromfile(args.pose_config)
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
            cv2.putText(frames_[i], action_result, (10, 30), FONTFACE,
                        FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]

            # add action result for whole video
            cv2.putText(frame, action_result, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)

            # add spatio-temporal action detection results
            for ann in anno:
                box = ann[0]
                label = ann[1]
                if not len(label):
                    continue
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_data_samples:
                    cv2.rectangle(frame, st, ed, plate[0], 2)

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
                    cv2.rectangle(frame, diag0, diag1, plate[k + 1], -1)
                    cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--rgb-stdet-config',
        default=(
            'configs/detection/slowonly/'
            'slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py'
        ),
        help='rgb-based spatio temporal detection config file path')
    parser.add_argument(
        '--rgb-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/detection/ava/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/'
                 'slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb'
                 '_20201217-16378594.pth'),
        help='rgb-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--skeleton-stdet-checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'posec3d_ava.pth'),
        help='skeleton-based spatio temporal detection checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/'
                 'faster_rcnn/faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs'
        '/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--skeleton-config',
        default='configs/skeleton/posec3d'
        '/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py',
        help='skeleton-based action recognition config file path')
    parser.add_argument(
        '--skeleton-checkpoint',
        default='https://download.openmmlab.com/mmaction/skeleton/posec3d/'
        'posec3d_k400.pth',
        help='skeleton-based action recognition checkpoint file/url')
    parser.add_argument(
        '--rgb-config',
        default='configs/recognition/tsn/'
        'tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py',
        help='rgb-based action recognition config file path')
    parser.add_argument(
        '--rgb-checkpoint',
        default='https://download.openmmlab.com/mmaction/recognition/'
        'tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/'
        'tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth',
        help='rgb-based action recognition checkpoint file/url')
    parser.add_argument(
        '--use-skeleton-stdet',
        action='store_true',
        help='use skeleton-based spatio temporal detection method')
    parser.add_argument(
        '--use-skeleton-recog',
        action='store_true',
        help='use skeleton-based action recognition method')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--action-score-thr',
        type=float,
        default=0.4,
        help='the threshold of action prediction score')
    parser.add_argument(
        '--video',
        default='demo/test_video_structuralize.mp4',
        help='video file/url')
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
        '--out-filename',
        default='demo/test_stdet_recognition_output.mp4',
        help='output filename')
    parser.add_argument(
        '--predict-stepsize',
        default=8,
        type=int,
        help='give out a spatio-temporal detection prediction per n frames')
    parser.add_argument(
        '--output-stepsize',
        default=1,
        type=int,
        help=('show one frame per n frames in the demo, we should have: '
              'predict_stepsize % output_stepsize == 0'))
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
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


def skeleton_based_action_recognition(args, pose_results, h, w):
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    num_class = len(label_map)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset

    skeleton_model = init_recognizer(
        skeleton_config, args.skeleton_checkpoint, device=args.device)
    result = inference_skeleton(skeleton_model, pose_results, (h, w))
    action_idx = result.pred_score.argmax().item()
    return label_map[action_idx]


def rgb_based_action_recognition(args):
    rgb_config = mmengine.Config.fromfile(args.rgb_config)
    rgb_config.model.backbone.pretrained = None
    rgb_model = init_recognizer(rgb_config, args.rgb_checkpoint, args.device)
    action_results = inference_recognizer(rgb_model, args.video)
    rgb_action_result = action_results.pred_score.argmax().item()
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    return label_map[rgb_action_result]


def skeleton_based_stdet(args, label_map, human_detections, pose_results,
                          h, w):
    num_frame = len(pose_results)

    skeleton_config = mmengine.Config.fromfile(args.skeleton_config)
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config.model.cls_head.num_classes = num_class
    skeleton_stdet_model = init_recognizer(skeleton_config,
                                           args.skeleton_stdet_checkpoint,
                                           args.device)

    skeleton_predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    # for timestamp in timestamps:
    proposal = human_detections[0]
    if proposal.shape[0] == 0:  # no people detected
        skeleton_predictions.append(None)
        return

    # start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
    # frame_inds = start_frame + np.arange(0, window_size, frame_interval)
    # frame_inds = list(frame_inds - 1)
    # num_frame = len(frame_inds)  # 30
    print(num_frame)
    # pose_result = [pose_results[ind] for ind in frame_inds]

    skeleton_prediction = []
    for i in range(proposal.shape[0]):  # num_person
        skeleton_prediction.append([])

        fake_anno = dict(
            frame_dict='',
            label=-1,
            img_shape=(h, w),
            origin_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=num_frame)
        num_person = 1

        num_keypoint = 17
        keypoint = np.zeros(
            (num_person, num_frame, num_keypoint, 2))  # M T V 2
        keypoint_score = np.zeros(
            (num_person, num_frame, num_keypoint))  # M T V

        # pose matching
        person_bbox = proposal[i][:4]
        area = expand_bbox(person_bbox, h, w)

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
            keypoint[0, j] = poses['keypoints'][index]
            keypoint_score[0, j] = poses['keypoint_scores'][index]

        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

        output = inference_recognizer(skeleton_stdet_model, fake_anno)
        # for multi-label recognition
        score = output.pred_score.tolist()
        for k in range(len(score)):  # 81
            if k not in label_map:
                continue
            print(label_map[k])
            print(score[k])
            # if score[k] > args.action_score_thr:
            skeleton_prediction[i].append((label_map[k], score[k]))
    print(skeleton_prediction)
    skeleton_predictions.append(skeleton_prediction)

    return skeleton_predictions


def rgb_based_stdet(args, frames, label_map, human_detections, w, h, new_w,
                    new_h, w_ratio, h_ratio):

    rgb_stdet_config = mmengine.Config.fromfile(args.rgb_stdet_config)
    rgb_stdet_config.merge_from_dict(args.cfg_options)

    val_pipeline = rgb_stdet_config.val_pipeline
    sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
    clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
    assert clip_len % 2 == 0, 'We would like to have an even clip_len'

    window_size = clip_len * frame_interval
    num_frame = len(frames)
    # Note that it's 1 based here
    timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2,
                           args.predict_stepsize)

    # Get img_norm_cfg
    img_norm_cfg = dict(
        mean=np.array(rgb_stdet_config.model.data_preprocessor.mean),
        std=np.array(rgb_stdet_config.model.data_preprocessor.std),
        to_rgb=False)

    # Build STDET model
    try:
        # In our spatiotemporal detection demo, different actions should have
        # the same number of bboxes.
        rgb_stdet_config['model']['test_cfg']['rcnn'] = dict(action_thr=0)
    except KeyError:
        pass

    rgb_stdet_config.model.backbone.pretrained = None
    rgb_stdet_model = init_detector(
        rgb_stdet_config, args.rgb_stdet_checkpoint, device=args.device)

    predictions = []

    print('Performing SpatioTemporal Action Detection for each clip')
    prog_bar = mmengine.ProgressBar(len(timestamps))
    # for timestamp, proposal in zip(timestamps, human_detections):
    for timestamp in timestamps:
        proposal = human_detections[timestamp - 1]
        if proposal.shape[0] == 0:
            predictions.append(None)
            continue

        start_frame = timestamp - (clip_len // 2 - 1) * frame_interval
        frame_inds = start_frame + np.arange(0, window_size, frame_interval)
        frame_inds = list(frame_inds - 1)

        imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(args.device)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = rgb_stdet_model(
                input_tensor, [datasample], mode='predict')
            scores = result[0].pred_instances.scores
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            # Perform action score thr
            for i in range(scores.shape[1]):
                if i not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if scores[j, i] > args.action_score_thr:
                        prediction[j].append((label_map[i], scores[j,
                                                                   i].item()))
            predictions.append(prediction)
        prog_bar.update()

    return timestamps, predictions

def frame_extract(video_path: str,
                  short_side: Optional[int] = None,
                  out_dir: str = './tmp'):
    """Extract frames given video_path.

    Args:
        video_path (str): The video path.
        short_side (int): Target short-side of the output image.
            Defaults to None, means keeping original shape.
        out_dir (str): The output directory. Defaults to ``'./tmp'``.
    """
    # Load the video, extract frames into OUT_DIR/video_name
    target_dir = osp.join(out_dir, osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    assert osp.exists(video_path), f'file not exit {video_path}'
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    timestamps = []
    video = []
    flag, frame = vid.read()
    cnt = 0
    fps = vid.get(cv2.CAP_PROP_FPS)  
    new_h, new_w = None, None
    # 한 프레임당 시간 계산
    time_per_frame = 1 / fps

    while flag:
        if short_side is not None:
            if new_h is None:
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        timestamp = cnt / fps
        timestamps.append(timestamp)
        cnt += 1
        flag, frame = vid.read()
        if flag:
            video.append(frame) 

    return frame_paths, frames ,time_per_frame, fps,video

def main():
    args = parse_args()
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames ,time_per_frame, fps,video = frame_extract(
        args.video, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape
    #TODO: video_extraction
    args2 = OmegaConf.create({
    'feature_type': 'i3d',
    'device': 'cuda:0',  # 또는 'cpu'를 사용하시면 됩니다.
    'on_extraction': 'ignore',  # 이 설정은 무시될 것입니다.
    'output_path': 'ignore',  # 이 설정도 무시됩니다.
    'stack_size': 16,  # 이 값들은 예시이며 실제 값으로 대체해야 합니다.
    'step_size': 16,
    'streams': None,
    'flow_type': 'raft',
    'extraction_fps': 25,
    'tmp_path': './tmp/i3d',
    'keep_tmp_files': False,
    'show_pred': False,
    'config': None
    # 여기에 args_cli에 필요한 나머지 설정을 추가하십시오.
})  
    extractor = ExtractI3D(args2)
    features = extractor.extract(video)
    rgb_features = features['rgb']
    flow_features = features['flow']
    concatenated_features = np.concatenate((rgb_features, flow_features), axis=1)
    print(concatenated_features.shape)
    final_proposal =  optimization.hr_pro(optimization.parse_args(),concatenated_features)
    # Get Human detection results and pose results
    sorted_data = sorted(final_proposal, key=lambda x: x['segment'][0])
    sorted_data_with_frames = [{
    'label': item['label'],
    'score': item['score'],
    'frames': (int(round(item['segment'][0] * fps)), int(round(item['segment'][1] * fps)))
    } for item in sorted_data]
    print( sorted_data_with_frames)
    # 시작과 끝 프레임 사이의 프레임들만 추출
    filtered_frames=[]
    filtered_frames_paths=[]
    for item in sorted_data_with_frames:
        (start_frame,end_frame)=item["frames"]
        filtered_frames.append(original_frames[start_frame:end_frame + 1])
        filtered_frames_paths.append(frame_paths[start_frame:end_frame + 1])

    for idx ,item in enumerate(zip(filtered_frames_paths, filtered_frames)) :
        filtered_frames_path,filtered_frame =item
        human_detections, _ = detection_inference(
            args.det_config,
            args.det_checkpoint,
            filtered_frames_path,
            args.det_score_thr,
            device=args.device)
        pose_datasample = None
        if args.use_skeleton_recog or args.use_skeleton_stdet:
            pose_results, pose_datasample = pose_inference(
                args.pose_config,
                args.pose_checkpoint,
                filtered_frames_path,
                human_detections,
                device=args.device)

        # resize frames to shortside 256
        new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
        frames = [mmcv.imresize(img, (new_w, new_h)) for img in filtered_frame]
        w_ratio, h_ratio = new_w / w, new_h / h

        # Load spatio-temporal detection label_map
        stdet_label_map = load_label_map(args.label_map_stdet)
        rgb_stdet_config = mmengine.Config.fromfile(args.rgb_stdet_config)
        rgb_stdet_config.merge_from_dict(args.cfg_options)
        try:
            if rgb_stdet_config['data']['train']['custom_classes'] is not None:
                stdet_label_map = {
                    id + 1: stdet_label_map[cls]
                    for id, cls in enumerate(rgb_stdet_config['data']['train']
                                            ['custom_classes'])
                }
        except KeyError:
            pass

        action_result = None
        if args.use_skeleton_recog:
            print('Use skeleton-based recognition')
            action_result = skeleton_based_action_recognition(
                args, pose_results, h, w)
        else:
            print('Use rgb-based recognition')
            action_result = rgb_based_action_recognition(args)

        stdet_preds = None
        if args.use_skeleton_stdet:
            print('Use skeleton-based SpatioTemporal Action Detection')
            stdet_preds = skeleton_based_stdet(args, stdet_label_map,
                                                        human_detections,
                                                        pose_results, h, w)
            for i in range(len(human_detections)):
                det = human_detections[i]
                det[:, 0:4:2] *= w_ratio
                det[:, 1:4:2] *= h_ratio
                human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

        else:
            print('Use rgb-based SpatioTemporal Action Detection')
            for i in range(len(human_detections)):
                det = human_detections[i]
                det[:, 0:4:2] *= w_ratio
                det[:, 1:4:2] *= h_ratio
                human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)
            timestamps, stdet_preds = rgb_based_stdet(args, frames,
                                                    stdet_label_map,
                                                    human_detections, w, h,
                                                    new_w, new_h, w_ratio,
                                                    h_ratio)

        stdet_results = []
        if(stdet_preds==None):
            continue
        for idx,  prediction in enumerate(stdet_preds) :
            human_detection = human_detections[idx]
            stdet_results.append(
                pack_result(human_detection, prediction, new_h, new_w))

        def dense_timestamps(timestamps, n):
            """Make it nx frames."""
            old_frame_interval = (timestamps[1] - timestamps[0])
            start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
            new_frame_inds = np.arange(
                len(timestamps) * n) * old_frame_interval / n + start
            return new_frame_inds.astype(np.int64)

        dense_n = int(args.predict_stepsize / args.output_stepsize)
        # output_timestamps = dense_timestamps(timestamps, dense_n)
        frames = [
            cv2.imread(item)
            for item in filtered_frames_path
        ]

        # if args.use_skeleton_recog or args.use_skeleton_stdet:
        #     pose_datasample = [
        #         pose_datasample[timestamp - 1] for timestamp in output_timestamps
        #     ]

        vid_frames=visualize(args, frames, stdet_results, pose_datasample,
                            action_result)
        vid = mpy.ImageSequenceClip(vid_frames, fps=fps)
        vid.write_videofile(args.out_filename[:-4]+f"{idx}.mp4")

    tmp_dir.cleanup()


if __name__ == '__main__':
    main()
