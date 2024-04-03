# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import time
from collections import defaultdict, deque
from operator import itemgetter
from threading import Thread
import multiprocessing as mp
import mmengine
import mmcv
from queue import Queue
from HR_pro import optimization 
import cv2
import numpy as np
from omegaconf import OmegaConf
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose, pseudo_collate
from ultralytics import YOLO
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth
from firebase_admin import messaging
# 현재 시간
from mmaction.apis import init_recognizer
from mmaction.apis.inference import inference_recognizer
from mmaction.utils import get_str_type
from video_extract import ExtractI3D
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
PREDICT_STACK=100
PREDICT_STEP=100
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


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    # parser.add_argument(
    #     '--label-map-stdet',
    #     default='checkpoints/label_map_my_stdet.txt',
    #     help='label map file for spatio-temporal action detection')
    # parser.add_argument(
    #     '--label-map',
    #     default='checkpoints/label_map_my.txt',
    #     help='label map file for action recognition')
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


def show_results():
    print('Press "Esc", "q" or "Q" to exit')
    # 비디오 저장을 위한 설정

    current_frame = 0
    text_info = {}
    bbox_info = None
    bbox_info2 = None
    cur_time = time.time()
    first_frame = True
    predict_stack=0
    frame_list =[]
    start_time =None
    end_time =None
    start_time = datetime.now()
    while True:
        msg = 'Waiting for action ...'
        
        frame_exists, frame = camera.read()
        results2=None
        if first_frame:
            first_frame = False
            if not frame_exists:
                continue
        if frame_exists:
            if (len(frame_list)!=0 and  len(frame_list)% PREDICT_STACK==0 ):
                end_time = datetime.now()
                frame_queue.append(copy.deepcopy([start_time, end_time, frame_list]))
                start_time = datetime.now()
                frame_list.clear()
            frame_list.append(frame)

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
        if frame_exists:
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
def inference_pose(hr_to_pose_queue,pose_queue_stdet):
    yolo =YOLO('checkpoints/yolov8x-pose-p6.pt')
    while True:
        if hr_to_pose_queue.qsize()!=0 :
            filtered_frames = hr_to_pose_queue.get()
            for frames in filtered_frames:
                start_time,end_time,frames = frames
                pose_deque=deque()
                for i in  frames:
                    r = yolo.track(i,persist=True,verbose=False)
                    pose_deque.append(r[0])
                if(len(pose_deque)!=0):
                    # pose_queue.put(copy.deepcopy([start_time,end_time, start_frame,end_frame,pose_deque]))
                    pose_queue_stdet.put(copy.deepcopy([start_time,end_time, pose_deque ,frames]))
                  
# 겹치는 세그먼트들을 합치는 함수 구현
def merge_overlapping_segments(data):
    if not data:
        return []

    # 결과 리스트 초기화
    merged_segments = []
    # 첫 번째 세그먼트로 시작
    current_start, current_end = data[0]['segment']

    for item in data[1:]:
        start, end = item['segment']
        # 현재 세그먼트와 겹치는 경우, 세그먼트를 합침
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            # 겹치지 않는 경우, 현재까지의 세그먼트를 결과에 추가하고, 새 세그먼트로 시작
            merged_segments.append({'label': 'Merged', 'score': None, 'segment': [current_start, current_end]})
            current_start, current_end = start, end
    
    # 마지막 세그먼트 추가
    merged_segments.append({'label': 'Merged', 'score': None, 'segment': [current_start, current_end]})

    return merged_segments
def hr_pro(hr_pro_queue , hr_to_pose_queue,queue):
    fps = queue.get()
    print(fps)
    while True:
        if hr_pro_queue.qsize()!=0:
            start_time,end_time,frames ,concatenated_features = hr_pro_queue.get()
            final_proposal =  optimization.hr_pro(optimization.parse_args(),concatenated_features)
            merged_data = merge_overlapping_segments(final_proposal)
            sorted_data = sorted(merged_data, key=lambda x: x['segment'][0])
            print(sorted_data)
            sorted_data_with_frames = [{
            'label': item['label'],
            'score': item['score'],
            'frames': (int(round(item['segment'][0] * fps)), int(round(item['segment'][1] * fps)))
            } for item in sorted_data]
            filtered_frames=[]
            for item in sorted_data_with_frames:
                (start_frame,end_frame)=item["frames"]
                total_length = end_time - start_time
                total_frames = fps * total_length.total_seconds()
                start_frame_time = start_time + timedelta(seconds=(start_frame / fps))
                end_frame_time = start_time + timedelta(seconds=(end_frame / fps))
                filtered_frames.append([start_frame_time,end_frame_time,frames[start_frame:end_frame + 1]])
            hr_to_pose_queue.put(filtered_frames)
def bn_wvad(bn_wvad_queue,bn_to_result_queue):
    pass
def i3d(hr_pro_queue):
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
    predict_step=0
    extractor = ExtractI3D(args2)
    while True:
        if  len(frame_queue)!=0:
            start_time,end_time,frame_list = frame_queue.popleft()
            features = extractor.extract(frame_list)
            rgb_features = features['rgb']
            flow_features = features['flow']
            concatenated_features = np.concatenate((rgb_features, flow_features), axis=1)
            hr_pro_queue.put(copy.deepcopy([start_time,end_time,frame_list,concatenated_features]))
            # bn_wvad_queue.put((list(frame_queue), copy.deepcopy(rgb_features)))
            
            
def get_items_from_queue(queue):
    """Queue에서 num_items 개수만큼 아이템을 꺼내어 리스트로 반환"""
    keypoint=[]
    keypoint_score=[]
    start_frame, end_frame , r = queue.get()
    num_frame=len(r)
    for item in r:
        keypoint.append( item.keypoints.xy.cpu().numpy())
        keypoint_score.append( item.keypoints.conf.cpu().numpy())
    return keypoint ,keypoint_score,num_frame
def get_items_from_queue_stdet(queue):
    """Queue에서 num_items 개수만큼 아이템을 꺼내어 리스트로 반환"""
    track_history=defaultdict(lambda:[])
    start_time,end_time,item ,frames= queue.get()
    
    for r in item:
        if r.boxes.id!=None:
            track_ids = r.boxes.id.int().cpu().tolist()
            for box,keypoint, track_id in zip(r.boxes.xyxy, r.keypoints, track_ids):
                track = track_history[track_id]
                track.append((keypoint.xy.cpu().numpy(),keypoint.conf.cpu().numpy(),box)) 
                if len(track) > NUM_FRAME:  # retain 90 tracks for 90 frames
                    track.pop(0)

    return start_time,end_time,  track_history ,item ,frames
def inference(pose_queue,queue,result_queue):
    data = queue.get()  # 첫 번째 아이템 (dict)
    args = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    cfg = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    label_map = [x.strip() for x in open(args["label_map"]).readlines()]
    num_class = len(label_map)
    skeleton_config = mmengine.Config.fromfile(args["config"])
    skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset
    model = init_recognizer(skeleton_config, args["checkpoint"], device=args["device"])
    while True:
        keypoint_score= []
        while len(keypoint_score) == 0:
            if pose_queue.qsize() > 0:
                keypoint, keypoint_score, num_frame= get_items_from_queue(pose_queue)
        # frame_queue.append((np.array(r[0].keypoints.xy.cpu().numpy()),np.array(r[0].keypoints.conf.cpu().numpy())))
                num_person = max([len(x) for x in keypoint])
                combined_keypoint = np.zeros((num_frame, num_person, NUM_KEYPOINT, 2),
                        dtype=np.float16)
                combined_keypoint_score = np.zeros((num_frame, num_person, NUM_KEYPOINT),
                              dtype=np.float16)
                for f_idx, frm_pose in enumerate(zip(keypoint,keypoint_score)):
                    keypoint_m,keypoint_score_m = frm_pose 
                    frm_num_persons = keypoint_m.shape[0]
                    for p_idx in range(frm_num_persons):
                        combined_keypoint[f_idx, p_idx] = keypoint_m[p_idx]
                        combined_keypoint_score[f_idx, p_idx] = keypoint_score_m[p_idx]

        cur_data = data.copy()
        cur_data['total_frames'] = combined_keypoint_score.shape[0]
        cur_data['keypoint'] = combined_keypoint.transpose((1, 0, 2, 3))
        cur_data['keypoint_score'] = combined_keypoint_score.transpose((1, 0, 2))

        output = inference_recognizer(model, cur_data)
        max_pred_index = output.pred_score.argmax().item()
        action_label = label_map[max_pred_index]
        action_score = output.pred_score[max_pred_index]
        my_list = []
        my_tuple= (action_label,action_score.cpu().numpy() )
        my_list.append(my_tuple)
        print(my_list)
        result_queue.put(copy.deepcopy(my_list))
def visualize(result_queue_stdet,queue):
    cred = credentials.Certificate('/home/bigdeal/mnt2/workspace/mmaction2/demo/cctv-cc0b7-firebase-adminsdk-chce8-439053bc3e.json')
    firebase_admin.initialize_app(cred)
    # Firestore 인스턴스를 가져옵니다.
    db = firestore.client()
    # 사용자의 이메일 주소
    user_email = 'dudnjsckrgo@gmail.com'

    try:
        # 사용자의 이메일로 UID 조회
        user = auth.get_user_by_email(user_email)
        print(f"User ID: {user.uid}")
    except auth.UserNotFoundError:
        print(f"No user found for email: {user_email}")
    fps = queue.get()  
    h = queue.get() 
    w = queue.get()  
    # user = queue.get()  
    while True:
        if result_queue_stdet.qsize()>0:
            start_time,end_time,pose_deque,results2,frames = result_queue_stdet.get()
            file_name= f"{start_time}-{end_time}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(file_name, fourcc, fps, (w, h))
            for frame, pose in zip(frames,pose_deque)  :
                results = pose
                if(results.boxes.id!=None):
                    bbox_info = results2[0]
                    track_id2,skeleton_prediction,box  = results2[0]
                    track_ids = results.boxes.id.int().cpu().tolist()
                    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int64)
                    for box, track_id  in zip(boxes,track_ids):
                        (startX, startY, endX, endY) = box
                        # if score < threshold:
                        #     break
                        if results2!=None and (track_id2==track_id):
                            cv2.putText(frame, "label", (startX+1, startY), FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, str(track_id), (startX, startY), FONTFACE, FONTSCALE,
                                    FONTCOLOR, THICKNESS, LINETYPE)
                # elif bbox_info2!=None:
                #     (boxes,track_ids) = bbox_info2
                #     for box, track_id  in zip(boxes,track_ids):
                #         (startX, startY, endX, endY) = box
                #         # if score < threshold:
                #         #     break
                #         if(bbox_info!=None):
                #             track_id2,skeleton_prediction,box  = bbox_info
                #             if  (track_id2==track_id):
                #                 cv2.putText(frame, "label", (startX+1, startY), FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
                            
                #         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                #         cv2.putText(frame, str(track_id), (startX, startY), FONTFACE, FONTSCALE,
                #                     FONTCOLOR, THICKNESS, LINETYPE)
                out.write(frame)
            out.release()
            # 비디오 로그 저장
            video_ref = db.collection('video_logs').document()
            video_data = {
                'video_name': file_name,
                'timestamp': firestore.SERVER_TIMESTAMP, # 서버 시간을 사용
                'cctv_id': 'cctv_12345',
                'user_id': user.uid
            }
            video_ref.set(video_data)
            print(f"Saved video log with ID: {video_ref.id}")

            # 비디오에 관련된 행동 로그 저장
            action_ref = db.collection('action_logs').document()
            action_data = {
                'action_type': 'door_open',
                'timestamp': firestore.SERVER_TIMESTAMP, # 서버 시간을 사용
                'cctv_id': 'cctv_12345',
                'video_id': video_ref.id, # 위에서 저장한 비디오 로그의 ID를 참조
                'isTheft': False,
                'user_id': 'user_12345',
                'track_id': track_id
            }
            action_ref.set(action_data)
            print(f"Saved action log with ID: {action_ref.id}")    
            # 이 토큰은 클라이언트 애플리케이션에서 등록 절차를 거쳐 생성된 것입니다.
            # registration_token = 'your-registration-token'

            # # 메시지 정의
            # message = messaging.Message(
            #     data={
            #         'score': '850',
            #         'time': '2:45'
            #     },
            #     token=registration_token,
            # )

            # # 메시지 보내기
            # response = messaging.send(message)
            # print('Successfully sent message:', response)


    pass
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
            if pose_queue_stdet.qsize() > 0:
                start_time,end_time,track_history ,pose_deque ,frames= get_items_from_queue_stdet(pose_queue_stdet)
        # frame_queue.append((np.array(r[0].keypoints.xy.cpu().numpy()),np.array(r[0].keypoints.conf.cpu().numpy())))

                print(start_time,end_time)
                combined_keypoint = np.zeros((NUM_FRAME, 1, NUM_KEYPOINT, 2),
                        dtype=np.float16)
                combined_keypoint_score = np.zeros((NUM_FRAME, 1, NUM_KEYPOINT),
                              dtype=np.float16)
                my_list = []
                for track_id , item in track_history.items():
                    one_person_keypoint_score=[]
                    one_person_keypoint=[]
                    box=item[0][2]
                    for j in item:
                        item_keypoint,item_keypoint_score,item_box= j
                        one_person_keypoint.append(item_keypoint)
                        one_person_keypoint_score.append(item_keypoint_score)
                    combined_keypoint_score=np.array(one_person_keypoint_score)
                    combined_keypoint=np.array(one_person_keypoint)
                    cur_data = data.copy()
                    cur_data['keypoint'] = combined_keypoint.transpose((1, 0, 2, 3))
                    cur_data['keypoint_score'] = combined_keypoint_score.transpose((1, 0, 2))
                    cur_data["total_frames"]=combined_keypoint.shape[0]
                    output = inference_recognizer(model, cur_data)
                                # for multi-label recognition
                    score = output.pred_score.tolist()
                    skeleton_prediction=[]
                    for k in range(len(score)):  # 81
                        if k not in label_map:
                            continue
                        if score[k] > args["action_score_thr"]:
                            skeleton_prediction.append((label_map[k], score[k]))

                    print(f"track_id:{track_id},action_label: {skeleton_prediction},box:{box.cpu().numpy()}")
                    my_tuple= (track_id,skeleton_prediction,box.cpu().numpy() )
                    my_list.append(my_tuple)
                result_queue_stdet.put(copy.deepcopy([start_time,end_time,pose_deque,my_list,frames]))


def main():
    user = 'dudnjsckrgo@gmail.com'

        
    global average_size, threshold, drawing_fps, inference_fps, \
        device, model, camera, data, label, sample_length, \
        args, frame_queue, result_queue ,img_shape ,pose_queue,out,pose_deque,fps
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
    # camera = cv2.VideoCapture("/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_02.구매행동_02.선택/C_2_2_65_BU_SYB_10-15_10-03-46_CA_RGB_DF1_F4_F4.mp4")
    camera = cv2.VideoCapture("demo/test_video_structuralize.mp4")
    w = round(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = mmcv.rescale_size((w, h), (256, np.Inf))
    w_ratio, h_ratio = new_w / w, new_h / h
    sample_length=30
    fps = camera.get(cv2.CAP_PROP_FPS)  
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
        frame_queue = deque()
        hr_pro_queue = mp.Queue(maxsize=1)
        hr_to_pose_queue = mp.Queue(maxsize=1)
        bn_to_result_queue = Queue(maxsize=1)
        bn_wvad_queue = Queue(maxsize=1)
        pose_deque=deque(maxlen=30)
        
        # pose_queue = deque(maxlen=sample_length)
        result_queue =  mp.Queue()
        result_queue_stdet =  mp.Queue()
        pose_queue = mp.Queue()
        pose_queue_stdet = mp.Queue()
        data_info_queue3 = mp.Queue()
        data_info_queue4 = mp.Queue()
        data_info_queue3.put(fps)  # dict 전달
        data_info_queue4.put(fps)  # dict 전달
        data_info_queue4.put(h)  # dict 전달
        data_info_queue4.put(w)  # dict 전달
        data_info_queue4.put(user)  # dict 전달
        
        data_info_queue2 = mp.Queue()
        data_info_queue2.put(data)  # dict 전달
        data_info_queue2.put(vars(args))
        data_info_queue2.put(cfg)
        data_info_queue = mp.Queue()
        data_info_queue.put(data)  # dict 전달
        data_info_queue.put(vars(args))
        data_info_queue.put(cfg)
        pw = Thread(target=show_results, args=(), daemon=True)
        ps_i3d = Thread(target=i3d, args=(hr_pro_queue,), daemon=True)
        ps = mp.Process(target=hr_pro, args=(hr_pro_queue,hr_to_pose_queue,data_info_queue3), daemon=True)
        # ps = mp.Process(target=bn_wvad, args=(pose_queue,pose_queue_stdet,), daemon=True)
        ps_pose = mp.Process(target=inference_pose, args=(hr_to_pose_queue,pose_queue_stdet,), daemon=True)
        # pr = Thread(target=inference, args=(), daemon=True)
        # pr = mp.Process(target=inference, args=(pose_queue,data_info_queue2,result_queue,))
        pr2 = mp.Process(target=inference_stdet, args=(pose_queue_stdet,data_info_queue,result_queue_stdet,))
        pr3 = mp.Process(target=visualize, args=(result_queue_stdet,data_info_queue4))
        pw.start()
        ps_i3d.start()
        ps_pose.start()
        ps.start()
        # pr.start()
        pr2.start()
        pr3.start()
        pw.join()
        ps_i3d.join()
        ps_pose.join()
        # pr.join()
        pr2.join()
        pr3.join()
        ps.join()

    except KeyboardInterrupt:
        if out!=None:
            out.release()
        pass


if __name__ == '__main__':
    main()
