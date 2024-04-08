# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import json
import os
import shutil
import time
from collections import defaultdict, deque
from operator import itemgetter
from threading import Thread
import multiprocessing as mp
from types import SimpleNamespace
from typing import Optional
from fastapi import FastAPI
import mmengine
import mmcv
from queue import Queue
import cv2
import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel
import pytz
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
from HR_pro import optimization 
from bn_wvad import my_infer_ljy
app = FastAPI()
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
    cfg_options = {} 
    
    args = SimpleNamespace(
        config=os.getenv('CONFIG', 'configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        checkpoint=os.getenv('CHECKPOINT', 'https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_ava.pth'),
        label_map_stdet=os.getenv('LABEL_MAP_STDET', 'tools/data/ava/label_map.txt'),
        label_map=os.getenv('LABEL_MAP', 'tools/data/kinetics/label_map_k400.txt'),
        device=os.getenv('DEVICE', 'cuda:0'),
        camera_id=int(os.getenv('CAMERA_ID', '0')),
        threshold=float(os.getenv('THRESHOLD', '0.01')),
        average_size=int(os.getenv('AVERAGE_SIZE', '1')),
        drawing_fps=int(os.getenv('DRAWING_FPS', '20')),
        inference_fps=int(os.getenv('INFERENCE_FPS', '4')),
        output_file=os.getenv('OUTPUT_FILE', None),
        action_score_thr=float(os.getenv('ACTION_SCORE_THR', '0.4')),
        # cfg_options는 여기서 단순화를 위해 생략했으나, 필요에 따라 적절히 처리
        cfg_options=cfg_options
    )
    return args


def frame_read(stop_event):
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
    while not stop_event.is_set():
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

def inference_pose(hr_to_pose_queue,pose_queue_stdet ,stop_event):
    yolo =YOLO('checkpoints/yolov8x-pose-p6.pt')
    while not stop_event.is_set():
        if hr_to_pose_queue.qsize()!=0 :
            print("inference_pose")
            filtered_frames = hr_to_pose_queue.get()
            for frames in filtered_frames:
                start_time,end_time,frames = frames
                pose_deque=deque()
                for i in  frames:
                    r = yolo.track(i,persist=True,verbose=False)
                    pose_deque.append(r[0])
                if(len(pose_deque)!=0):
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

def hr_pro(hr_pro_queue , hr_to_pose_queue,queue, stop_event):
    fps = queue.get()
    threshold = queue.get()
    args_hrpro = optimization.parse_env_vars()
    model1, model2 = optimization.hr_model(args_hrpro)
    
    print(fps)
    while not stop_event.is_set():
        if hr_pro_queue.qsize()!=0:
            print("hr_pro")
            print(f"theshold:{threshold}")
            start_time,end_time,frames ,concatenated_features = hr_pro_queue.get()
            # 제안 생성
            final_proposal =  optimization.hr_pro(args_hrpro,concatenated_features,model1, model2,threshold)
            #시간대 겹치는 부분 병합
            merged_data = merge_overlapping_segments(final_proposal)
            # 시간순 정렬
            sorted_data = sorted(merged_data, key=lambda x: x['segment'][0])
            print(sorted_data)
            # 프레임으로 바꿈
            
            sorted_data_with_frames = [{
            'label': item['label'],
            'score': item['score'],
            'frames': (int(round(item['segment'][0] * fps)), int(round(item['segment'][1] * fps)))
            } for item in sorted_data]
            filtered_frames=[]
            for item in sorted_data_with_frames:
                (start_frame,end_frame)=item["frames"]
                # 프레임이랑 timestamp를 더해서 해당 time 구하기
                start_frame_time = start_time + timedelta(seconds=(start_frame / fps))
                formatted_start_frame_time = start_frame_time.strftime('%Y-%m-%d %H:%M:%S')
                end_frame_time = start_time + timedelta(seconds=(end_frame / fps))
                formatted_end_frame_time = end_frame_time.strftime('%Y-%m-%d %H:%M:%S')
                
                filtered_frames.append([formatted_start_frame_time,formatted_end_frame_time,frames[start_frame:end_frame + 1]])
            hr_to_pose_queue.put(filtered_frames)
def bn_wvad(bn_wvad_queue , bn_wvad_to_pose_queue, stop_event):
    args = my_infer_ljy.parse_args()
    net = my_infer_ljy.wvad_model_load(args)
    while not stop_event.is_set():
        if bn_wvad_queue.qsize()!=0:
            start_time,end_time,frames ,features = bn_wvad_queue.get()
            res = my_infer_ljy.infer_numpy(net,features)
            print(res)
            res =res.mean()
            bn_wvad_to_pose_queue.put(res)
            
            
    
    pass
def i3d(hr_pro_queue, stop_event,bn_wvad_queue):
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
    while not stop_event[0].is_set():
        if  len(frame_queue)!=0:
            start_time,end_time,frame_list = frame_queue.popleft()
            features = extractor.extract(frame_list)
            rgb_features = features['rgb']
            flow_features = features['flow']
            concatenated_features = np.concatenate((rgb_features, flow_features), axis=1)
            hr_pro_queue.put(copy.deepcopy([start_time,end_time,frame_list,concatenated_features]))
            bn_wvad_queue.put(copy.deepcopy([start_time,end_time,frame_list,rgb_features]))
            
            
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
            #yolov8 공식 문서 참고
            for box,keypoint, track_id in zip(r.boxes.xyxy, r.keypoints, track_ids):
                track = track_history[track_id]
                #트랙 아이디가 키 값 키포인트 와 신뢰도 리스트가 담김
                track.append((keypoint.xy.cpu().numpy(),keypoint.conf.cpu().numpy(),box)) 
                # if len(track) > NUM_FRAME:  # retain 90 tracks for 90 frames
                #     track.pop(0)

    return start_time,end_time,  track_history ,item ,frames
def inference(pose_queue,queue,result_queue, stop_event):
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
    while not stop_event.is_set():
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
@app.get("/cleanup")
def cleanup_videos():
    # 비디오 폴더 경로 설정
    cred = credentials.Certificate("/home/bigdeal/mnt2/workspace/mmaction2/demo/cctv-cc0b7-firebase-adminsdk-chce8-439053bc3e.json")
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    video_folder = "video"

    # 비디오 폴더 내의 모든 파일 삭제
    for filename in os.listdir(video_folder):
        file_path = os.path.join(video_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # Firestore에서 문서 삭제
    # 예시로 'video_logs' 컬렉션의 모든 문서를 삭제합니다.
    video_logs_ref = db.collection('video_logs')
    action_logs_ref = db.collection('action_logs')
    try:
        docs = video_logs_ref.stream()
        for doc in docs:
            doc.reference.delete()
        docs = action_logs_ref.stream()
        for doc in docs:
            doc.reference.delete()
    except Exception as e:
        return {"error": f"Failed to delete Firestore documents. Reason: {e}"}
    return {"status": "Cleanup completed successfully."}
def visualize(result_queue_stdet,queue,stop_event,):
    #관리자 인증
    cred = credentials.Certificate('/home/bigdeal/mnt2/workspace/mmaction2/demo/cctv-cc0b7-firebase-adminsdk-chce8-439053bc3e.json')
    #초기화
    firebase_admin.initialize_app(cred)
    # Firestore 인스턴스를 가져옵니다.
    db = firestore.client()
    fps = queue.get()  
    h = queue.get() 
    w = queue.get()  
    user = queue.get()  
    is_recording = queue.get()  
    rtspURL = queue.get()  
    push_tokens= queue.get()  
    # user = queue.get()  
    try:
        # 사용자의 이메일로 UID 조회
        user = auth.get_user_by_email(user)
        print(f"User ID: {user.uid}")
    except auth.UserNotFoundError:
        print(f"No user found for email: {user}")

    while not stop_event.is_set():
        if result_queue_stdet.qsize()>0:
            print("visualize")
            start_time,end_time,pose_deque,results2,frames = result_queue_stdet.get()
            file_name= f"{start_time}-{end_time}.mp4"
            if(is_recording): 
                print(is_recording)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(os.path.join("video",file_name), fourcc, fps, (w, h))
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
                        # if results2!=None and (track_id2==track_id):
                            # cv2.putText(frame, "label", (startX+1, startY), FONTFACE, FONTSCALE,FONTCOLOR, THICKNESS, LINETYPE)
                        # 박스랑  트랙아이디 그려주기
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
                if(is_recording):
                    out.write(frame)

            if(is_recording):
                out.release()
            # 비디오 로그 저장
            if is_recording:
                video_ref = db.collection('video_logs').document()
                start_time_converted = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                end_time_converted = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                timezone_str = "Asia/Seoul"
                timezone = pytz.timezone(timezone_str)
                start_time_converted = timezone.localize(start_time_converted)
                end_time_converted = timezone.localize(end_time_converted)
                video_data = {
                    'video_name': file_name,
                    'timestamp': firestore.SERVER_TIMESTAMP,  # 서버 시간을 사용
                    'start-stamp': start_time_converted,
                    'end-stamp': end_time_converted,
                    'cctv_id': rtspURL,
                    'user_id': user.uid,
                    'action_ids': [],
                }
                video_ref.set(video_data)
                print(f"Saved video log with ID: {video_ref.id}")

                action_ids = []
                # 비디오에 관련된 행동 로그 저장
                for item in results2:
                    track_id, skeleton_prediction, box = item
                    print(skeleton_prediction)
                    data_to_store = json.dumps(skeleton_prediction)

                    action_ref = db.collection('action_logs').document()
                    action_data = {
                        'timestamp': firestore.SERVER_TIMESTAMP,  # 서버 시간을 사용
                        'cctv_id': rtspURL,
                        'video_id': video_ref.id,  # 위에서 저장한 비디오 로그의 ID를 참조
                        'prediction': data_to_store,
                        'user_id': user.uid,
                        'track_id': track_id
                    }
                    action_ref.set(action_data)
                    print(f"Saved action log with ID: {action_ref.id}")    
                    action_ids.append(action_ref.id)

                if action_ids:  # action_ids 리스트가 비어 있지 않은 경우에만 업데이트를 수행
                    video_ref.update({'action_ids': firestore.ArrayUnion(action_ids)})
            
            if(push_tokens!=None):
                registration_token = push_tokens

                message = messaging.Message(
                data={
                    'score': '850',
                    'time': '2:45'
                },
                notification=messaging.Notification(
                    title='절도 알림',
                    body='절도 확률 높음.',
                ),
                token=registration_token,
                )

                # 메시지 보내기
                response = messaging.send(message)
                print('Successfully sent message:', response)

def inference_stdet(pose_queue_stdet,queue,result_queue_stdet,stop_event,bn_to_result_queue):
    data = queue.get()  # 첫 번째 아이템 (dict)
    args = queue.get()  # 두 번째 아이템 (argparse.Namespace)
    label_map = load_label_map(args["label_map_stdet"])
    
    num_class = max(label_map.keys()) + 1  # for AVA dataset (81)
    skeleton_config = mmengine.Config.fromfile(args["config"])
    skeleton_config.model.cls_head.num_classes = num_class # for K400 dataset
    model = init_recognizer(skeleton_config, args["checkpoint"], device=args["device"])
    while not stop_event.is_set():
        keypoint_score= []
        while len(keypoint_score) == 0:
            if pose_queue_stdet.qsize() > 0:
                print("pose_queue_stdet")
                start_time,end_time,track_history ,pose_deque ,frames= get_items_from_queue_stdet(pose_queue_stdet)
                res =  bn_to_result_queue.get()
                print(start_time,end_time)
                #빈 차원 (frame len, 1, 17, 2)
                combined_keypoint = np.zeros((len(frames), 1, NUM_KEYPOINT, 2),
                        dtype=np.float16)
                combined_keypoint_score = np.zeros((len(frames), 1, NUM_KEYPOINT),
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
                    #빈 차원 채우기
                    combined_keypoint_score=np.array(one_person_keypoint_score)
                    combined_keypoint=np.array(one_person_keypoint)
                    cur_data = data.copy()
                    #차원 형식 맞춰주기 (배치 차원, frame num  , keypoints , xy)
                    cur_data['keypoint'] = combined_keypoint.transpose((1, 0, 2, 3))
                    #차원 형식 맞춰주기 (배치 차원, frame num  , keypoints_score , )
                    cur_data['keypoint_score'] = combined_keypoint_score.transpose((1, 0, 2))
                    cur_data["total_frames"]=combined_keypoint.shape[0]
                    #추론
                    output = inference_recognizer(model, cur_data)
                                # for multi-label recognition
                    score = output.pred_score.tolist()
                    skeleton_prediction=[]
                    #라벨, 추론 점수 가져오기
                    for k in range(len(score)):  # 81
                        if k not in label_map:
                            continue
                        # TODO:가중치 기반 앙상블 실험을 통해 최적의 값을 찾아야됨
                        if(k==len(score)-1):
                            score[k]=score[k]*res
                        if score[k] > args["action_score_thr"]:
                            
                            skeleton_prediction.append((label_map[k], score[k]))

                    print(f"track_id:{track_id},action_label: {skeleton_prediction},box:{box.cpu().numpy()}")
                    my_tuple= (track_id,skeleton_prediction,box.cpu().numpy() )
                    my_list.append(my_tuple)
                result_queue_stdet.put(copy.deepcopy([start_time,end_time,pose_deque,my_list,frames]))
# 카메라 로직 중지
stop_event=None
pw = ps_i3d = ps = ps_pose = pr2 = pr3 = camera = ps_bn = None

@app.get("/stop")
def stop_camera():
    global stop_event, pw, ps_i3d, ps, ps_pose, pr2, pr3, camera, ps_bn
    isStart=False
    if stop_event is not None:
        stop_event.set()  # 이벤트 설정으로 프로세스 내 루프 종료 시도

        # 모든 프로세스에 대해 강제 종료 시도 전 약간의 대기 시간 제공
        time.sleep(2)  # 예: 2초 대기

        # 각 프로세스의 생존 여부 확인 후 강제 종료
        # if pw is not None and pw.is_alive(): pw.terminate()
        if ps_pose is not None and ps_pose.is_alive(): ps_pose.terminate()
        if ps is not None and ps.is_alive(): ps.terminate()
        if pr2 is not None and pr2.is_alive(): pr2.terminate()
        if pr3 is not None and pr3.is_alive(): pr3.terminate()
        if ps_bn is not None and ps_bn.is_alive(): pr3.terminate()

        # # 프로세스가 종료될 때까지 기다립니다.
        # if pw is not None: pw.join()
        # if ps_i3d is not None: ps_i3d.join()
        # if ps_pose is not None: ps_pose.join()
        # if ps is not None: ps.join()
        # if pr2 is not None: pr2.join()
        # if pr3 is not None: pr3.join()

        if camera is not None and camera.isOpened():
            camera.release()

        return {"status": "Camera monitoring forcefully stopped."}
    else:
        return {"status": "Camera is not monitoring or already stopped."}
push_tokens=None
rtspURL=None
@app.get("/status")
def get_state():
    """
    현재 프로세스 상태를 확인하여 반환합니다.
    """
    global pw ,user ,is_recording ,push_tokens ,rtspURL
    if pw is not None and pw.is_alive():
        pw_value= True
    else:
        pw_value= False
        
    return {"process_alive": pw_value,"is_recording": is_recording,"token": push_tokens,"rtsp":rtspURL,"user":user,"threshold":threshold,"push_threshold":push_threshold }
# 상태 업데이트를 위한 Pydantic 모델
class StatusUpdate(BaseModel):
    is_recording: Optional[bool]
    process_alive: Optional[bool]
    token: Optional[str]
    rtsp: Optional[str]
    user: Optional[str]
    threshold: Optional[float]
    push_threshold: Optional[float]
user=None
isStart=False
push_threshold= 0.0 
@app.post("/update-status")
def update_status(update: StatusUpdate):
    global is_recording,rtspURL ,push_tokens ,pw ,user ,isStart,threshold,push_threshold
    is_recording = update.is_recording
    user = update.user
    push_tokens = update.token
    rtspURL = update.rtsp
    threshold =update.threshold
    push_threshold = update.push_threshold
    if(update.process_alive==True):
        # and (pw is not None and pw.is_alive()==False) or pw is None
        if(isStart!=True  ):
            main()
    elif(update.process_alive==False):
         if(isStart!=False):
             stop_camera()
    # 여기서 다른 상태 업데이트 로직을 추가할 수 있습니다.
    return {"messege": "update"}
# 전역 변수 선언
is_recording = False

class RecordCommand(BaseModel):
    record: bool

@app.post("/record")
def set_recording(command: RecordCommand):
    global is_recording
    is_recording = command.record
    return {"is_recording": is_recording}

@app.get("/apply")
def apply():
    
    if(isStart==True):
        stop_camera()
        main()
    if(isStart==False):
        main()
    
    return {"apply":"ok"}


class RTSPData(BaseModel):
    url: str
@app.post("/rtsp")
def set_rtsp(rtsp_data: RTSPData):
    """
    새로운 RTSP URL을 설정합니다.
    """
    global rtspURL,camera
    rtspURL = rtsp_data.url
    return {"message": f"RTSP URL has been set to {rtspURL}"}
# 사용자 데이터를 설정하기 위한 요청 바디 모델
class UserData(BaseModel):
    email: str
# 사용자 정보를 저장하기 위한 전역 변수, 실제 사용 사례에서는 데이터베이스 등을 사용할 수 있습니다.
@app.post("/user")
def set_user(user_data: UserData):
    """
    사용자 정보를 설정합니다.
    """
    
    global user
    user = user_data.email
    print(user)
    return {"message": "User data has been set successfully."}
cred=None
@app.get("/push")
def send_push():
    """
    사용자 정보를 설정합니다.
    """
    global push_tokens ,cred
    
    if(cred==None):
        cred = credentials.Certificate('/home/bigdeal/mnt2/workspace/mmaction2/demo/cctv-cc0b7-firebase-adminsdk-chce8-439053bc3e.json')
        firebase_admin.initialize_app(cred)
    # Firestore 인스턴스를 가져옵니다.
    # 푸시 토큰 저장
    registration_token = push_tokens

    message = messaging.Message(
    data={
    'score': '850',
    'time': '2:45'
    },
    notification=messaging.Notification(
    title='절도 알림',
    body='절도 확률 높음.',
    ),
    token=registration_token,
    )
    response = messaging.send(message)
    print('Successfully sent message:', response)
    return {"message": "User data has been set successfully."}
# 푸시 토큰 설정을 위한 요청 바디 모델
class PushTokenData(BaseModel):
    push_token: Optional[str] 
# 푸시 토큰 정보를 저장하기 위한 전역 변수, 실제 애플리케이션에서는 보다 안전한 저장소 사용을 고려해야 합니다.
@app.post("/token")
def set_push_token(token_data: PushTokenData):
    """
    사용자 디바이스의 푸시 알림 토큰을 설정합니다.
    """
    global push_tokens
    # 푸시 토큰 저장
    push_tokens = token_data.push_token
    return {"message": "Push token has been set successfully."}
# 카메라 로직 시작
threshold= 0.0 
@app.get("/start")
def main():
    mp.set_start_method('spawn', force=True)
    
    global average_size, threshold, drawing_fps, inference_fps, \
        device,  camera, data, label, sample_length, \
        threshold,isStart,args, frame_queue, result_queue ,img_shape ,pose_queue,out,pose_deque,fps,stop_event,pw,ps_i3d,ps,ps_pose,pr2,pr3,ps_bn
    isStart=True
    img_shape=None
    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps
    device = torch.device(args.device)
    out=None
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    print(rtspURL)
    # Build the recognizer from a config file and checkpoint file/url
    # camera = cv2.VideoCapture("/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS_02.구매행동_02.선택/C_2_2_65_BU_SYB_10-15_10-03-46_CA_RGB_DF1_F4_F4.mp4")
    
    camera = cv2.VideoCapture("demo/test_video_structuralize.mp4")
    if not camera.isOpened():
        print("스트림을 열 수 없습니다.")
        return {"status": "Camera monitoring ended."}
    else:
        # 스트림 처리 로직...
        print("스트림을 열 수있습니다.")
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
        stop_event = mp.Event()
        frame_queue = deque()
        hr_pro_queue = mp.Queue(maxsize=1)
        hr_to_pose_queue = mp.Queue(maxsize=1)
        bn_to_result_queue = mp.Queue(maxsize=1)
        bn_wvad_queue = mp.Queue(maxsize=1)
        pose_deque=deque(maxlen=30)
        
        # pose_queue = deque(maxlen=sample_length)
        result_queue =  mp.Queue()
        result_queue_stdet =  mp.Queue()
        pose_queue = mp.Queue()
        pose_queue_stdet = mp.Queue()
        data_info_queue3 = mp.Queue()
        data_info_queue4 = mp.Queue()
        data_info_queue3.put(fps)  
        data_info_queue3.put(threshold)  
        data_info_queue4.put(fps)  
        data_info_queue4.put(h)  
        data_info_queue4.put(w)  
        data_info_queue4.put(user)  
        data_info_queue4.put(is_recording)  
        data_info_queue4.put(rtspURL)  
        data_info_queue4.put(push_tokens)  
        
        data_info_queue2 = mp.Queue()
        data_info_queue2.put(data)  
        data_info_queue2.put(vars(args))
        data_info_queue2.put(cfg)
        data_info_queue = mp.Queue()
        data_info_queue.put(data)  
        data_info_queue.put(vars(args))
        data_info_queue.put(cfg)
        pw = Thread(target=frame_read, args=([stop_event]), daemon=True)
        ps_i3d = Thread(target=i3d, args=(hr_pro_queue,[stop_event],bn_wvad_queue), daemon=True)
        ps = mp.Process(target=hr_pro, args=(hr_pro_queue,hr_to_pose_queue,data_info_queue3,stop_event), daemon=True)
        ps_bn = mp.Process(target=bn_wvad, args=(bn_wvad_queue,bn_to_result_queue,stop_event), daemon=True)
        ps_pose = mp.Process(target=inference_pose, args=(hr_to_pose_queue,pose_queue_stdet,stop_event), daemon=True)
        pr2 = mp.Process(target=inference_stdet, args=(pose_queue_stdet,data_info_queue,result_queue_stdet,stop_event,bn_to_result_queue), daemon=True)
        pr3 = mp.Process(target=visualize, args=(result_queue_stdet,data_info_queue4,stop_event) , daemon=True)
        pw.start()
        ps_i3d.start()
        ps_pose.start()
        ps.start()
        ps_bn.start()
        pr2.start()
        pr3.start()
        pw.join()
        ps_i3d.join()
        ps_pose.join()
        pr2.join()
        pr3.join()
        ps_bn.join()
        ps.join()

        print("메인 프로세스 종료.")
    except KeyboardInterrupt:
        if out!=None:
            out.release()
        pass
        return {"status": "Camera monitoring started."}
    else:
        return {"status": "Camera already monitoring."}


