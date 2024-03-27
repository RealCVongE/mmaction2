# %%
import xmltodict
import os
def create_directory_recursively(path):
    try:
        # exist_ok=True 옵션은 해당 경로의 폴더가 이미 존재해도 에러를 발생시키지 않음
        os.makedirs(path, exist_ok=True)
        # print(f"'{path}' 디렉토리가 생성되었거나 이미 존재합니다.")
    except PermissionError:
        # 권한 오류 처리
        print(f"권한 오류: '{path}' 디렉토리를 생성할 수 없습니다. 접근 권한을 확인하세요.")
    except Exception as e:
        # 기타 오류 처리
        print(f"오류 발생: {e}")
def find_files(root_directory, partial_name):
    """
    주어진 루트 디렉토리에서 부분 이름을 포함하는 모든 파일의 경로를 찾습니다.

    :param root_directory: 검색을 시작할 루트 디렉토리
    :param partial_name: 찾고자 하는 파일 이름의 일부
    :return: 찾은 파일의 전체 경로 목록
    """
    matching_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if partial_name in file:
                matching_files.append(os.path.join(root, file))
    return matching_files
def is_point_inside_bbox(bbox, point):
    """
    주어진 bbox 내에 특정 좌표가 포함되어 있는지 확인합니다.

    :param bbox: Bounding box의 좌표 (x_min, y_min, x_max, y_max)
    :param point: 확인할 좌표 (x, y)
    :return: 좌표가 bbox 내에 있으면 True, 아니면 False
    """
    x_min, y_min, x_max, y_max = bbox
    x, y = point

    return x_min <= x <= x_max and y_min <= y <= y_max
def filter_tracks_by_label(annotations, labels):
    """
    주어진 라벨에 해당하는 트랙들만 필터링합니다.

    Parameters:
    - annotations: 'dict_data['annotations']["track"]' 형태의 리스트
    - labels: 필터링할 라벨들의 리스트

    Returns:
    - filtered_tracks: 주어진 라벨들에 해당하는 트랙들의 리스트
    """
    filtered_tracks = [track for track in annotations if track["@label"] in labels]
    return filtered_tracks

# 필터링할 라벨들
labels_to_filter = [
    "Pelvis",
    "Center head",
    "Left hip",
    "Left knee",
    "Left foot",
    "Right  hip",  # 'Right hip' 오타 수정
    "Right knee",
    "Right foot",
    "Spine naval",
    "Spine chest",
    "Neck base",
    "Right shoulder",
    "Right elbow",
    "Right hand",
    "Left shoulder",
    "Left elbow",
    "Left hand"
]
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bbox_with_keypoints(keypoints):
    """
    주어진 키포인트들을 포함하는 바운딩 박스를 그립니다.

    Parameters:
    - keypoints: [(x1, y1), (x2, y2), ..., (x17, y17)] 형태의 키포인트 리스트
    """

    # 키포인트들로부터 x와 y 좌표 분리
    x_coords = [kp[0] for kp in keypoints]
    y_coords = [kp[1] for kp in keypoints]
    
    # 바운딩 박스의 최소 x, 최대 x, 최소 y, 최대 y 계산
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    # # 바운딩 박스의 폭과 높이 계산
    # width = max_x - min_x
    # height = max_y - min_y

    # # 바운딩 박스 및 키포인트 그리기
    # fig,ax = plt.subplots(1)
    # ax.scatter(x_coords, y_coords)  # 키포인트 그리기
    # bbox = patches.Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(bbox)
    # plt.show()
    return  min_x, min_y, max_x, max_y
def group_points_by_frame(filtered_tracks):
    """
    필터링된 트랙들에서 동일 프레임의 키포인트들을 리스트로 묶습니다.

    Parameters:
    - filtered_tracks: 필터링된 트랙들의 리스트

    Returns:
    - grouped_points: 프레임별 키포인트 리스트의 딕셔너리
    """
    frame_points = {}

    # 모든 트랙과 포인트를 순회하면서 프레임별로 키포인트 그룹화
    for track in filtered_tracks:
            point=    track["points"][0]
            frame = point["@frame"]
            # 각 포인트를 (x, y) 형태로 변환
            point_tuple =  tuple([int(x)  for x in point["@points"].split(",")])
            if frame not in frame_points:
                frame_points[frame] = [point_tuple]
            else:
                frame_points[frame].append(point_tuple)

    # 동일 프레임 내의 키포인트들을 정렬하여 튜플로 변환
    grouped_points = {frame: sorted(points, key=lambda x: x[0]) for frame, points in frame_points.items()}

    return grouped_points

# %%
import os
def count_mp4_files(path,end):
  """
  주어진 경로와 하위 폴더들을 재귀적으로 탐색하여 mp4 파일 개수를 센다.

  Args:
    path: 탐색을 시작할 경로

  Returns:
    mp4 파일 개수
  """
  mp4_files = []
  mp4_count = 0
  def _count_mp4_files_in_dir(dir_path):
    nonlocal mp4_count
    nonlocal mp4_files
    for file in os.listdir(dir_path):
      full_path = os.path.join(dir_path, file)
      if os.path.isfile(full_path):
        if file.endswith(f".{end}") and not "매장이동" in full_path:
          mp4_count += 1
          mp4_files.append(full_path)
      elif os.path.isdir(full_path):
        _count_mp4_files_in_dir(full_path)

  _count_mp4_files_in_dir(path)

  return mp4_count, mp4_files

# 예시 코드
path = "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터"
xml_count, xml_files = count_mp4_files(path,"xml")
print(xml_count)

# %%
import json
import xmltodict



# %%
import pickle
data=[]
with open("output.pkl", 'wb') as file2:
    pickle.dump(data, file2)
with open("output.pkl", 'rb') as file:
    data = pickle.load(file)
print(data)

# %%
import copy
import pickle
import numpy as np  
root_directory = "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/PreProcessToPkl" # 검색을 시작할 루트 디렉토리 경로로 변경하세요.
error_list=[]
action_list= [
    "select_start","select_end",
    "test_start", "test_end",
    "buying_start", "buying_end",
    "return_start", "return_end",
    "compare_start", "compare_end"
    ]
for idx, xml_file in enumerate(xml_files):
    print("-------------------",idx)
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_string = file.read()
        # XML 문자열을 파이썬 딕셔너리로 변환
    dict_data = xmltodict.parse(xml_string)
    actionbbox_list=[]
    for i in dict_data['annotations']["track"]:
        if i["@label"] in action_list:
            if "start" in i["@label"]:
                if(type(i['box'])==list):
                    bbox = (i["@label"] , i['box'][0]["@frame"] ,(float(i['box'][0]["@xtl"]),float(i['box'][0]["@ytl"]),float(i['box'][0]["@xbr"]),float(i['box'][0]["@ybr"])))
                    actionbbox_list.append(bbox)
                else:
                    bbox = (i["@label"] , i['box']["@frame"] ,(float(i['box']["@xtl"]),float(i['box']["@ytl"]),float(i['box']["@xbr"]),float(i['box']["@ybr"])))
                    actionbbox_list.append(bbox)
    # annotations 데이터 예시 (사용자의 실제 데이터를 여기에 맞게 조정 필요)
    annotations = dict_data['annotations']["track"]

    # 필터링 실행
    filtered_annotations = filter_tracks_by_label(annotations, labels_to_filter)


    # 필터링된 트랙을 사용하여 함수 호출
    grouped_points_by_frame = group_points_by_frame(filtered_annotations)

    # 결과 확인
    # for frame, points in grouped_points_by_frame.items():
    #     print(len(points))
    #     print(f"Frame {frame}: {points}")
    bbox_list=[]
    for frame, points in grouped_points_by_frame.items():
        min_x, min_y, max_x, max_y = draw_bbox_with_keypoints(points)
        bbox_list.append( (min_x, min_y, max_x, max_y))
    pkl_path = copy.deepcopy(xml_file)
    pklsplit=pkl_path.split("/")[-1][:-4]
    found_files = find_files(root_directory, pklsplit)
    if(len(found_files)!=len(actionbbox_list)):
        print(len(found_files))
        print(found_files)
        error_list.append(found_files)
        print(len(actionbbox_list))
        print(actionbbox_list)
        continue
    for idx , found_file in enumerate(found_files):
        with open(found_file, 'rb') as file:
            data = pickle.load(file)
            newfilepath=found_file.replace("PreProcessToPkl","PreProcessToPkl3")
            if(os.path.exists(newfilepath)):
                continue
            if(data["keypoint"].shape[0]==1):
                continue
            data_keypoints = list(data["keypoint"].transpose((1, 0, 2, 3)))
            data_keypoint_scores = list(data["keypoint_score"].transpose((1, 0, 2)))
            new_data_keypoint_list=[]
            new_data_keypoint_score_list=[]
            print(found_files)
            print(data["keypoint"].shape)
            
            for item in zip(data_keypoints,data_keypoint_scores,bbox_list):
                data_keypoint,data_keypoint_score,bbox= item
                points = list(data_keypoint)
                scores = list(data_keypoint_score)
                for idx2, item2 in enumerate(zip(points,scores)):
                    point ,score=item2
                    for m in list(point):
                        if is_point_inside_bbox(bbox, m):
                            point = np.expand_dims(point,axis=0)
                            score = np.expand_dims(score,axis=0)
                            new_data_keypoint_list.append(point)
                            new_data_keypoint_score_list.append(score)
                            
                            break
                    
            new_video_dir = os.path.dirname(newfilepath)
            create_directory_recursively(new_video_dir)
            new_data_keypoint=np.array(new_data_keypoint_list)
            print(new_data_keypoint.shape)
            new_data_keypoint_score=np.array(new_data_keypoint_score_list)
            print(new_data_keypoint_score.shape)
            data["keypoint"] = new_data_keypoint.transpose((1, 0, 2, 3))
            data["keypoint_score"] = new_data_keypoint_score.transpose((1, 0, 2))
            print("바뀐keypoint:",data["keypoint"].shape)
            with open(newfilepath, 'wb') as file2:
                pickle.dump(data, file2)
with open("error_from_xml_pkl_topkl", 'wb') as file2:
    pickle.dump(error_list, file2)

