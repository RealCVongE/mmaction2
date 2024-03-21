import os
import cv2
import xmltodict
def create_directory_recursively(path):
    try:
        # exist_ok=True 옵션은 해당 경로의 폴더가 이미 존재해도 에러를 발생시키지 않음
        os.makedirs(path, exist_ok=True)
        print(f"'{path}' 디렉토리가 생성되었거나 이미 존재합니다.")
    except PermissionError:
        # 권한 오류 처리
        print(f"권한 오류: '{path}' 디렉토리를 생성할 수 없습니다. 접근 권한을 확인하세요.")
    except Exception as e:
        # 기타 오류 처리
        print(f"오류 발생: {e}")
def cut_video(video_path, new_path,data_root_path ,start_frame,end_frame):
    new_video_path=video_path.replace(data_root_path, new_path)
    
    output_path = f'{new_video_path[:-4]}_{start_frame}_{end_frame}.mp4'
    if(os.path.exists(output_path)):
        print("이미 컷편집을 완료했습니다.")
        print("*************")
        return
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 총 프레임 수와 FPS 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 출력 비디오 설정 (코덱, FPS, 해상도)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_video_path=video_path.replace(data_root_path, new_path)

    new_video_dir = os.path.dirname(new_video_path)
    # Create the directory if it doesn't exist
    create_directory_recursively(new_video_dir)
    print(output_path)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    current_frame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret:
            if start_frame <= current_frame <= end_frame:
                out.write(frame)
            
            if current_frame > end_frame:
                break
            
            current_frame += 1
        else:
            break

    # 모든 작업 완료 후 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("완료")
    print("*************")
    


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
        if file.endswith(f".{end}"):
          mp4_count += 1
          mp4_files.append(full_path)
      elif os.path.isdir(full_path):
        _count_mp4_files_in_dir(full_path)

  _count_mp4_files_in_dir(path)

  return mp4_count, mp4_files

# 예시 코드
path = "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터"
mp4_count, mp4_files = count_mp4_files(path,"mp4")
xml_count, xml_files = count_mp4_files(path,"xml")

print(f"총 mp4 파일 개수: {mp4_count}")
print(f"총 xml 파일 개수: {xml_count}")
# print(f"mp4 파일 목록: {mp4_files}")
print(f"mp4 파일 목록: {len(mp4_files)}")
print(f"mp4 파일 목록: {mp4_files[0]}")
print(f"mp4 파일 목록: {xml_files[0]}")
# 영상 파일 경로

for idx, j in enumerate(mp4_files):
    data_root_path= "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터"
    new_path= "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/PreProcess"
    # XML 파일을 열고 읽기
    print("------")
    print(idx)

    if "매장이동" in j:
        path_parts = j.split('/')
        print(path_parts[-1])
        filtered_items = [item for item in xml_files if path_parts[-1][:-3] in item]
        print(filtered_items[0])

        action_list= [
    "moving"
    ]
        start_q = []
        end_q = []
        print(j)
        with open(filtered_items[0], 'r', encoding='utf-8') as file:
            xml_string = file.read()
        # XML 문자열을 파이썬 딕셔너리로 변환
        dict_data = xmltodict.parse(xml_string)
        for i in range(len(dict_data['annotations']['track'])):
            if dict_data['annotations']['track'][i]['@label'] in action_list:
                print(dict_data['annotations']['track'][i]['@label'])
                if(isinstance( dict_data['annotations']['track'][i]["box"],list)):
                    moving_start=int(dict_data['annotations']['track'][i]["box"][0]["@frame"])
                    moving_end=int(dict_data['annotations']['track'][i]["box"][0]["@frame"])+len(dict_data['annotations']['track'][i]["box"])-1
                    
                else:
                    moving_start=int(dict_data['annotations']['track'][i]["box"]["@frame"])
                    moving_end=int(dict_data['annotations']['track'][i]["box"]["@frame"])+len(dict_data['annotations']['track'][i]["box"])-1
                start_q.append(moving_start)
                end_q.append(moving_end)

        start_q=sorted(start_q)
        end_q= sorted(end_q)
        print("start_q",start_q)
        print("end_q",end_q)
        shorter_list_count = min(len(start_q), len(end_q))
        for i in range(shorter_list_count):
            start_frame = start_q.pop()
            end_frame = end_q.pop()
            cut_video( j,new_path,data_root_path,start_frame,end_frame)


        
    else:
        
        path_parts = j.split('/')
        print(path_parts[-1])
        filtered_items = [item for item in xml_files if path_parts[-1][:-3] in item]
        print(filtered_items[0])

        action_list= [
    "select_start","select_end",
    "test_start", "test_end",
    "buying_start", "buying_end",
    "return_start", "return_end",
    "compare_start", "compare_end"
    ]
        start_q = []
        end_q = []
        print(j)
        with open(filtered_items[0], 'r', encoding='utf-8') as file:
            xml_string = file.read()
        # XML 문자열을 파이썬 딕셔너리로 변환
        dict_data = xmltodict.parse(xml_string)
        for i in range(len(dict_data['annotations']['track'])):
            if dict_data['annotations']['track'][i]['@label'] in action_list:
                print(dict_data['annotations']['track'][i]['@label'])
                if(isinstance(dict_data['annotations']['track'][i]["box"], list)):
                    print(dict_data['annotations']['track'][i]["box"][0]["@frame"])
                    if "start" in dict_data['annotations']['track'][i]['@label'] :
                        start_q.append(int(dict_data['annotations']['track'][i]["box"][0]["@frame"]))
                    if "end" in dict_data['annotations']['track'][i]['@label'] :
                        end_q.append(int(dict_data['annotations']['track'][i]["box"][0]["@frame"]))
                else:
                    print(dict_data['annotations']['track'][i]["box"]["@frame"])
                    if "start" in dict_data['annotations']['track'][i]['@label'] :
                        start_q.append(int(dict_data['annotations']['track'][i]["box"]["@frame"]))
                    if "end" in dict_data['annotations']['track'][i]['@label'] :
                        end_q.append(int(dict_data['annotations']['track'][i]["box"]["@frame"]))
        start_q=sorted(start_q)
        end_q= sorted(end_q)
        print("start_q",start_q)
        print("end_q",end_q)
        shorter_list_count = min(len(start_q), len(end_q))
        for i in range(shorter_list_count):
            start_frame = start_q.pop()
            end_frame = end_q.pop()
            cut_video( j,new_path,data_root_path,start_frame,end_frame)
