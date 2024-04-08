import os
import cv2
import xmltodict
import pickle
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
def cut_video(video_path, new_path,data_root_path):
    new_video_path=video_path.replace(data_root_path, new_path)
    output_path = f'{new_video_path[:-4]}.mp4'
    new_video_list.append(output_path)
    if(os.path.exists(output_path)):
        print("이미 컷편집을 완료했습니다.")
        print("*************")
        return
    new_video_path=video_path.replace(data_root_path, new_path)

    new_video_dir = os.path.dirname(new_video_path)
    # Create the directory if it doesn't exist
    create_directory_recursively(new_video_dir)
    print(output_path)
  # 비디오 파일 로드
    video = VideoFileClip(video_path)

    # 비디오의 특정 부분을 선택 (예: 처음 1분)
    clip = video.subclip(0, 60)  # 시작 시간, 끝 시간 (초 단위)

    # 결과 저장
    clip.write_videofile(output_path)
        


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
        if file.endswith(f".{end}") and  "매장이동" in full_path:
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
erro_xml_list=[]
erro_video_list=[]
erro_xml2_list=[]
erro_video2_list=[]
new_video_list=[]
new_video2_list=[]
count=0 #  700 90 35 40
for idx, j in enumerate(mp4_files):
    data_root_path= "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터"
    new_path= "/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/PreProcessToOneMinute"
    # XML 파일을 열고 읽기
    print("------")
    print(idx)
    
    cut_video( j,new_path,data_root_path)
# with open("output_xml_error.pkl", 'wb') as file2:
#     pickle.dump(erro_xml_list, file2)
# with open("output_video_error.pkl", 'wb') as file2:
#     pickle.dump(erro_video_list, file2)
    
                
# with open("output_xml_moving_error.pkl", 'wb') as file2:
#     pickle.dump(erro_xml2_list, file2)
# with open("output_video_moving_error.pkl", 'wb') as file2:
#     pickle.dump(erro_video2_list, file2)

