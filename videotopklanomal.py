import subprocess
import os
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
def run_extraction_on_files(mp4_files,path, new_path):
    for idx, file_path in enumerate(mp4_files):
        print("----------------------")
        print(idx)
        output_path = file_path.replace(path, new_path)
        output_path = output_path.replace('.mp4', '.pkl')
        if(os.path.exists(output_path)):
          print("건너뜁니다.")
          return
        new_dir = os.path.dirname(output_path)
        create_directory_recursively(new_dir)
        command = ['python', 'tools/data/skeleton/ntu_pose_extraction_yyw.py', file_path, output_path]
        with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, text=True) as proc:
            for line in proc.stdout:
                print(line, end='')
        proc.wait()
        print("************************")
        

# 예시 코드
path = "/home/bigdeal/mnt2/238-2.실내(편의점,_매장)_사람_이상행동_데이터/PreProcess"
new_path = "/home/bigdeal/mnt2/238-2.실내(편의점,_매장)_사람_이상행동_데이터/PreProcessToPkl"
mp4_count, mp4_files = count_mp4_files(path,"mp4")

print(f"총 mp4 파일 개수: {mp4_count}")
# print(f"mp4 파일 목록: {mp4_files}")
print(f"mp4 파일 목록: {len(mp4_files)}")
print(f"mp4 파일 목록: {mp4_files[0]}")
# 명령어와 인자들을 리스트로 준비
run_extraction_on_files(mp4_files,path,new_path)