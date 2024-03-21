import cv2

def get_video_fps(video_path):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 파일이 열렸는지 확인
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return None
    
    # FPS 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 비디오 파일 닫기
    cap.release()
    
    return fps

# 비디오 파일 경로
video_path = 'C_3_12_1_BU_SMC_08-07_13-30-11_CC_RGB_DF2_M1.mp4'

# FPS 가져오기
fps = get_video_fps(video_path)

if fps is not None:
    print("비디오의 FPS는 {}입니다.".format(fps))
