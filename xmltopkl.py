import xmltodict
action_list= [
    "select_start","select_end",
    "test_start", "test_end",
    "buying_start", "buying_end",
    "return_start", "return_end",
    "compare_start", "compare_end"
    ]
with open("/home/bigdeal/mnt2/238-1.실내(편의점,_매장)_사람_구매행동_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL_01.매장이동_01.매장이동/C_1_1_85_BU_DYB_10-19_13-12-56_CB_DF1_M4_M4.xml", 'r', encoding='utf-8') as file:
    xml_string = file.read()
# XML 문자열을 파이썬 딕셔너리로 변환
dict_data = xmltodict.parse(xml_string)
for i in range(len(dict_data['annotations']['track'])):
    if dict_data['annotations']['track'][i]['@label'] in action_list:
        print(dict_data['annotations']['track'][i]['@label'])
        moving_start=int(dict_data['annotations']['track'][i]["box"][0]["@frame"])
        moving_end=int(dict_data['annotations']['track'][i]["box"][0]["@frame"])+len(dict_data['annotations']['track'][i]["box"])-1
        print(moving_start)
        print(moving_end)

#action으로 시작과 끝 프레임을 큐에 담고
#우선 관절 딕셔너리를 만든다 {관절이름 : 정수}
#딕셔너리를 만든다 {프레임:[]}