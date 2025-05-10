import os
import json

# 설정
test_dir = "data/vizwiz/test"
json_path = "data/vizwiz/test_grounding.json"
expected_total = 2373  # 기대되는 총 test 이미지 개수

# 실제 존재하는 이미지 파일들 (jpg만 대상)
test_images = set(f for f in os.listdir(test_dir) if f.lower().endswith('.jpg'))

# JSON에 명시된 이미지 파일들
with open(json_path, 'r') as f:
    json_data = json.load(f)
json_images = set(json_data.keys())

# 분석
json_but_missing_in_folder = sorted(json_images - test_images)
folder_but_missing_in_json = sorted(test_images - json_images)

# 출력
print(f"✅ test 폴더에 있는 이미지 수: {len(test_images)}")
print(f"✅ JSON에 있는 항목 수: {len(json_images)}")
print(f"🔻 JSON에는 있으나 test 폴더에 *없는* 이미지 수: {len(json_but_missing_in_folder)}")
for f in json_but_missing_in_folder[:5]:
    print("    ❌", f)

print(f"🔻 test 폴더에는 있으나 JSON에 *없는* 이미지 수: {len(folder_but_missing_in_json)}")
for f in folder_but_missing_in_json[:5]:
    print("    ❌", f)

# 누락 여부 확인
if len(test_images) != expected_total:
    print(f"\n⚠️ 예상과 다른 이미지 개수입니다! (예상: {expected_total}, 실제: {len(test_images)})")
else:
    print("\n✅ 이미지 수가 예상과 일치합니다.")