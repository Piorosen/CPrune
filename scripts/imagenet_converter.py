
#%%
import os
import shutil

# validation 이미지 폴더 경로 및 매핑 파일 경로
val_dir = '/work/frontend/imagenet'  # validation 이미지가 저장된 경로
mapping_file = '/work/frontend/txt.txt'  # 클래스 매핑 파일

# 클래스 ID에 해당하는 폴더를 만들어서 이미지들을 옮김
def reorganize_val_set(val_dir, mapping_file):
    # 클래스 라벨 파일 열기
    with open(mapping_file, 'r') as f:
        lines = f.readlines()
    
    # 이미지 파일들 가져오기
    image_files = sorted(os.listdir(val_dir))

    # 클래스별로 폴더를 만들고 이미지 이동
    for i, line in enumerate(lines):
        class_id = line.strip()  # 클래스 ID 가져오기
        image_file = image_files[i]  # 이미지 파일명
        
        # 해당 클래스 폴더가 없으면 생성
        class_dir = os.path.join(val_dir, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # 이미지를 해당 클래스 폴더로 이동
        src = os.path.join(val_dir, image_file)
        dst = os.path.join(class_dir, image_file)
        shutil.move(src, dst)

# 실행
reorganize_val_set(val_dir, mapping_file)

# %%
