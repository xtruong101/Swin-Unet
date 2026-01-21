"""
create_train_val_lists.py
=========================
Chia 18 training cases của Synapse thành train/val sets
"""

import os
import glob
from pathlib import Path


def create_train_val_lists(root_path, list_dir, train_ratio=0.8):
    """
    Chia 18 training cases thành train/val
    
    Args:
        root_path: Đường dẫn chứa .npz files (ví dụ: ./data/Synapse/train_npz)
        list_dir: Đường dẫn tạo lists (ví dụ: ./lists/lists_Synapse)
        train_ratio: Tỷ lệ train (0.8 = 80% train, 20% val)
    
    Ví dụ:
        - Train: 14 cases (80% của 18)
        - Val: 4 cases (20% của 18)
    """
    
    # 1️⃣ Tạo thư mục
    os.makedirs(list_dir, exist_ok=True)
    print(f"✓ Tạo thư mục: {list_dir}")
    
    # 2️⃣ Kiểm tra root_path
    if not os.path.exists(root_path):
        print(f"❌ Lỗi: root_path không tồn tại: {root_path}")
        return False
    
    # 3️⃣ Tìm tất cả .npz files
    npz_files = sorted(glob.glob(os.path.join(root_path, "*.npz")))
    
    if not npz_files:
        print(f"❌ Lỗi: Không tìm thấy .npz files trong {root_path}")
        return False
    
    print(f"✓ Tìm thấy {len(npz_files)} files .npz")
    
    # 4️⃣ Tách tên file (bỏ .npz)
    all_cases = sorted([os.path.splitext(os.path.basename(f))[0] for f in npz_files])
    
    # 5️⃣ GIỚI HẠN CHỈ 18 CASES ĐẦU TIÊN (training set)
    # Synapse: 18 cases training, 12 cases test
    training_cases = all_cases[:18]
    test_cases = all_cases[18:30] if len(all_cases) >= 30 else []
    
    print(f"\n📊 Dữ liệu Synapse:")
    print(f"  - Training cases (sẽ chia train/val): {len(training_cases)} cases")
    print(f"  - Test cases (dùng sau): {len(test_cases)} cases")
    
    # 6️⃣ Chia training cases thành train/val
    split_idx = int(len(training_cases) * train_ratio)
    train_cases = training_cases[:split_idx]
    val_cases = training_cases[split_idx:]
    
    print(f"\n📈 Chia training set ({len(training_cases)} cases):")
    print(f"  - Train: {len(train_cases)} cases ({train_ratio*100:.0f}%)")
    print(f"  - Val: {len(val_cases)} cases ({(1-train_ratio)*100:.0f}%)")
    
    # 7️⃣ Ghi train.txt
    train_txt_path = os.path.join(list_dir, "train.txt")
    with open(train_txt_path, 'w') as f:
        for case in train_cases:
            f.write(case + '\n')
    print(f"\n✓ Tạo {train_txt_path}")
    print(f"  Nội dung: {', '.join(train_cases[:3])}... ({len(train_cases)} cases)")
    
    # 8️⃣ Ghi val.txt
    val_txt_path = os.path.join(list_dir, "val.txt")
    with open(val_txt_path, 'w') as f:
        for case in val_cases:
            f.write(case + '\n')
    print(f"\n✓ Tạo {val_txt_path}")
    print(f"  Nội dung: {', '.join(val_cases)}")
    
    # 9️⃣ In thông tin test set
    if test_cases:
        print(f"\n📋 Test set (dùng cho test.py sau):")
        print(f"  {len(test_cases)} cases: {', '.join(test_cases[:3])}...")
    
    print(f"\n{'='*80}")
    print("✅ Hoàn thành! Bây giờ có thể training:")
    print(f"{'='*80}")
    print(f"""
python train.py \\
  --dataset Synapse \\
  --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \\
  --root_path {root_path} \\
  --list_dir {list_dir} \\
  --max_epochs 'your number of epochs' \\
  --output_dir ./outputs/swin_unet \\
  --img_size 224 \\
  --base_lr 0.05 \\
  --batch_size 'your batch size'
""")
    
    return True


def create_train_val_lists_custom(list_dir, train_cases, val_cases):
    """
    Tạo train/val lists tuỳ chỉnh
    
    Dùng khi bạn muốn chỉ định chính xác case nào train, case nào val
    
    Args:
        list_dir: Thư mục tạo lists
        train_cases: List tên cases training (ví dụ: ["case_0001", "case_0002", ...])
        val_cases: List tên cases validation
    """
    
    os.makedirs(list_dir, exist_ok=True)
    
    # Ghi train.txt
    train_txt_path = os.path.join(list_dir, "train.txt")
    with open(train_txt_path, 'w') as f:
        for case in train_cases:
            f.write(case + '\n')
    
    # Ghi val.txt
    val_txt_path = os.path.join(list_dir, "val.txt")
    with open(val_txt_path, 'w') as f:
        for case in val_cases:
            f.write(case + '\n')
    
    print(f"✓ Tạo {train_txt_path} ({len(train_cases)} cases)")
    print(f"✓ Tạo {val_txt_path} ({len(val_cases)} cases)")


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("SYNAPSE: Chia Training Set thành Train/Val")
    print("="*80)
    
    # ✏️ CHỈNH SỬA ĐỂ MATCH VỚI HỆ THỐNG CỦA BẠN
    ROOT_PATH = "./data/Synapse/train_npz"
    LIST_DIR = "./lists/lists_Synapse"
    TRAIN_RATIO = 0.8  # 80% train (14 cases), 20% val (4 cases)
    
    success = create_train_val_lists(
        root_path=ROOT_PATH,
        list_dir=LIST_DIR,
        train_ratio=TRAIN_RATIO
    )
    
    if not success:
        print("\n❌ Tạo lists thất bại!")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✅ Sẵn sàng training!")
    print("="*80)