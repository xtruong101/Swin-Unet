#!/usr/bin/env python3
"""
Script tạo file val.txt bằng cách chia dữ liệu từ train.txt
"""
import random
import os

# Đọc file train.txt
train_file = './lists/lists_Synapse/train.txt'
with open(train_file, 'r') as f:
    all_samples = f.readlines()

print(f"Tổng số samples: {len(all_samples)}")

# Chia 80/20 hoặc 75/25
random.seed(1234)  # Để reproducible
random.shuffle(all_samples)

split_idx = int(len(all_samples) * 0.8)  # 80% train, 20% val
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

print(f"Train samples: {len(train_samples)}")
print(f"Val samples: {len(val_samples)}")

# Lưu file
output_dir = './lists/lists_Synapse'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    f.writelines(train_samples)

with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    f.writelines(val_samples)

print(f"✓ Tạo file thành công!")
print(f"  - {os.path.join(output_dir, 'train.txt')} ({len(train_samples)} samples)")
print(f"  - {os.path.join(output_dir, 'val.txt')} ({len(val_samples)} samples)")
