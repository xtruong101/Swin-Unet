# Hướng dẫn Training Swin UNet trên tập ACDC

## 1. Chuẩn bị dữ liệu

### 1.1 Cấu trúc dữ liệu ACDC

Dữ liệu cần được tổ chức như sau:
```
../data/ACDC/
├── ACDC_training_slices/        # File .h5 của slice (cho training)
│   ├── patient001_0.h5
│   ├── patient001_1.h5
│   ├── ...
│   └── patient100_10.h5
└── ACDC_training_volumes/       # File .h5 của volume (cho validation/test)
    ├── patient001.h5
    ├── patient002.h5
    └── ...
    └── patient100.h5
```

### 1.2 Format file .h5

Mỗi file H5 phải chứa 2 key:
- `image`: mảng numpy shape `(H, W)` cho slice hoặc `(D, H, W)` cho volume
- `label`: mảng numpy có cùng shape, giá trị nhãn (0-3 cho ACDC)

Nhãn ACDC:
- 0: Background
- 1: Right Ventricle (RV)
- 2: Myocardium (MYO)
- 3: Left Ventricle (LV)
- 4: (ignored class - dùng cho ignore_index trong CrossEntropyLoss)

### 1.3 Tách dataset

Dữ liệu được tách tự động như sau:
- **Training**: patient001 - patient080 (80 ca)
- **Validation**: patient081 - patient090 (10 ca)
- **Testing**: patient091 - patient100 (10 ca)

File `dataset_acdc.py` xử lý tự động sự phân chia này thông qua hàm `_get_ids()`.

---

## 2. Cài đặt và chuẩn bị môi trường

### 2.1 Cài đặt dependencies
```bash
pip install -r requirements.txt
```

**Các thư viện cần:**
- torch, torchvision
- numpy, scipy, h5py
- tensorboard, tensorboardX
- tqdm, timm, einops
- medpy, SimpleITK
- ml-collections

### 2.2 Kiểm tra GPU
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 3. Chạy Training

### 3.1 Command cơ bản
```bash
python train_acdc.py --dataset ACDC
```

### 3.2 Các tham số quan trọng

| Tham số | Giá trị mặc định | Mô tả |
|---------|-----------------|--------|
| `--dataset` | ACDC | Tên dataset (ACDC hoặc Synapse) |
| `--root_path` | ../data/ACDC | Đường dẫn đến thư mục dữ liệu |
| `--num_classes` | 4 | Số lớp (4 cho ACDC) |
| `--img_size` | 224 | Kích thước input ảnh (224x224) |
| `--batch_size` | 24 | Batch size cho training |
| `--max_iterations` | 30000 | Số iteration tối đa |
| `--max_epochs` | 150 | Số epoch tối đa |
| `--base_lr` | 0.01 | Learning rate ban đầu |
| `--seed` | 1234 | Random seed |
| `--n_gpu` | 1 | Số GPU sử dụng |
| `--deterministic` | 1 | Sử dụng deterministic training |

### 3.3 Ví dụ chạy với tham số custom
```bash
# Training với batch size nhỏ hơn
python train_acdc.py --dataset ACDC --batch_size 12 --max_iterations 20000

# Training với learning rate khác
python train_acdc.py --dataset ACDC --base_lr 0.005 --max_epochs 200

# Training với seed khác (để test robustness)
python train_acdc.py --dataset ACDC --seed 42 --batch_size 16
```

---

## 4. Cấu trúc Code chính

### 4.1 File `train_acdc.py`
- **Tác dụng**: Script chính để run training
- **Chức năng**:
  1. Parse arguments từ command line
  2. Set deterministic training (seed + cudnn)
  3. Khởi tạo model Swin UNet
  4. Gọi hàm `trainer_acdc()` để bắt đầu training

### 4.2 File `trainer_acdc.py`
- **Hàm `trainer_acdc(args, model, snapshot_path)`**:
  - Load dataset từ `ACDC_training_slices` (train) và `ACDC_training_volumes` (val)
  - Setup optimizer (SGD) và loss functions (CrossEntropy + Dice)
  - Vòng lặp training với:
    - Forward pass
    - Calculate loss
    - Backward pass + optimize
    - Learning rate scheduling (poly decay)
    - Validation mỗi 500 iterations
    - Save best model

### 4.3 File `dataset_acdc.py`
- **Class `BaseDataSets`**:
  - Load từ file H5
  - Tự động tách train/val/test theo patient ID
  - Áp dụng augmentation (RandomGenerator)

### 4.4 File `utils.py`
- **`DiceLoss`**: Loss function kết hợp Dice coefficient
- **`test_single_volume()`**: Hàm evaluate trên single volume
- **`calculate_metric_percase()`**: Tính Dice và HD95 metrics

---

## 5. Output và Monitoring

### 5.1 Cấu trúc thư mục output
```
../model/TU_ACDC224/TU/
├── log.txt                          # Log file
├── log/                             # TensorBoard logs
│   ├── events.out.tfevents.*
└── best_model.pth                  # Best model checkpoint
```

### 5.2 TensorBoard
Để monitor training:
```bash
tensorboard --logdir=../model/TU_ACDC224/TU/log/
```

Sau đó mở `http://localhost:6006` trên browser.

**Các metrics được track:**
- `info/lr`: Learning rate
- `info/total_loss`: Total loss
- `info/loss_ce`: CrossEntropy loss
- `info/val_*_dice`: Dice score cho từng class
- `info/val_*_hd95`: HD95 metric cho từng class
- `info/val_mean_dice`: Trung bình Dice trên tất cả classes

### 5.3 Log file content
```
[HH:MM:SS.mmm] iteration 1 : loss : 2.123, loss_ce: 2.456
[HH:MM:SS.mmm] iteration 500 : mean_dice : 0.654 mean_hd95 : 15.23
[HH:MM:SS.mmm] Best model | iteration 500 : mean_dice : 0.654 mean_hd95 : 15.23
```

---

## 6. Training workflow chi tiết

### Step 1: Data Preparation
```bash
# Đặt dữ liệu vào thư mục
mkdir -p ../data/ACDC
# Copy ACDC_training_slices/ và ACDC_training_volumes/ vào thư mục trên
```

### Step 2: Kiểm tra paths
```bash
# Xác nhận cấu trúc
ls -la ../data/ACDC/ACDC_training_slices/ | head
ls -la ../data/ACDC/ACDC_training_volumes/ | head
```

### Step 3: Chạy training
```bash
# Option 1: Default settings
python train_acdc.py --dataset ACDC

# Option 2: Nếu có GPU memory issues, giảm batch size
python train_acdc.py --dataset ACDC --batch_size 8

# Option 3: Quick test (iteration thấp)
python train_acdc.py --dataset ACDC --max_iterations 1000 --batch_size 12
```

### Step 4: Monitor progress
```bash
# Terminal 1: Xem real-time log
tail -f ../model/TU_ACDC224/TU/log.txt

# Terminal 2: Launch TensorBoard
tensorboard --logdir=../model/TU_ACDC224/TU/log/
```

### Step 5: Inference sau khi training
```bash
python test_acdc.py --model_path ../model/TU_ACDC224/TU/best_model.pth
```

---

## 7. Lỗi thường gặp và cách khắc phục

### 7.1 "FileNotFoundError: .../ACDC_training_slices"
**Nguyên nhân**: Dữ liệu chưa được đặt đúng vị trí
**Giải pháp**: 
```bash
# Kiểm tra cấu trúc
find ../data/ACDC -type d | head -5
```

### 7.2 "CUDA out of memory"
**Giải pháp**: Giảm batch size
```bash
python train_acdc.py --dataset ACDC --batch_size 4 --img_size 192
```

### 7.3 "No module named 'vit_seg_modeling'"
**Nguyên nhân**: Import sai module (đã được fix trong train_acdc.py)
**Giải pháp**: File đã được sửa, dùng version hiện tại

### 7.4 Model không hội tụ
**Giải pháp**:
- Tăng learning rate: `--base_lr 0.02`
- Giảm learning rate: `--base_lr 0.005`
- Tăng số iterations: `--max_iterations 50000`

---

## 8. Advanced Configuration

### 8.1 Multi-GPU training (nếu có)
```bash
# Hiện tại code hỗ trợ single GPU
# Để multi-GPU, cần sửa trainer_acdc.py thêm:
# if args.n_gpu > 1:
#     model = nn.DataParallel(model)
```

### 8.2 Mixed Precision Training
```python
# Thêm vào trainer_acdc.py nếu muốn tăng tốc độ:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Trong vòng training loop:
with autocast():
    outputs = model(volume_batch)
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 8.3 Custom augmentation
Sửa `RandomGenerator` trong `dataset_acdc.py`:
```python
class RandomGenerator(object):
    def __init__(self, output_size, low_res=False):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        
        # Custom augmentation here
        
        return {'image': image, 'label': label}
```

---

## 9. Số liệu tham khảo

### 9.1 Expected Results trên ACDC
Sau ~30k iterations:
- **Dice Score**: ~0.85-0.90
- **HD95**: ~10-15 mm (tùy class)

### 9.2 Training time
- GPU V100: ~5-6 giờ cho 30k iterations (batch_size=24)
- GPU A100: ~2-3 giờ
- GPU RTX 3090: ~3-4 giờ

---

## 10. Các files quan trọng từ TransUnet_acdc_supplymentary

Thư mục `TransUnet_acdc_supplymentary/` chứa file gốc từ TransUNet. Những file này đã được refactor để:

1. **train_acdc.py**: 
   - Từ: Dùng `vit_seg_modeling` (TransUNet VisionTransformer)
   - Sang: Dùng `vision_transformer` (Swin UNet)
   - Import được sửa để dùng SwinUnet thay vì ViT

2. **trainer_acdc.py**:
   - Import `test_single_volume` từ utils
   - Fix lỗi undefined `testloader`
   - Thêm logging configuration

3. **dataset_acdc.py**:
   - Tương tự giữa TransUNet và Swin UNet
   - Hỗ trợ cả slice-based (training) và volume-based (validation) loading

**Tip**: Nếu gặp lỗi, so sánh với files trong `TransUnet_acdc_supplymentary/` để hiểu cấu trúc gốc.

---

## 11. Checklist trước khi training

- [ ] Dữ liệu H5 đã được chuẩn bị và đặt trong `../data/ACDC/`
- [ ] Dependencies được cài: `pip install -r requirements.txt`
- [ ] GPU khả dụng: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Thư mục model output tồn tại hoặc script sẽ tạo tự động
- [ ] Batch size phù hợp với GPU memory
- [ ] Learning rate được chọn hợp lý
- [ ] TensorBoard được cài: `pip install tensorboard tensorboardX`

---

## 12. Kết luận

Để train Swin UNet trên ACDC:
```bash
# 1. Chuẩn bị dữ liệu
mkdir -p ../data/ACDC
# Copy ACDC_training_slices và ACDC_training_volumes

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Run training
python train_acdc.py --dataset ACDC --batch_size 24 --max_iterations 30000

# 4. Monitor trên TensorBoard
tensorboard --logdir=../model/TU_ACDC224/TU/log/
```

Happy training! 🚀
