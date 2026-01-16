"""
config_utils.py
===============
Utility script để hiển thị config training trước khi bắt đầu
Hỗ trợ Swin-Unet training
"""

import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not installed. Using basic print.")

# ============================================================================
# CLASS HIỂN THỊ CONFIG
# ============================================================================

class ConfigDisplay:
    """Hiển thị config training một cách đẹp và rõ ràng"""
    
    def __init__(self, use_rich=True):
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console = Console()
        else:
            self.console = None
    
    def print_header(self, title):
        """In header"""
        if self.use_rich:
            self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        else:
            print(f"\n{'='*80}")
            print(f"{title:^80}")
            print(f"{'='*80}")
    
    def print_section(self, section_name, items_dict):
        """In một section config"""
        if self.use_rich:
            # Sử dụng Rich table
            table = Table(
                title=section_name,
                style="cyan",
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("Parameter", style="green", width=30)
            table.add_column("Value", style="yellow", width=50)
            
            for key, value in items_dict.items():
                table.add_row(str(key), str(value))
            
            self.console.print(table)
        else:
            # Sử dụng print bình thường
            print(f"\n{section_name}")
            print("-" * 80)
            for key, value in items_dict.items():
                print(f"  {key:<28} {value}")
    
    def print_success(self, message):
        """In thông báo thành công"""
        if self.use_rich:
            self.console.print(f"[bold green] {message}[/bold green]")
        else:
            print(f" {message}")
    
    def print_warning(self, message):
        """In thông báo cảnh báo"""
        if self.use_rich:
            self.console.print(f"[bold yellow]  {message}[/bold yellow]")
        else:
            print(f"  {message}")
    
    def print_error(self, message):
        """In thông báo lỗi"""
        if self.use_rich:
            self.console.print(f"[bold red] {message}[/bold red]")
        else:
            print(f" {message}")
    
    def print_footer(self, message):
        """In footer"""
        if self.use_rich:
            footer_text = Text(message, style="bold green")
            self.console.print(Panel(footer_text, border_style="green"))
        else:
            print(f"\n{'='*80}")
            print(f"{message:^80}")
            print(f"{'='*80}\n")

# ============================================================================
# HÀM KIỂM TRA CONFIG
# ============================================================================

def check_paths(args):
    """Kiểm tra các đường dẫn"""
    issues = []
    
    # Kiểm tra list_dir
    list_dir = args.list_dir
    if not os.path.exists(list_dir):
        issues.append(f"List directory không tồn tại: {list_dir}")
    else:
        # Kiểm tra train.txt
        train_txt = os.path.join(list_dir, "train.txt")
        if not os.path.exists(train_txt):
            issues.append(f"train.txt không tồn tại: {train_txt}")
        else:
            with open(train_txt, 'r') as f:
                train_lines = len(f.readlines())
            if train_lines == 0:
                issues.append("train.txt rỗng")
        
        # Kiểm tra val.txt
        val_txt = os.path.join(list_dir, "val.txt")
        if not os.path.exists(val_txt):
            issues.append(f"val.txt không tồn tại: {val_txt}")
        else:
            with open(val_txt, 'r') as f:
                val_lines = len(f.readlines())
    
    # Kiểm tra root_path
    root_path = args.root_path
    if not os.path.exists(root_path):
        issues.append(f"Root path không tồn tại: {root_path}")
    
    # Kiểm tra output_dir có thể tạo được không
    output_dir = args.output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        issues.append(f"Không thể tạo output directory: {str(e)}")
    
    # Kiểm tra config file
    if args.cfg and not os.path.exists(args.cfg):
        issues.append(f"Config file không tồn tại: {args.cfg}")
    
    return issues

def check_parameters(args):
    """Kiểm tra các tham số training"""
    issues = []
    
    # Kiểm tra num_classes
    if hasattr(args, 'num_classes'):
        if args.num_classes != 9:
            issues.append(f"num_classes = {args.num_classes}, nên là 9 cho Synapse")
    
    # Kiểm tra batch_size
    if hasattr(args, 'batch_size'):
        if args.batch_size < 1 or args.batch_size > 128:
            issues.append(f"batch_size = {args.batch_size} có thể quá lớn")
    
    # Kiểm tra img_size
    if hasattr(args, 'img_size'):
        if args.img_size != 224:
            issues.append(f"img_size = {args.img_size}, khuyến nghị 224")
    
    # Kiểm tra base_lr
    if hasattr(args, 'base_lr'):
        if args.base_lr <= 0 or args.base_lr > 1.0:
            issues.append(f"base_lr = {args.base_lr} không hợp lệ")
    
    return issues

# ============================================================================
# HÀM HIỂN THỊ CONFIG
# ============================================================================

def display_training_config(args, config=None):
    """
    Hiển thị toàn bộ config training
    
    Args:
        args: Namespace từ argparse
        config: Config object từ get_config() (tùy chọn)
    
    Returns:
        bool: True nếu config hợp lệ, False nếu có lỗi
    """
    
    display = ConfigDisplay(use_rich=RICH_AVAILABLE)
    
    # ========================================================================
    # PHẦN 1: HEADER
    # ========================================================================
    if display.use_rich:
        display.console.clear()
    
    print("\n" + "="*80)
    print("| "*40)
    print("| SWIN-UNET TRAINING CONFIGURATION |")
    print("| "*40)
    print("="*80)
    
    # ========================================================================
    # PHẦN 2: DATASET CONFIGURATION
    # ========================================================================
    dataset_config = {
        "Dataset": args.dataset if hasattr(args, 'dataset') else "N/A",
        "Root Path": args.root_path if hasattr(args, 'root_path') else "N/A",
        "List Directory": args.list_dir if hasattr(args, 'list_dir') else "N/A",
        "Output Directory": args.output_dir if hasattr(args, 'output_dir') else "N/A",
    }
    display.print_section(" DATASET CONFIGURATION", dataset_config)
    
    # ========================================================================
    # PHẦN 3: MODEL CONFIGURATION
    # ========================================================================
    model_config = {
        "Config File": args.cfg if hasattr(args, 'cfg') else "N/A",
        "Image Size": f"{args.img_size}x{args.img_size}" if hasattr(args, 'img_size') else "N/A",
        "Number of Classes": args.num_classes if hasattr(args, 'num_classes') else "N/A",
    }
    display.print_section("🔧 MODEL CONFIGURATION", model_config)
    
    # ========================================================================
    # PHẦN 4: TRAINING PARAMETERS
    # ========================================================================
    training_params = {
        "Batch Size": args.batch_size if hasattr(args, 'batch_size') else "N/A",
        "Number of GPUs": args.n_gpu if hasattr(args, 'n_gpu') else "N/A",
        "Max Epochs": args.max_epochs if hasattr(args, 'max_epochs') else "N/A",
        "Base Learning Rate": f"{args.base_lr:.5f}" if hasattr(args, 'base_lr') else "N/A",
        "Weight Decay": "0.0001",
    }
    display.print_section("📈 TRAINING PARAMETERS", training_params)
    
    # ========================================================================
    # PHẦN 5: MISC SETTINGS
    # ========================================================================
    misc_config = {
        "Number of Workers": args.num_workers if hasattr(args, 'num_workers') else "4",
        "Evaluation Interval": args.eval_interval if hasattr(args, 'eval_interval') else "1",
        "Random Seed": args.seed if hasattr(args, 'seed') else "1234",
        "Deterministic": args.deterministic if hasattr(args, 'deterministic') else "1",
    }
    display.print_section("  MISC SETTINGS", misc_config)
    
    # ========================================================================
    # PHẦN 6: KIỂM TRA HỢPLỆ
    # ========================================================================
    print("\n" + "="*80)
    print(" KIỂM TRA CẤU HÌNH")
    print("="*80)
    
    # Kiểm tra đường dẫn
    path_issues = check_paths(args)
    param_issues = check_parameters(args)
    
    all_issues = path_issues + param_issues
    
    if not all_issues:
        display.print_success("Tất cả đường dẫn và tham số đều hợp lệ!")
        display.print_footer(" SẴN SÀNG TRAIN - Bắt đầu training ngay!")
        return True
    else:
        if path_issues:
            print("\n PATH ISSUES:")
            for issue in path_issues:
                display.print_error(issue)
        
        if param_issues:
            print("\n  PARAMETER WARNINGS:")
            for issue in param_issues:
                display.print_warning(issue)
        
        if any("không tồn tại" in issue or "lỗi" in issue.lower() for issue in all_issues):
            print("\n" + "="*80)
            print(" LỖI NGHIÊM TRỌNG - KHÔNG THỂ TRAIN")
            print("Vui lòng sửa các lỗi trên trước khi training")
            print("="*80 + "\n")
            return False
        else:
            print("\n" + "="*80)
            print("  CÓ CẢNH BÁO - Hãy kiểm tra kỹ trước khi training")
            print("="*80 + "\n")
            return True

# ============================================================================
# HÀM GỌISATTU CHƯƠNG TRÌNH CHÍNH
# ============================================================================

def validate_before_training(args, config=None):
    """
    Hàm gọi từ train.py để kiểm tra và hiển thị config
    
    Cách sử dụng trong train.py:
    
        from config_utils import validate_before_training
        
        if __name__ == "__main__":
            # ... parse args ...
            config = get_config(args)
            
            # ← Thêm dòng này
            if not validate_before_training(args, config):
                sys.exit(1)
            
            net = ViT_seg(config, ...)
            trainer_synapse(args, net, args.output_dir)
    """
    
    valid = display_training_config(args, config)
    
    if not valid:
        print("\n⏹  Training stopped due to configuration errors")
        return False
    
    return True

# ============================================================================
# MAIN (cho debug)
# ============================================================================

if __name__ == "__main__":
    # Test script này
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Synapse')
    parser.add_argument('--root_path', default='/content/project_TransUNet/data/Synapse/train_npz')
    parser.add_argument('--list_dir', default='./lists/lists_Synapse')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--cfg', default='configs/swin_tiny_patch4_window7_224_lite.yaml')
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--base_lr', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--deterministic', type=int, default=1)
    
    args = parser.parse_args()
    
    # Hiển thị config
    validate_before_training(args)