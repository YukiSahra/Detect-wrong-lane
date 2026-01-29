# Import thư viện argparse để xử lý dòng lệnh
import argparse


def parse_arguments():
    # Khởi tạo ArgumentParser với mô tả về chức năng của chương trình
    parser = argparse.ArgumentParser(description="Process a video with the YOLO model.")
    
    # Tham số --video: Đường dẫn tới file video đầu vào
    # Mặc định là video bentre.mp4
    parser.add_argument(
        '--video',
        type=str,
        default='../video/Video/bentre.mp4',
        help='Path to the input video file'
    )
    
    # Tham số --model: Đường dẫn tới file model YOLO đã train
    # Mặc định là best.pt ở thư mục imgsz_224
    parser.add_argument(
        '--model',
        type=str,
        default='../train/imgsz_224/best.pt',
        help='Path to the YOLO model file'
    )
    
    # Tham số --size: Kích thước ảnh (chiều rộng và chiều cao) đầu vào cho model
    # Mặc định là 224x224 pixels
    # Cấu trúc: --size WIDTH HEIGHT (ví dụ: --size 224 224)
    parser.add_argument(
        '--size',
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=('WIDTH', 'HEIGHT'),
        help='Image size as width height (e.g. --image 224 224)'
    )
    
    # Phân tích các tham số từ dòng lệnh và trả về đối tượng chứa các tham số
    return parser.parse_args()