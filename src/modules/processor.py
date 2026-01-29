# Các thư viện sử dụng
import cv2  # Thư viện xử lý ảnh/video
from . import object_counter  # Module đếm và theo dõi objects
from time import time  # Để tính toán thời gian và FPS

# Số frame đầu tiên để warm-up (không tính vào FPS trung bình)
FPS_WARMUP_FRAMES = 5


def setup_object_counter(model_names, number_lane, lanes):
    """
    Khởi tạo và cấu hình bộ đếm objects để theo dõi phương tiện.
    
    Tham số:
        model_names: Danh sách tên các class (lớp) đối tượng
        number_lane: Số lượng làn đường cần theo dõi
        lanes: Tọa độ các điểm xác định làn đường
    
    Trả về:
        counter: Đối tượng ObjectCounter được cấu hình sẵn
    """
    counter = object_counter.ObjectCounter()
    # Cài đặt các tham số để theo dõi objects
    counter.set_args(
        view_img=True,  # Hiển thị ảnh khi xử lý
        reg_pts=lanes,  # Các điểm tạo vùng theo dõi (làn đường)
        classes_names=dict(model_names),  # Tên các lớp đối tượng
        draw_tracks=True,  # Vẽ đường đi của objects
        line_thickness=2,  # Độ dày đường vẽ
        region_thickness=1,  # Độ dày vùng làn đường
        track_thickness=1,  # Độ dày đường theo dõi
        region_lane=number_lane  # Số lượng làn đường
    )
    return counter


def process_video_frames(cap, model, video_writer, object_count, fps_warmup_frames, img_size):
    """
    Xử lý từng frame của video: phát hiện objects, theo dõi, và tính toán FPS.
    
    Tham số:
        cap: Video capture object từ OpenCV
        model: Mô hình YOLO để phát hiện và theo dõi objects
        video_writer: Writer để lưu video output
        object_count: ObjectCounter object để đếm và theo dõi
        fps_warmup_frames: Số frame để warm-up trước khi tính FPS
        img_size: Kích thước ảnh input cho mô hình
    
    Trả về:
        average_fps: FPS trung bình sau khi loại bỏ frame warm-up
        min_fps: FPS tối thiểu
        max_fps: FPS tối đa
        average_inference_time: Thời gian inference trung bình (ms)
    """
    # Biến lưu trữ thông tin FPS
    total_fps = 0  # Tổng FPS để tính trung bình
    frame_count = 0  # Số frame đã xử lý
    min_fps = float("inf")  # FPS tối thiểu
    max_fps = 0  # FPS tối đa
    inference_time_ms = 0  # Tổng thời gian inference

    # Vòng lặp xử lý từng frame cho đến khi video kết thúc
    while cap.isOpened():
        success, frame = cap.read()  # Đọc frame tiếp theo
        if not success:  # Nếu không đọc được frame (video kết thúc)
            print("Video processing completed.")
            break

        # Bắt đầu tính toán thời gian frame này
        frame_start_time = time()

        # Sử dụng mô hình YOLOv8 để phát hiện và theo dõi objects
        # persist=True: giữ ID của objects qua các frame
        # conf=0.1: confidence threshold
        tracks = model.track(frame, persist=True, show=False, imgsz=img_size, conf=0.1)

        # Đếm objects và vẽ lên frame
        frame, current_fps = object_count.start_counting(frame, tracks, frame_start_time)

        # Lấy thời gian inference từ mô hình
        speed = tracks[0].speed
        inference_time_ms += speed['inference']

        # Chỉ tính FPS sau khi đủ frame warm-up (bỏ qua frame đầu tiên)
        if frame_count >= fps_warmup_frames:
            total_fps += current_fps
            min_fps = min(min_fps, current_fps)
            max_fps = max(max_fps, current_fps)

        # In FPS hiện tại
        print(f"FPS: {current_fps:.2f}")

        # Lưu frame đã xử lý vào video output
        video_writer.write(frame)
        frame_count += 1

        # Nhấn 'q' để dừng xử lý
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Tính toán các chỉ số thống kê
    # Tính FPS trung bình (không tính frame warm-up)
    average_fps = total_fps / (frame_count - fps_warmup_frames) if frame_count > fps_warmup_frames else 0
    # Tính thời gian inference trung bình
    average_inference_time = inference_time_ms / frame_count
    

