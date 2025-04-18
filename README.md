# Selfdrivingcar
Project deep learning dự đoán góc xe tự lái bằng Python + Keras.

## 🖼️ Dữ liệu đầu vào

- **Nguồn**: Udacity Self-Driving Car Simulator
- **Dạng ảnh**: RGB/Grayscale (100x100 hoặc 160x320)
- **Dữ liệu nhãn**: `driving_log.csv` với góc lái (steering angle)

## 🧠 Mô hình học sâu

- Mô hình sử dụng kiến trúc **CNN** dựa theo mạng NVIDIA.
- Hàm mất mát: `Mean Squared Error (MSE)`
- Tối ưu hóa bằng `Adam Optimizer`
- Có thể lựa chọn các mô hình nhẹ hơn như: `ResNet18`, `EfficientNet`, `MobileNetV2`.

## 📈 Đánh giá

- Metrics: `MSE`, `MAE`, biểu đồ loss/accuracy theo epoch
- Có thể kiểm tra performance bằng cách chạy mô hình trong simulator.

## 🚀 Triển khai

1. Thu thập dữ liệu bằng Udacity simulator
2. Huấn luyện mô hình:
    ```bash
    python train.py --data_dir data/ --epochs 20
    ```
3. Kiểm tra mô hình:
    ```bash
    python test.py --model models/model_final.keras
    ```
4. Chạy mô hình trực tiếp trên simulator:
    ```bash
    python drive.py
    ```

## 📌 Yêu cầu hệ thống

- Python ≥ 3.8
- TensorFlow hoặc PyTorch
- OpenCV, NumPy, Pandas, Matplotlib
- Udacity Simulator (phiên bản Desktop)

## 👨‍💻 Tác giả

