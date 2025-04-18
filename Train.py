# File train.py này huấn luyện mô hình lái xe tự động sử dụng CNN theo kiến trúc NVIDIA.
# Đọc dữ liệu từ driving_log.csv, huấn luyện với ảnh camera và lưu mô hình .keras sau mỗi epoch.

# 1. Import thư viện
import os
import cv2
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda, Cropping2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle   # Xáo trộn dữ liệu

from Utlis import loadData, batchGen  # Hàm loadData và batchGen được định nghĩa trong file Utlis.py

# 2. Các hàm cần thiết
# 2.1 Hàm load dữ liệu
def load_data(data_path):
    """
    Tải và xử lý dữ liệu lái xe từ file driving_log.csv trong thư mục chỉ định.
    """
    data = pd.read_csv(
        os.path.join(data_path, 'driving_log.csv'),
        names=['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    )
    data['Center'] = data['Center'].apply(lambda x: os.path.basename(x))
    data['Left'] = data['Left'].apply(lambda x: os.path.basename(x))
    data['Right'] = data['Right'].apply(lambda x: os.path.basename(x))

    data['Steering'] = data['Steering'].astype(float)

    return data

# 2.2 Hàm cân bằng dữ liệu
def balance_data(data, num_bins=50, samples_per_bin=700, show_hist=True):
    """
    Cân bằng dữ liệu góc lái bằng cách giảm số lượng ảnh ở các vùng xuất hiện quá nhiều (thường là gần 0).
    """
    hist, bins = np.histogram(data['Steering'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5

    if show_hist:
        plt.figure(figsize=(10, 5))
        plt.bar(center, hist, width=0.05)
        plt.title('Histogram trước khi cân bằng')
        plt.xlabel('Steering Angle')
        plt.ylabel('Số lượng ảnh')
        plt.show()

    balanced_data = []

    for i in range(num_bins):
        bin_data = data[(data['Steering'] >= bins[i]) & (data['Steering'] < bins[i+1])]
        bin_count = len(bin_data)

        if bin_count > samples_per_bin:
            bin_data = bin_data.sample(samples_per_bin)
        # Nếu ít hơn, giữ nguyên

        balanced_data.append(bin_data)

    data_balanced = pd.concat(balanced_data).reset_index(drop=True)

    if show_hist:
        hist_bal, _ = np.histogram(data_balanced['Steering'], num_bins)
        plt.figure(figsize=(10, 5))
        plt.bar(center, hist_bal, width=0.05)
        plt.title('Histogram sau khi cân bằng')
        plt.xlabel('Steering Angle')
        plt.ylabel('Số lượng ảnh')
        plt.show()

    print(f"✅ Dữ liệu sau khi cân bằng: {len(data_balanced)} ảnh (giới hạn {samples_per_bin} ảnh mỗi bin)")
    return data_balanced

# 2.3 Hàm tạo mô hình
# Hàm này xây dựng mô hình CNN dựa trên kiến trúc NVIDIA
def build_model_CNN():
    """
    Xây dựng mô hình CNN cho bài toán lái xe tự động.
    """
    model = Sequential()
    model.add(Input(shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model

# 2.4 Hàm huấn luyện mô hình
# Hàm này huấn luyện mô hình và lưu mô hình sau mỗi epoch vào thư mục models/
def train_model(model, imagesPath, steerings, batch_size=32, epochs=10, save_dir='TH1'):

    # Tạo thư mục lưu mô hình nếu chưa có
    os.makedirs(save_dir, exist_ok=True)

    # Shuffle dữ liệu
    imagesPath, steerings = shuffle(imagesPath, steerings, random_state=42)

    # Tạo generator
    train_generator = batchGen(
        imagesPath, steerings,
        batchSize=batch_size,
        trainFlag=True
    )

    # Callback lưu mô hình sau mỗi epoch
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_dir, 'model_epoch{epoch:02d}.keras'),
        monitor='loss',
        save_best_only=False,  # lưu tất cả các epoch
        save_weights_only=False,
        verbose=1
    )

    # Huấn luyện mô hình
    history = model.fit(
        x=train_generator,
        steps_per_epoch=len(imagesPath) // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint]
    )

    # Vẽ đồ thị loss
    plt.plot(history.history['loss'], label='loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    with open(os.path.join(save_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    print(f"✅ Mô hình sau mỗi epoch đã được lưu vào thư mục: {save_dir}")
    print(f"✅ File history.pkl đã được lưu, bạn có thể dùng lại để phân tích mà không cần train lại.")
    return history

##############################
def check_dataset(data, steering_col='Steering', bins=25, show_stats=True, show_plot=True):

    steering = data[steering_col]
    total = len(steering)

    if show_stats:
        print(f"✅ Tổng số ảnh: {total}")
        print(f"Số ảnh |góc lái| > 0.5: {np.sum(abs(steering) > 0.5)}")
        print(f"Số ảnh |góc lái| ≤ 0.1: {np.sum(abs(steering) <= 0.1)}")
        print(f"Tỷ lệ rẽ gắt: {100 * np.sum(abs(steering) > 0.5) / total:.2f}%")

    if show_plot:
        hist, bins_range = np.histogram(steering, bins)
        center = (bins_range[:-1] + bins_range[1:]) * 0.5
        plt.figure(figsize=(10, 5))
        plt.bar(center, hist, width=0.05, color='teal', edgecolor='black')
        plt.title("Phân bố góc lái ")
        plt.xlabel("Steering Angle")
        plt.ylabel("Số lượng ảnh")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# 3. Chương trình chính
# Chương trình chính
# Bước 1: Đọc dữ liệu từ file driving_log.csv
# Bước 2: Tiền xử lý dữ liệu
# Bước 3: Tăng cường dữ liệu
# Bước 4: Xây dựng mô hình CNN
# Bước 5: Huấn luyện mô hình

if __name__ == '__main__':
    # Đường dẫn đến thư mục chứa dữ liệu
    data_path = r'C:\Users\ADMIN\Desktop\Selfdrivingcar\Traindata_3vong'

    # Bước 1: Đọc dữ liệu từ file driving_log.csv
    data = load_data(data_path)

    # Kiểm tra dữ liệu
    # check_dataset(data, bins=25, show_stats=True, show_plot=True)

    # Bước 2: Tiền xử lý dữ liệu
    # (Có thể thêm các bước tiền xử lý khác nếu cần thiết)

    # Bước 3: Tăng cường dữ liệu
    # (Có thể thêm các bước tăng cường khác nếu cần thiết)

    # Cân bằng dữ liệu
    data = balance_data(data, num_bins=50, samples_per_bin=500, show_hist=True)


    # Bước 4: Xây dựng mô hình CNN
    model = build_model_CNN()

    # Bước 5: Huấn luyện mô hình
    imagesPath, steerings = loadData(data_path, data)   # Tải đường dẫn ảnh và góc lái
    history = train_model(model, imagesPath, steerings, batch_size=32, epochs=30, save_dir='TH3')

