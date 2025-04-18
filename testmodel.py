import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # ➕ Thanh tiến trình
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

# ========== CẤU HÌNH ==========
DATA_PATH = 'Traindata_3vong'
MODEL_PATH = 'models19/model_epoch_11.keras'
# MODEL_PATH = 'modeltrainfull_1.keras'
IMAGE_SIZE = (66, 200)
BATCH_SIZE = 16
SAMPLE_PLOT = 30

# ========== 1. HÀM TẢI DỮ LIỆU ==========
def load_data(data_path):
    data = pd.read_csv(
        os.path.join(data_path, 'driving_log.csv'),
        names=['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    )
    data['Center'] = data['Center'].apply(lambda x: os.path.basename(x))
    data['Steering'] = data['Steering'].astype(float)
    return data

# ========== 2. XỬ LÝ ẢNH ==========
def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    img = img.astype(np.float32) / 255.0
    return img

# ========== 3. LOAD DỮ LIỆU ==========
print("🔍 Đang load dữ liệu từ driving_log.csv...")
df = load_data(DATA_PATH)

image_paths = df['Center'].tolist()
steerings = df['Steering'].tolist()

x_data = []
for filename in tqdm(image_paths, desc="📷 Đang xử lý ảnh"):
    full_path = os.path.join(DATA_PATH, 'IMG', filename)
    img = cv2.imread(full_path)
    if img is not None:
        x_data.append(preprocess_image(img))
    else:
        print(f"⚠️ Không thể đọc ảnh: {full_path}")

# ========== 3. CHUẨN BỊ TẬP TEST ==========
total_samples = len(steerings)
one_third = total_samples // 3  # lấy 1/3 số mẫu

# Chia theo thứ tự
X_train = x_data[:one_third]
y_train = steerings[:one_third]

X_valid = x_data[one_third:2 * one_third]
y_valid = steerings[one_third:2 * one_third]

X_test = x_data[2 * one_third:]
y_test = steerings[2 * one_third:]

x_data = np.array(X_test)
y_data = np.array(y_test).reshape(-1, 1)

print(f"✅ Dữ liệu đầu vào shape: {x_data.shape}, Dữ liệu nhãn: {y_data.shape}")

# ========== 4. LOAD MÔ HÌNH ==========
print(f"📦 Đang load mô hình từ: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ========== 5. ĐÁNH GIÁ ==========
loss = model.evaluate(x_data, y_data, batch_size=BATCH_SIZE)
print(f"✅ Loss (MSE): {loss:.6f}")

# ========== 6. DỰ ĐOÁN & LỖI ==========
predictions = model.predict(x_data)

# ➕ TÍNH TOÁN MSE & MAE
mse = mean_squared_error(y_data, predictions)
mae = mean_absolute_error(y_data, predictions)
print(f"📊 Mean Squared Error (sklearn): {mse:.6f}")
print(f"📊 Mean Absolute Error: {mae:.6f}")

# # ========== 7. HIỂN THỊ KẾT QUẢ ==========
# random_indices = np.random.choice(len(predictions), size=SAMPLE_PLOT, replace=False)
# random_predictions = predictions[random_indices]
# random_ground_truth = y_data[random_indices]
#
# fig, axs = plt.subplots(1, 2, figsize=(15, 6))
#
# # Scatter plot
# axs[0].scatter(range(SAMPLE_PLOT), random_predictions, label="Predicted", color='blue', s=60)
# axs[0].scatter(range(SAMPLE_PLOT), random_ground_truth, label="Actual", color='orange', s=60)
# axs[0].set_title("Scatter Plot: Predicted vs Actual")
# axs[0].set_xlabel("Sample Index")
# axs[0].set_ylabel("Steering Angle")
# axs[0].legend()
# axs[0].grid(True)
#
# # Line plot
# axs[1].plot(random_predictions, label="Predicted", marker='o', color='blue')
# axs[1].plot(random_ground_truth, label="Actual", marker='x', color='orange')
# axs[1].set_title("Line Plot: Predicted vs Actual")
# axs[1].set_xlabel("Sample Index")
# axs[1].set_ylabel("Steering Angle")
# axs[1].legend()
# axs[1].grid(True)
#
# plt.tight_layout()
# plt.show()

# ========== 8. DỰ ĐOÁN & ĐÁNH GIÁ TỪNG PHÂN ĐOẠN ==========
# print("📊 Đang đánh giá mô hình theo từng đoạn 3000 ảnh...")
#
# CHUNK_SIZE = 3275
# total_samples = len(x_data)
#
# for start_idx in range(0, total_samples, CHUNK_SIZE):
#     end_idx = min(start_idx + CHUNK_SIZE, total_samples)
#
#     x_chunk = x_data[start_idx:end_idx]
#     y_chunk = y_data[start_idx:end_idx]
#
#     preds_chunk = model.predict(x_chunk)
#
#     # Tính toán lỗi
#     mse_chunk = mean_squared_error(y_chunk, preds_chunk)
#     mae_chunk = mean_absolute_error(y_chunk, preds_chunk)
#
#     print(f"📦 Đánh giá từ ảnh {start_idx} đến {end_idx - 1}:")
#     print(f"    🔹 MSE: {mse_chunk:.6f}")
#     print(f"    🔹 MAE: {mae_chunk:.6f}")
#
#     # --- Vẽ biểu đồ ---
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))
#
#     # Scatter
#     axs[0].scatter(range(len(preds_chunk)), preds_chunk, label="Predicted", color='blue', s=10)
#     axs[0].scatter(range(len(y_chunk)), y_chunk, label="Actual", color='orange', s=10)
#     axs[0].set_title(f"Scatter Plot: Predicted vs Actual\n(Ảnh {start_idx} - {end_idx - 1})")
#     axs[0].set_xlabel("Sample Index")
#     axs[0].set_ylabel("Steering Angle")
#     axs[0].legend()
#     axs[0].grid(True)
#
#     # # Line
#     # axs[1].plot(preds_chunk, label="Predicted", color='blue')
#     # axs[1].plot(y_chunk, label="Actual", color='orange')
#     # axs[1].set_title("Line Plot")
#     # axs[1].set_xlabel("Sample Index")
#     # axs[1].set_ylabel("Steering Angle")
#     # axs[1].legend()
#     # axs[1].grid(True)
#
#     plt.suptitle(f"📈 Dự đoán & Đánh giá: Ảnh {start_idx} đến {end_idx - 1}", fontsize=16)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88)
#     plt.show()
#
#     if end_idx < total_samples:
#         input("👉 Nhấn Enter để tiếp tục batch tiếp theo...")
#     else:
#         print("✅ Đã đánh giá toàn bộ ảnh theo lô.")

plt.figure(figsize=(14, 6))
plt.scatter(range(len(y_data)), y_data, label='Actual', color='orange', s=20)
plt.scatter(range(len(predictions)), predictions, label='Predicted', color='blue', s=20, alpha=0.6)

plt.title("📊 Biểu đồ Scatter: Dự đoán vs Thực tế (Toàn bộ tập test)")
plt.xlabel("Chỉ số ảnh")
plt.ylabel("Góc lái (Steering Angle)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
