# Thư viện Utlis.py hỗ trợ xử lý ảnh cho mô hình lái xe tự động: gồm tiền xử lý, tăng cường dữ liệu và tạo batch huấn luyện.
# Dùng kèm với train.py để đọc ảnh từ CSV, biến đổi và cấp dữ liệu cho mô hình học sâu.

# Import các thư viện cần thiết
import os
import cv2
import random
import numpy as np
import matplotlib.image as mpimg
from imgaug import augmenters as iaa


# Hàm này dùng để đọc dữ liệu từ file CSV và trả về danh sách đường dẫn ảnh và giá trị góc lái
def loadData(path, data):

    imagesPath = []  # Khởi tạo danh sách rỗng để lưu đường dẫn đến các ảnh
    steerings = []   # Khởi tạo danh sách rỗng để lưu giá trị góc lái

    for i in range(len(data)):
        # Lặp qua từng hàng trong DataFrame data

        image_file = data.iloc[i]['Center']
        # Lấy tên file ảnh từ cột 'Center' của hàng thứ i trong DataFrame
        # data.iloc[i] truy cập hàng thứ i, sau đó ['Center'] truy cập giá trị cột 'Center'

        angle = data.iloc[i]['Steering']
        # angle = float(data.iloc[i]['Steering']) # chuyển đổi thành số thực
        # Lấy giá trị góc lái từ cột 'Steering'

        full_path = os.path.normpath(os.path.join(path, 'IMG', image_file))
        # ❗ Sửa lại: Ghép đường dẫn đầy đủ đến ảnh với cấu trúc
        #  Mục đích chuyển đổi giữa các thu mục khác nhau

        imagesPath.append(full_path)  # Thêm đường dẫn đầy đủ vào danh sách imagesPath
        steerings.append(angle)       # Thêm giá trị góc lái vào danh sách steerings

    return np.array(imagesPath), np.array(steerings)

# Hàm này dùng để xử lý ảnh đầu vào cho mô hình học sâu
def preProcess(img):
    img = img[60:135, :, :]  # Cắt bỏ phần trên (bầu trời) và dưới (nắp capo)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Chuyển ảnh sang không gian màu YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)      # Làm mờ nhẹ để giảm nhiễu
    img = cv2.resize(img, (200, 66))            # Resize về kích thước đầu vào input_shape = (66, 200, 3)  # Chiều cao, chiều rộng, số kênh màu của NVIDIA
    img = img / 255.0                           # Chuẩn hóa ảnh về [0, 1]
    return img


# Hàm này dùng để tăng cường dữ liệu
def augmentImage(imgPath, steering):

    # Đọc ảnh từ đường dẫn (RGB)
    img = mpimg.imread(imgPath)

    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)

    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)

    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

# Hàm này dùng để tạo ra các batch dữ liệu cho mô hình
def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    # imagesPath: danh sách đường dẫn đến các file ảnh
    # steeringList: danh sách các giá trị góc lái tương ứng
    # batchSize: kích thước của mỗi batch
    # trainFlag: cờ đánh dấu chế độ huấn luyện (True) hoặc kiểm định (False)

    while True:
        # Vòng lặp vô hạn để liên tục tạo ra các batch mới khi cần

        imgBatch = []  # Khởi tạo danh sách rỗng để lưu các ảnh trong batch
        steeringBatch = []  # Khởi tạo danh sách rỗng để lưu các giá trị góc lái trong batch

        for i in range(batchSize):
            # Lặp qua số lượng mẫu cần cho một batch

            index = random.randint(0, len(imagesPath) - 1)
            # Chọn ngẫu nhiên một chỉ số trong phạm vi dữ liệu có sẵn

            if trainFlag:
                # Nếu đang ở chế độ huấn luyện, thực hiện augmentation (tăng cường dữ liệu)
                img, steering = augmentImage(imagesPath[index], steeringList[index])
                # Hàm augmentImage thực hiện biến đổi ảnh và điều chỉnh giá trị góc lái tương ứng
            else:
                # Nếu đang ở chế độ kiểm định, sử dụng ảnh gốc không biến đổi
                img = mpimg.imread(imagesPath[index])  # Đọc ảnh từ đường dẫn
                steering = steeringList[index]  # Lấy giá trị góc lái tương ứng

            img = preProcess(img)
            # img = preprocess_image_vgg(img)  # Tiền xử lý ảnh cho mô hình VGG16
            # Tiền xử lý ảnh (có thể bao gồm thay đổi kích thước, chuẩn hóa, chuyển đổi không gian màu...)

            imgBatch.append(img)  # Thêm ảnh đã xử lý vào batch
            steeringBatch.append(steering)  # Thêm giá trị góc lái vào batch

        yield (np.asarray(imgBatch), np.asarray(steeringBatch))
        # Trả về một cặp mảng NumPy: mảng các ảnh và mảng các giá trị góc lái tương ứng
        # Sử dụng yield để tạo ra một generator, giúp tiết kiệm bộ nhớ khi xử lý dữ liệu lớn


