# Skin Pigmentation Analyzer

Hệ thống phát hiện và đánh giá đốm sắc tố trên da mặt sử dụng thị giác máy tính và học sâu.

## Tổng quan

Skin Pigmentation Analyzer là một hệ thống phát hiện và đánh giá mức độ đốm sắc tố trên da mặt thông qua ảnh chụp từ camera phổ thông. Hệ thống kết hợp các kỹ thuật xử lý ảnh truyền thống và các mô hình học sâu để cung cấp kết quả chính xác và tin cậy.

### Tính năng

- Phát hiện khuôn mặt trong ảnh đầu vào
- Tiền xử lý ảnh khuôn mặt để tăng cường chất lượng
- Phát hiện đốm sắc tố sử dụng cả phương pháp xử lý ảnh truyền thống và học sâu
- Trích xuất đặc trưng từ đốm sắc tố (kích thước, màu sắc, hình dạng)
- Đánh giá mức độ nghiêm trọng của đốm sắc tố
- Phân loại đốm sắc tố thành các nhóm khác nhau
- Hiển thị kết quả phân tích bằng hình ảnh trực quan

## Cấu trúc dự án

```
skin_pigmentation_analyzer/
├── data/                    # Thư mục dữ liệu
│   ├── raw/                 # Dữ liệu thô
│   └── processed/           # Dữ liệu đã xử lý
├── models/                  # Mô hình học sâu
│   ├── pigmentation_model.py # Định nghĩa các mô hình học sâu
│   └── checkpoints/         # Checkpoint của mô hình đã huấn luyện
├── utils/                   # Các module tiện ích
│   ├── image_preprocessing.py # Tiền xử lý ảnh
│   ├── feature_extraction.py # Trích xuất đặc trưng
│   └── pigmentation_analyzer.py # Phân tích đốm sắc tố
├── scripts/                 # Script cho các tác vụ
│   ├── data_preparation.py  # Chuẩn bị dữ liệu
│   └── train_models.py      # Huấn luyện mô hình
├── notebooks/               # Jupyter notebooks cho phân tích và thử nghiệm
├── app.py                   # Ứng dụng chính
└── requirements.txt         # Dependencies
```

## Cài đặt

### Yêu cầu

- Python 3.8+
- OpenCV
- TensorFlow 2.x
- dlib
- face-recognition
- scikit-image
- scikit-learn
- và các thư viện khác được liệt kê trong `requirements.txt`

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Hướng dẫn sử dụng

### Phân tích đốm sắc tố từ ảnh

```bash
python app.py --image path/to/image.jpg --output_dir path/to/output
```

### Phân tích đốm sắc tố từ thư mục ảnh

```bash
python app.py --image_dir path/to/images --output_dir path/to/output
```

### Tùy chọn khác

```bash
python app.py --help
```

## Quy trình làm việc

1. **Tiền xử lý ảnh**:
   - Phát hiện khuôn mặt
   - Cắt và căn chỉnh khuôn mặt
   - Tăng cường độ tương phản
   - Chuẩn hóa ảnh

2. **Phát hiện đốm sắc tố**:
   - Sử dụng mô hình phân đoạn để xác định vùng đốm sắc tố
   - Hoặc sử dụng phương pháp xử lý ảnh truyền thống

3. **Trích xuất đặc trưng**:
   - Tính toán các thuộc tính của đốm (kích thước, hình dạng, màu sắc)
   - Tính toán các đặc trưng tổng thể (mật độ, độ phủ)

4. **Phân tích và đánh giá**:
   - Đánh giá mức độ nghiêm trọng của đốm sắc tố
   - Phân loại đốm sắc tố
   - Tạo báo cáo phân tích

## Huấn luyện mô hình

### Chuẩn bị dữ liệu

```bash
python scripts/data_preparation.py
```

### Huấn luyện mô hình phân đoạn

```bash
python scripts/train_models.py --model segmentation --backbone mobilenetv2 --epochs 50
```

### Huấn luyện mô hình phân loại

```bash
python scripts/train_models.py --model classification --backbone mobilenetv2 --epochs 50 --num_classes 5
```

## Ví dụ kết quả

Kết quả phân tích bao gồm:

- Ảnh khuôn mặt đã xử lý
- Mặt nạ vùng đốm sắc tố
- Ảnh chồng lớp hiển thị vùng đốm sắc tố
- Thông tin phân tích số lượng, kích thước và mức độ nghiêm trọng của đốm sắc tố
- Biểu đồ phân loại các loại đốm sắc tố

## Dữ liệu

Hệ thống sử dụng các nguồn dữ liệu công khai như:

- Fitzpatrick17k Dataset
- ISIC Skin Lesion Dataset

## Triển khai

Hệ thống có thể được triển khai như:

- Dịch vụ web API
- Ứng dụng di động
- Tích hợp vào các hệ thống camera thông minh

## Tác giả

- Quản lý dự án và phát triển mô hình - Senior AI Engineer

## Tài liệu tham khảo

- [Fitzpatrick17k: A Dataset for Skin Lesion Analysis](https://arxiv.org/abs/2104.09957)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Deep Learning for Skin Lesion Segmentation and Classification](https://www.mdpi.com/2079-9292/9/11/1770) 