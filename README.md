
# Dự Án Dự Đoán Giá Bất Động Sản

Dự án này là một phần của khóa học **Big Data** tại **Samsung Innovation Campus**. Mục tiêu của dự án là phân tích và dự đoán giá bất động sản tại Bắc Kinh sử dụng các kỹ thuật **Big Data** và mô hình **Máy Học (Machine Learning)**.

## Thông Tin Dự Án

- **Ngày**: 14/08/2024
- **Tên Nhóm**: BD 02
- **Thành Viên Nhóm**:
  - **Lê Đông Anh Kiệt** (Nhóm trưởng)
  - **Trần Tuấn Kiệt**
  - **Ngô Văn Lâu**
  - **Ngân Hoàng Huy**
  - **La Hồng Lộc**
  - **Nguyễn Đặng Phú Mẫn**

## Mục Lục

1. [Giới Thiệu](#giới-thiệu)
2. [Thực Thi Dự Án](#thực-thi-dự-án)
3. [Kết Quả](#kết-quả)
4. [Tác Động Dự Kiến](#tác-động-dự-kiến)

## Giới Thiệu

### 1.1 Thông Tin Nền Tảng

Thị trường bất động sản ngày càng phức tạp, chịu ảnh hưởng bởi nhiều yếu tố như kinh tế vĩ mô, vị trí, và đặc điểm của tài sản. Các phương pháp truyền thống để định giá thường không nắm bắt hết được các yếu tố này. **Big Data** cho phép chúng ta phân tích một cách toàn diện và cung cấp dự đoán chính xác hơn về xu hướng giá bất động sản.

### 1.2 Động Lực và Mục Tiêu

Dự án sử dụng **Big Data** để cung cấp các dự đoán giá bất động sản chính xác, thời gian thực, hỗ trợ nhà đầu tư, nhà phát triển và người tiêu dùng đưa ra các quyết định hợp lý hơn.

### 1.3 Phân Công Vai Trò

| Tên                  | Vai Trò                                |
|----------------------|----------------------------------------|
| **Lê Đông Anh Kiệt**  | Tiền xử lý dữ liệu, Huấn luyện mô hình |
| **Trần Tuấn Kiệt**    | Trực quan hóa dữ liệu, Đánh giá mô hình|
| **Ngô Văn Lâu**       | Viết báo cáo, Thu thập dữ liệu         |
| **Ngân Hoàng Huy**    | Thu thập dữ liệu, Phân tích mô hình    |
| **La Hồng Lộc**       | Thu thập dữ liệu, Huấn luyện mô hình   |
| **Nguyễn Đặng Phú Mẫn**| Trực quan hóa dữ liệu, Viết báo cáo    |

### 1.4 Lịch Trình và Mốc Thời Gian

| Ngày                  | Nhiệm Vụ                              |
|-----------------------|---------------------------------------|
| 01/06 - 24/06         | Thu thập dữ liệu từ **Kaggle** và **Lianjia** |
| 18/06 - 07/07         | Tiền xử lý và làm sạch dữ liệu        |
| 25/06 - 07/07         | Xây dựng mô hình                      |
| 30/06 - 05/08         | Trực quan hóa dữ liệu                 |
| 08/07 - 14/08         | Huấn luyện mô hình                    |
| 01/08 - 14/08         | Viết báo cáo cuối cùng                |

## Thực Thi Dự Án

### 2.1 Mô Tả Kịch Bản Giả Lập

Dự án sử dụng bộ dữ liệu gồm hơn **318,000** giao dịch bất động sản tại Bắc Kinh để dự đoán giá bất động sản bằng các kỹ thuật **Máy Học**. Bộ dữ liệu bao gồm các thông tin về vị trí, giá cả, đặc điểm tài sản và nhiều thuộc tính khác.

### 2.2 Bộ Dữ Liệu

Bộ dữ liệu bao gồm **26 thuộc tính** chứa thông tin về vị trí, giá giao dịch và các đặc điểm của bất động sản. Dữ liệu đã được làm sạch và xử lý để sử dụng trong huấn luyện mô hình.

### 2.3 Quy Trình Nhập Dữ Liệu

Dữ liệu được đọc bằng **Pandas**, sử dụng mã hóa `GBK` để xử lý các ký tự tiếng Trung.

```python
df = pd.read_csv('rawdata.csv', encoding='gbk', low_memory=False)
```

### 2.4 Biến Đổi và Xử Lý Dữ Liệu

Dữ liệu đã được làm sạch bằng cách xử lý các giá trị thiếu, ngoại lệ và những lỗi nhập liệu sai. Biến đổi đặc trưng bao gồm **Chuẩn hóa Min-Max** và phát hiện ngoại lệ bằng **IQR**.

### 2.5 Phương Pháp Dự Đoán

Bốn mô hình dự đoán đã được thử nghiệm:

- **Hồi Quy Tuyến Tính**
- **Cây Quyết Định**
- **Gradient Boosting**
- **Mạng Nơ-ron Nhân Tạo (ANN)**

### 2.6 Trực Quan Hóa Dữ Liệu

Dữ liệu được trực quan hóa để khám phá các mẫu và xác thực kết quả mô hình.

```python
sns.histplot(df['totalPrice'], kde=True, log_scale=True)
```

## Kết Quả

### 3.1 Mã và Kịch Bản Nhập Dữ Liệu

Dữ liệu được nhập và làm sạch bằng mã sau:
```python
df = pd.read_csv('rawdata.csv', encoding='gbk')
```

### 3.2 Mã Biến Đổi Dữ Liệu

Các giá trị ngoại lệ được loại bỏ bằng **IQR**:
```python
Q1 = df['totalPrice'].quantile(0.25)
Q3 = df['totalPrice'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['totalPrice'] < (Q1 - 1.5 * IQR)) | (df['totalPrice'] > (Q3 + 1.5 * IQR)))]
```

### 3.3 Hiệu Suất Mô Hình

Mô hình tốt nhất là **Random Forest**, với sai số **MAE**, **MSE**, **RMSE** thấp nhất và giá trị **R²** cao nhất.

## Tác Động Dự Kiến

### 4.1 Thành Tựu và Lợi Ích

- **Xử Lý Dữ Liệu**: Thành công trong việc làm sạch và chuẩn bị một tập dữ liệu lớn.
- **Phát Triển Mô Hình**: Xây dựng và thử nghiệm nhiều mô hình dự đoán khác nhau.
- **Hiểu Biết Từ Dữ Liệu**: Cung cấp các dự đoán giá bất động sản chính xác, hỗ trợ quyết định đầu tư và phân tích thị trường.

### 4.2 Cải Tiến Tương Lai

- Thử nghiệm các thuật toán nâng cao như **XGBoost** để cải thiện độ chính xác.
- Sử dụng **Phân Tích Không Gian Địa Lý** để đánh giá ảnh hưởng của vị trí đến giá bất động sản.
- Khám phá việc sử dụng **Học Sâu** (Deep Learning) để cải thiện kết quả.

