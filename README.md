# Physiognomy

Một dự án thử nghiệm về hình học khuôn mặt và sinh văn bản từ dữ liệu có cấu trúc. Repository gồm bộ ảnh, các bảng đặc trưng, notebook xử lý dữ liệu và giao diện Gradio để đọc báo cáo.

Dự án kết hợp các phép đo từ ảnh với nhãn văn bản do một dịch vụ bên ngoài cung cấp. Các nhãn về tính cách, nghề nghiệp, quan hệ, tương lai và tài chính chưa được kiểm chứng là thông tin đúng về một người. Không dùng các diễn giải này để đánh giá con người hoặc đưa ra quyết định quan trọng.

## Cách hoạt động

```text
Ảnh khuôn mặt → Trích xuất hình học → Bảng đặc trưng CSV
                                             ↓
                                  Biểu diễn semantic → Văn bản
                                                           ↓
                                              Báo cáo trên Gradio
```

- **Dữ liệu hình học:** các điểm mốc giúp biểu diễn khoảng cách, góc và tỷ lệ trên khuôn mặt.
- **Chuẩn bị dataset:** mã xử lý ghép bảng đặc trưng với nhãn của dịch vụ theo tên ảnh đã chuẩn hóa.
- **Sinh văn bản:** các thử nghiệm dùng template hoặc FLAN-T5 để tạo văn bản theo năm nhóm nội dung.
- **Giao diện đọc báo cáo:** năm phần có mục lục, nút sao chép và tệp CSV/JSON để tải xuống.
- **Sáng và tối:** công tắc trong góc phải của header ghi nhớ lựa chọn trên trình duyệt.

Hình khuôn mặt có điểm mốc trên giao diện là sơ đồ minh họa. Nó không phải kết quả trích xuất từ ảnh bạn vừa tải lên.

## Bạn cần gì?

- Git và Git LFS để tải mã nguồn cùng dataset.
- Môi trường Python có các phụ thuộc phù hợp nếu muốn mở notebook hoặc thử giao diện.
- Vài GB dung lượng trống cho ảnh và các bảng dữ liệu. Riêng thư mục `Humans/` có dung lượng khoảng 2,06 GB.
- Jupyter hoặc VS Code có hỗ trợ notebook để đọc tệp `.ipynb`.

Các lệnh bên dưới dùng PowerShell và chạy từ thư mục gốc của repository. Truy cập Internet cần thiết khi clone, tải dữ liệu LFS hoặc cài thư viện.

## Bắt đầu

### 1. Tải dự án và dataset

```powershell
git lfs install
git clone https://github.com/ElderOne9977/Physiognomy.git
cd Physiognomy
git lfs pull
```

Git LFS lưu ảnh và các dataset lớn ngoài lịch sử Git thông thường. Nếu chỉ muốn đọc mã nguồn trước, có thể clone mà chưa tải dữ liệu:

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone https://github.com/ElderOne9977/Physiognomy.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

Khi cần dataset, vào thư mục `Physiognomy` rồi chạy `git lfs pull`. Các tệp trỏ nhỏ sẽ được thay bằng dữ liệu thật; việc tải phụ thuộc dung lượng và băng thông LFS khả dụng của tài khoản GitHub.

### 2. Mở notebook và tìm hiểu dữ liệu

Bắt đầu với [Physiognomy_VSCode_Colab_Features.ipynb](Physiognomy_VSCode_Colab_Features.ipynb). Notebook giữ các cell và output hiện có để tiện theo dõi quá trình xử lý. Một số đường dẫn gắn với môi trường gốc, nên hãy đọc từng cell và chỉnh đường dẫn trước khi chạy.

| Dữ liệu | Nội dung |
| --- | --- |
| `Humans/` | 7.472 ảnh đầu vào. |
| `faces_full_features.csv` | Bảng đặc trưng ban đầu. |
| `faces_full_features_clean.csv` | Bảng đã làm sạch, 6.988 dòng × 452 cột. |
| `labels_enriched.jsonl` | 7.467 bản ghi nhãn từ dịch vụ bên ngoài. |
| `training_dataset.csv`, `merged_full_features.csv` | Hai bản dữ liệu ghép giữa đặc trưng và nhãn. |
| `semantic_schema.json`, `semantic_labels.csv` | Danh sách và bảng mã hóa 45 nhãn semantic. |
| `semantic_training_dataset.csv` | 6.983 dòng ghép được giữa đặc trưng và nhãn semantic. |
| `training_*.jsonl`, `text_generation_*.jsonl` | Dataset văn bản ở các định dạng thử nghiệm. |
| `Source/text_generation_controlled_train.jsonl` | Dataset khái niệm có cấu trúc và văn bản của thử nghiệm controlled. |
| `oof_semantic_*` | Dự đoán ngoài fold và metadata đã lưu. |

Số dòng khác nhau vì không phải ảnh nào cũng có đủ cả đặc trưng và nhãn ghép tương ứng. Các schema cũng phục vụ từng bước riêng: bảng clean có 452 cột, còn mô hình semantic dùng 95 đặc trưng gốc và 21 cột đánh dấu thiếu dữ liệu, theo thứ tự đã lưu.

### 3. Chuẩn bị môi trường

Các tệp phụ thuộc được chia theo mục đích:

| Tệp | Dùng cho |
| --- | --- |
| `requirements.txt` | Playwright và công cụ phân tích HTML của phần thu thập dữ liệu cũ. Playwright cũng được dùng trong kiểm thử trình duyệt. |
| `requirements-demo.txt` | Phiên bản Gradio của giao diện. Tệp này không bao gồm toàn bộ phụ thuộc được import bởi mã dự án. |
| `requirements-cpu.txt` | Các thư viện dữ liệu, ảnh và mô hình cho môi trường CPU. |

Tệp CPU dùng khoảng phiên bản, chưa khóa chính xác toàn bộ môi trường. Model weights, cache thư viện và kết quả thí nghiệm cục bộ không nằm trong bản upload. Vì vậy, clone repository chưa cung cấp một bộ suy luận offline hoàn chỉnh; một số script đánh giá cũng cần các artifact được giữ riêng ở máy.

## Xem thử giao diện

Trong môi trường Python đã có các phụ thuộc của dự án, chạy:

```powershell
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -u -B Source/experiments/ui_reading_20261014/mock_preview.py
```

Mở **http://127.0.0.1:7862/** trong trình duyệt. Tải một ảnh để thử bố cục, thanh tiến trình, mục lục, sao chép và tải tệp. Nhấn **Ctrl+C** trong terminal để dừng; chỉ chạy một preview trên cổng này.

Preview dùng **văn bản giả lập**, không nạp mô hình hay thực hiện dự đoán thật. CSV/JSON tải xuống từ preview cũng là dữ liệu mẫu để kiểm tra giao diện.

## Kiểm thử

Để chạy kiểm thử phần hiển thị trong môi trường đã chuẩn bị:

```powershell
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -u -B Source/test_demo_reading_view.py
```

Kiểm thử trình duyệt cần Playwright và Chromium. Khởi động preview mock ở một terminal, sau đó chạy `python -u -B Source/test_demo_ui_browser.py` ở terminal khác. Bài kiểm tra chỉ dùng ảnh tổng hợp với preview mock; dừng preview sau khi hoàn tất.

`Source/test_demo_gradio.py` có thêm kiểm tra sự hiện diện của artifact mô hình nên cần các artifact cục bộ. Kết quả kiểm thử mock xác nhận hành vi phần mềm, không xác nhận độ chính xác trên khuôn mặt mới.

## Khi gặp lỗi

- **Dataset chỉ là vài dòng văn bản:** kiểm tra Git LFS đã được cài, sau đó chạy `git lfs pull` trong thư mục repository.
- **Không tải được LFS:** kiểm tra kết nối, quyền truy cập và dung lượng/băng thông khả dụng trên GitHub.
- **Thiếu thư viện Python:** kiểm tra đúng môi trường đang dùng và tệp requirements phù hợp. Chỉ cài Gradio chưa đủ cho mọi import của dự án.
- **Notebook không tìm thấy ảnh:** chỉnh lại đường dẫn trong cell và kiểm tra thư mục `Humans/` đã tải đủ dữ liệu LFS.
- **Cổng 7862 đang được dùng:** dừng bản preview trước đó bằng `Ctrl+C`, rồi khởi động lại.
- **Thiếu model weights hoặc artifact:** đây là các tệp được giữ cục bộ, không nằm trong lần phát hành mã nguồn và dataset này.

## Giới hạn hiện tại

Nhãn dịch vụ là *weak supervision*: chúng giúp nghiên cứu cách biểu diễn và tạo văn bản, nhưng không phải nhãn sự thật được xác nhận độc lập.

Tập kiểm thử văn bản đã lưu có **702 bản ghi**, tương ứng **686 vector semantic khác nhau**. Checkpoint FLAN-T5 gốc chỉ tạo ra **5 văn bản khác nhau** trên tập này, cho thấy hiện tượng lặp đầu ra. Template có thể bám sát khái niệm được chọn hơn, nhưng điều đó không chứng minh khả năng suy đoán đặc điểm con người từ ngoại hình.

Các kết quả thử nghiệm cũ không phải cam kết về chất lượng hay mức độ sẵn sàng triển khai.

## Cấu trúc dự án

```text
Physiognomy/
├── Humans/                                  # Bộ ảnh, lưu bằng Git LFS
├── Physiognomy_VSCode_Colab_Features.ipynb    # Notebook xử lý dữ liệu
├── faces_full_features_clean.csv            # Đặc trưng đã làm sạch
├── labels_enriched.jsonl                    # Nhãn từ dịch vụ
├── semantic_training_dataset.csv            # Dataset semantic đã ghép
├── training_*.jsonl                         # Dataset văn bản
├── requirements*.txt                        # Phụ thuộc theo mục đích
└── Source/
    ├── assets/                             # CSS, JavaScript, sơ đồ vector
    ├── demo_gradio.py                       # Giao diện đọc báo cáo
    ├── physiognomy.py                       # Xử lý ảnh và hình học
    ├── build_*.py                          # Chuẩn bị dữ liệu
    ├── test_*.py                           # Kiểm thử
    └── experiments/ui_reading_20261014/
        └── mock_preview.py                 # Preview giao diện giả lập
```

README là tài liệu Markdown duy nhất được đăng lên repository. Tài liệu nội bộ, log, backup, cache, ảnh kiểm thử cá nhân, model weights, output chạy thử và cấu hình Docker phụ thuộc tài nguyên cục bộ được giữ ngoài Git.

## Công cụ sử dụng

- **NumPy, pandas và scikit-learn** cho xử lý bảng dữ liệu và các thử nghiệm semantic.
- **InsightFace, facexlib và OpenCV** cho xử lý ảnh và đặc trưng khuôn mặt.
- **PyTorch, Transformers và FLAN-T5** cho các thử nghiệm sinh văn bản.
- **Gradio** cho giao diện, **Playwright** cho kiểm thử trình duyệt.
- **Git LFS** cho ảnh và dataset lớn.

Thư viện, model và ảnh có điều kiện sử dụng riêng. Repository chưa kèm một giấy phép dữ liệu được xác nhận; hãy kiểm tra quyền sử dụng trước khi tái phân phối hoặc dùng lại các tài nguyên này.
