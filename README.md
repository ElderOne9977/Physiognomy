# Physiognomy

Physiognomy là dự án thử nghiệm về xử lý ảnh khuôn mặt và chuyển dữ liệu có cấu trúc thành văn bản. Repository lưu mã nguồn, bộ ảnh, các bảng dữ liệu đã xử lý và notebook ghi lại quá trình chuẩn bị dữ liệu.

Dự án bắt đầu từ việc đo hình học khuôn mặt, sau đó ghép các phép đo với nhãn văn bản từ một dịch vụ bên ngoài. Ứng dụng có giao diện Gradio để hiển thị báo cáo và tải dữ liệu CSV/JSON.

**Cần hiểu đúng bộ nhãn:** các diễn giải về tính cách, nghề nghiệp, quan hệ, tương lai và tài chính là nội dung do dịch vụ cung cấp. Chúng chưa được kiểm chứng là thông tin đúng về một người. Kết quả trong dự án không dùng để đánh giá con người hay ra quyết định tuyển dụng, tài chính hoặc các quyết định quan trọng khác.

## Trong repository có gì?

| Thành phần | Nội dung |
| --- | --- |
| `Humans/` | Bộ ảnh đầu vào gồm 7.472 tệp. |
| `Physiognomy_VSCode_Colab_Features.ipynb` | Notebook xử lý ảnh và đặc trưng khuôn mặt. |
| `faces_full_features.csv` | Bảng đặc trưng ban đầu. |
| `faces_full_features_clean.csv` | Bảng sau khi làm sạch: 6.988 dòng, 452 cột. |
| `labels_enriched.jsonl` | 7.467 bản ghi nhãn từ dịch vụ bên ngoài. |
| `training_dataset.csv`, `merged_full_features.csv` | Dữ liệu ghép giữa đặc trưng và nhãn, ở hai mức số lượng cột. |
| `semantic_schema.json`, `semantic_labels.csv` | Danh sách và bảng mã hóa 45 nhãn semantic. |
| `semantic_training_dataset.csv` | 6.983 dòng ghép được giữa dữ liệu đặc trưng và nhãn semantic. |
| `training_*.jsonl`, `text_generation_*.jsonl` | Các bản dữ liệu văn bản ở định dạng dùng trong những thử nghiệm trước đây. |
| `Source/text_generation_controlled_train.jsonl` | Dataset khái niệm có cấu trúc và văn bản của thử nghiệm controlled. |
| `oof_semantic_*` | Dự đoán ngoài fold và metadata của thử nghiệm semantic đã lưu. |
| `Source/` | Mã xử lý dữ liệu, mô hình, đánh giá, giao diện và kiểm thử. |
| `Source/assets/` | CSS, JavaScript và hình vector minh họa trên giao diện. |

Các bảng phục vụ những bước khác nhau nên số dòng và số cột không giống nhau. Riêng đầu vào của mô hình semantic dùng 95 đặc trưng gốc và 21 cột đánh dấu thiếu dữ liệu, theo thứ tự đã lưu; không phải toàn bộ 452 cột trong CSV.

Ảnh và các dataset lớn được lưu bằng **Git LFS**. Log, bản sao lưu, cache, ảnh kiểm thử cá nhân, output chạy thử, cấu hình công cụ cá nhân và model weights không nằm trong bản upload này. Các tài liệu Markdown nội bộ được giữ ở máy; README này là ngoại lệ. Các cấu hình Docker phụ thuộc weights và tài liệu nội bộ cũng chưa được phát hành trong bản này.

## Tải mã nguồn và dataset

Cài Git và Git LFS trước, sau đó chạy:

```powershell
git lfs install
git clone https://github.com/ElderOne9977/Physiognomy.git
cd Physiognomy
git lfs pull
```

Thư mục ảnh gốc có dung lượng khoảng 2,06 GB, chưa tính các bảng dữ liệu. Nếu chỉ cần đọc mã nguồn, có thể clone mà chưa tải các đối tượng LFS:

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone https://github.com/ElderOne9977/Physiognomy.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
```

Trong trường hợp này, các tệp LFS chỉ là tệp trỏ nhỏ. Chạy `git lfs pull` trong thư mục repository khi cần dữ liệu thật. Việc tải LFS phụ thuộc dung lượng lưu trữ và băng thông khả dụng của tài khoản GitHub.

## Đọc và tìm hiểu dữ liệu

Bắt đầu từ notebook `Physiognomy_VSCode_Colab_Features.ipynb` để xem quá trình xử lý. Notebook giữ nội dung và output hiện có; một số đường dẫn hoặc thiết lập gắn với môi trường gốc, vì vậy cần đọc từng cell và chỉnh lại đường dẫn trước khi chạy.

Các tệp trong `requirements*.txt` có mục đích khác nhau: `requirements.txt` dành cho phần thu thập dữ liệu cũ, `requirements-demo.txt` khai báo Gradio, còn `requirements-cpu.txt` liệt kê các phụ thuộc của môi trường CPU. Tệp CPU dùng khoảng phiên bản, chưa phải bản khóa chính xác của toàn bộ môi trường.

Model weights và bộ nhớ đệm của thư viện không được phát hành kèm repository. Clone mã nguồn không đồng nghĩa với việc đã có đầy đủ tài nguyên để chạy suy luận offline. Một số script đánh giá cũ cũng cần artifact thử nghiệm được giữ riêng ở máy.

## Xem thử giao diện

Bản preview dùng ảnh tải lên để thử thao tác giao diện và trả về **văn bản giả lập**. Nó không nạp mô hình hay thực hiện dự đoán thật.

Trong môi trường Python đã có các phụ thuộc của dự án, chạy từ thư mục gốc:

```powershell
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -u -B Source/experiments/ui_reading_20261014/mock_preview.py
```

Mở <http://127.0.0.1:7862> và nhấn `Ctrl+C` trong terminal để dừng. Chỉ chạy một bản preview trên cổng này.

Giao diện có chế độ sáng/tối, sơ đồ minh họa điểm mốc khuôn mặt, mục lục báo cáo, nút sao chép và tải CSV/JSON. Sơ đồ là hình minh họa, không phải kết quả đo từ ảnh được tải lên.

## Kiểm thử giao diện

Các bài kiểm tra dưới đây dùng dữ liệu mẫu và mock phần khởi tạo mô hình:

```powershell
$env:GRADIO_ANALYTICS_ENABLED='False'
$env:HF_HUB_OFFLINE='1'
$env:PYTHONIOENCODING='utf-8'
python -u -B Source/test_demo_reading_view.py
```

Bài `Source/test_demo_gradio.py` có thêm kiểm tra sự hiện diện của artifact mô hình nên cần các artifact cục bộ. Để kiểm tra trên trình duyệt, khởi động preview ở một terminal rồi chạy `python -u -B Source/test_demo_ui_browser.py` ở terminal khác. Bài kiểm tra này cần Playwright và Chromium, chỉ dùng ảnh tổng hợp với preview mock. Dừng preview sau khi kiểm tra.

## Giới hạn đã biết

- Nhãn từ dịch vụ là weak supervision, không phải nhãn sự thật được xác nhận độc lập.
- Tập kiểm thử văn bản đã lưu có 702 bản ghi và 686 vector semantic khác nhau. Checkpoint FLAN-T5 gốc chỉ tạo ra 5 văn bản khác nhau trên tập này; đây là dấu hiệu lặp đầu ra.
- Các chế độ template có thể bám sát khái niệm được chọn hơn, nhưng điều đó không chứng minh khả năng suy đoán đặc điểm con người từ ngoại hình.
- Kết quả kiểm thử mock xác nhận hành vi phần mềm, không xác nhận độ chính xác trên khuôn mặt mới.
- Repository không kèm một giấy phép dữ liệu được xác nhận. Có quyền truy cập tệp không đồng nghĩa với quyền tái phân phối hoặc sử dụng ảnh cho mọi mục đích; cần kiểm tra quyền sử dụng trước khi dùng lại.

README này mô tả phạm vi dữ liệu và phần mềm hiện có; các kết quả thử nghiệm cũ không phải cam kết về chất lượng hay khả năng sẵn sàng triển khai.
