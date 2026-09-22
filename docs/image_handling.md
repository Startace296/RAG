# Image Handling In Documents

Tài liệu này mô tả cách project xử lý PDF có hình trong pipeline RAG.

## Luồng Xử Lý Hiện Tại

```text
PDF page
  -> PyMuPDF page.get_images(full=True)
  -> DocumentImage metadata
  -> optional image export
  -> DocumentPage.images
  -> Document.pages
```

Khi loader đọc từng trang PDF, hệ thống làm hai việc song song:

- Extract text của trang bằng `page.get_text("text")`.
- Phát hiện ảnh trong trang bằng `page.get_images(full=True)`.

Mỗi ảnh được biểu diễn bằng một `DocumentImage`.

## Lưu Ảnh Ra File

Nếu `DOCUMENT_IMAGE_EXPORT=true`, ảnh trong PDF được export ra thư mục:

```text
storage/images/<document_id>/
```

Tên file ảnh có dạng:

```text
page_0001_image_0000_xref_123.png
```

Trong đó:

- `page_0001`: số trang chứa ảnh.
- `image_0000`: thứ tự ảnh trong trang.
- `xref_123`: mã tham chiếu nội bộ của ảnh trong PDF.
- Phần mở rộng lấy từ ảnh gốc nếu PyMuPDF đọc được.

Cấu hình nằm trong `.env`:

```env
DOCUMENT_IMAGE_EXPORT=true
IMAGE_OUTPUT_DIR=storage/images
```

Nếu không muốn lưu ảnh ra file:

```env
DOCUMENT_IMAGE_EXPORT=false
```

Khi tắt export, hệ thống vẫn ghi nhận ảnh trong `DocumentImage`, nhưng `image_path` sẽ là `None`.

## Metadata Của Ảnh

Mỗi `DocumentImage` lưu các thông tin:

```python
DocumentImage(
    document_id="...",
    page_number=1,
    image_index=0,
    image_path="storage/images/.../page_0001_image_0000_xref_123.png",
    ocr_text=None,
    caption=None,
    metadata={
        "xref": 123,
        "width": 800,
        "height": 600,
        "bits_per_component": 8,
        "colorspace": "DeviceRGB",
        "name": "...",
        "filter": "...",
        "extraction_status": "exported",
        "extension": "png",
        "file_size_bytes": 12345,
    },
)
```

Document-level metadata cũng có:

```python
{
    "image_count": 10,
    "image_output_dir": "storage/images/<document_id>",
}
```

Page-level metadata cũng có:

```python
{
    "image_count": 2,
}
```

## Ảnh Có Được Đưa Vào Retrieval Không?

Hiện tại: chưa.

Pipeline retrieval hiện tại chỉ embed text:

```text
DocumentPage.text -> TextChunk -> EmbeddingService -> VectorStore
```

Ảnh mới được:

- phát hiện,
- lưu ra file,
- gắn metadata vào `DocumentPage.images`.

Nội dung trong ảnh chưa được chuyển thành text, nên nếu một trang chỉ có sơ đồ/hình mà không có text giải thích, semantic search chưa thể tìm được nội dung đó.

## Cách Mở Rộng Sau Này

Có hai hướng chính để đưa ảnh vào RAG:

1. OCR

Dùng OCR để lấy chữ trong ảnh, ví dụ chữ trong screenshot, scan, biểu đồ có nhãn.

Luồng:

```text
DocumentImage.image_path
  -> OCR model
  -> DocumentImage.ocr_text
  -> append vào page text hoặc tạo image chunk riêng
  -> embedding
```

Phù hợp khi ảnh chứa nhiều chữ.

2. Image Captioning / Vision Model

Dùng vision model để mô tả nội dung ảnh, ví dụ sơ đồ, chart, kiến trúc hệ thống.

Luồng:

```text
DocumentImage.image_path
  -> vision model
  -> DocumentImage.caption
  -> append vào page text hoặc tạo image chunk riêng
  -> embedding
```

Phù hợp khi ảnh cần hiểu ngữ nghĩa thị giác, không chỉ đọc chữ.

## Kết Luận

Hiện project đã xử lý PDF có hình ở mức ingestion:

- Phát hiện ảnh trong từng trang.
- Lưu ảnh ra thư mục riêng.
- Ghi lại `image_path` và metadata kỹ thuật.
- Gắn ảnh vào `DocumentPage.images`.

Phần chưa làm là biến nội dung ảnh thành thông tin có thể search được. Bước tiếp theo hợp lý là thêm OCR hoặc caption để sinh `ocr_text`/`caption`, rồi đưa nội dung đó vào chunking.

