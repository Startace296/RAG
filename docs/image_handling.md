# Image Handling In Documents

Tài liệu này mô tả cách project xử lý PDF có hình trong pipeline RAG.

## Luồng Xử Lý Hiện Tại

```text
PDF page
  -> PyMuPDF page.get_images(full=True)
  -> DocumentImage metadata
  -> optional image export
  -> optional OCR
  -> DocumentPage.images
  -> DocumentPage.to_chunk_input()
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
DOCUMENT_IMAGE_OCR=false
IMAGE_OUTPUT_DIR=storage/images
```

Nếu không muốn lưu ảnh ra file:

```env
DOCUMENT_IMAGE_EXPORT=false
```

Khi tắt export, hệ thống vẫn ghi nhận ảnh trong `DocumentImage`, nhưng `image_path` sẽ là `None`.

Nếu bật `DOCUMENT_IMAGE_OCR=true`, loader cần có file ảnh để đưa vào OCR. Khi đó ảnh sẽ được ghi ra `IMAGE_OUTPUT_DIR` ngay cả khi `DOCUMENT_IMAGE_EXPORT=false`.

## OCR Ảnh

Nếu `DOCUMENT_IMAGE_OCR=true`, loader sẽ thử đọc chữ trong ảnh bằng `pytesseract`:

```text
PDF image
  -> exported image file
  -> pytesseract.image_to_string()
  -> DocumentImage.ocr_text
  -> DocumentPage.chunk_text()
  -> split_documents()
```

OCR là optional. Nếu chưa cài `pytesseract` hoặc Tesseract OCR binary, pipeline vẫn chạy tiếp và ghi trạng thái lỗi vào `DocumentImage.metadata["ocr_status"]`.

Các trạng thái OCR thường gặp:

- `disabled`: OCR đang tắt.
- `missing_dependency`: thiếu thư viện Python hoặc Tesseract OCR binary.
- `failed`: OCR chạy lỗi.
- `empty`: OCR chạy được nhưng không đọc ra chữ.
- `extracted`: OCR đọc được text và đưa text đó vào retrieval.

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
        "ocr_status": "extracted",
        "ocr_char_count": 120,
        "ocr_word_count": 18,
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
    "image_text_count": 1,
    "searchable_char_count": 820,
    "searchable_word_count": 130,
}
```

## Ảnh Có Được Đưa Vào Retrieval Không?

Hiện tại: có, nếu ảnh đã có `ocr_text` hoặc `caption`.

Pipeline retrieval embed text sau khi trang được chuyển thành chunk input:

```text
DocumentPage.text + DocumentImage.ocr_text/caption
  -> TextChunk
  -> EmbeddingService
  -> VectorStore
```

Ảnh luôn được:

- phát hiện,
- lưu ra file,
- gắn metadata vào `DocumentPage.images`.

Khi OCR đọc được chữ, nội dung đó được append vào text của page trước khi chunking. Nhờ vậy một trang chỉ có ảnh nhưng OCR ra text vẫn có thể được đưa vào index.

## Cách Mở Rộng Sau Này

Có hai hướng chính để làm giàu ảnh trong RAG:

1. OCR

Dùng OCR để lấy chữ trong ảnh, ví dụ chữ trong screenshot, scan, biểu đồ có nhãn.

Luồng:

```text
DocumentImage.image_path
  -> OCR model
  -> DocumentImage.ocr_text
  -> append vào page text
  -> embedding
```

Phù hợp khi ảnh chứa nhiều chữ. Hướng này đã có ở mức optional.

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
- OCR ảnh bằng `pytesseract` nếu bật `DOCUMENT_IMAGE_OCR=true`.
- Đưa `ocr_text`/`caption` vào text dùng cho chunking và retrieval.

Phần chưa làm là caption ảnh bằng vision model và tạo image chunk riêng nếu muốn tách ảnh khỏi text thường.
