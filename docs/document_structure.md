# Document Structure

Tài liệu này mô tả cách project tổ chức Document trong pipeline RAG.

## Luồng Tổng Quan

```text
PDF file
  -> load_pdf_document()
  -> Document
      -> pages: list[DocumentPage]
          -> text
          -> images: list[DocumentImage]
          -> metadata
  -> Document.to_chunk_inputs()
  -> split_documents()
  -> TextChunk
  -> EmbeddingService
  -> VectorStore metadata
  -> Retrieval source
```

Ý tưởng chính: file gốc được biểu diễn bằng một `Document`, mỗi trang là một `DocumentPage`, còn ảnh trong trang được ghi nhận bằng `DocumentImage`. Khi cần chunking, `Document` được chuyển sang page records để tương thích với pipeline hiện tại.

## Document

`Document` đại diện cho toàn bộ tài liệu gốc trước khi chia nhỏ thành chunk.

```python
@dataclass(frozen=True)
class Document:
    document_id: str
    source_path: str
    source_name: str
    document_type: str
    title: str | None = None
    author: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[DocumentPage] = field(default_factory=list)
```

Các trường quan trọng:

- `document_id`: ID ổn định của tài liệu, tạo từ đường dẫn, kích thước file và thời điểm sửa đổi.
- `source_path`: đường dẫn file gốc.
- `source_name`: tên file hiển thị trong nguồn trả lời.
- `document_type`: loại tài liệu, hiện là `pdf`.
- `title`, `author`, `created_at`: metadata đọc từ PDF nếu có.
- `metadata`: metadata mở rộng như `page_count`, `file_size_bytes`.
- `pages`: danh sách trang đã được extract.

## DocumentPage

`DocumentPage` đại diện cho một trang trong tài liệu.

```python
@dataclass(frozen=True)
class DocumentPage:
    document_id: str
    page_number: int
    text: str
    source_name: str
    images: list[DocumentImage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Các trường quan trọng:

- `document_id`: liên kết trang về tài liệu gốc.
- `page_number`: số trang, bắt đầu từ 1.
- `text`: nội dung text được extract bằng PyMuPDF.
- `source_name`: tên file nguồn.
- `images`: danh sách ảnh xuất hiện trong trang.
- `metadata`: thống kê cấp trang như `char_count`, `word_count`, `image_count`.

`DocumentPage.to_chunk_input()` chuyển trang sang dạng dict mà chunker hiện tại dùng:

```python
{
    "document_id": "...",
    "text": "...",
    "page": 1,
    "source": "document.pdf",
    "metadata": {...},
}
```

## DocumentImage

`DocumentImage` ghi nhận ảnh trong từng trang. Project có thể export ảnh gốc ra file và lưu `image_path`; OCR/caption ảnh chưa được thực hiện.

```python
@dataclass(frozen=True)
class DocumentImage:
    document_id: str
    page_number: int
    image_index: int
    image_path: str | None = None
    ocr_text: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Các trường quan trọng:

- `document_id`: liên kết ảnh về tài liệu gốc.
- `page_number`: ảnh nằm ở trang nào.
- `image_index`: thứ tự ảnh trong trang.
- `image_path`: nơi lưu ảnh đã export ra file nếu `DOCUMENT_IMAGE_EXPORT=true`.
- `ocr_text`: text trích xuất từ ảnh nếu sau này dùng OCR.
- `caption`: mô tả ảnh nếu sau này dùng vision model.
- `metadata`: thông tin kỹ thuật như `xref`, `width`, `height`, `colorspace`, `filter`.

## Liên Kết Sang Chunk Và Retrieval

Sau khi load PDF:

```python
document = load_pdf_document(pdf_path)
page_records = document.to_chunk_inputs()
chunks = split_documents(page_records)
```

Mỗi `TextChunk` giữ lại:

```python
{
    "document_id": "...",
    "text": "...",
    "page": 1,
    "source": "document.pdf",
    "chunk_index": 0,
}
```

Khi build vector store, các field này được lưu vào `metadata.json`. Khi retrieval trả kết quả, source trả về có thể truy ngược theo:

```text
document_id + source + page + chunk_index
```

Nhờ vậy hệ thống biết câu trả lời lấy từ tài liệu nào, trang nào, và chunk nào.

## Phạm Vi Hiện Tại

Đã có:

- Schema rõ cho `Document`, `DocumentPage`, `DocumentImage`.
- Metadata cấp document và page.
- Ghi nhận ảnh trong PDF ở dạng metadata.
- Export ảnh trong PDF ra `storage/images/<document_id>/`.
- `document_id` đi xuyên từ loader sang chunk, vector store, retrieval source.
- Tương thích ngược với hàm `load_pdf()` cũ.

Chưa có:

- OCR cho ảnh.
- Caption ảnh bằng vision model.
- Parse bảng, công thức, heading/section.
- Lưu document object đầy đủ vào database riêng.
