# Chunking

Tài liệu này mô tả vấn đề chunking trong RAG và cách project đã cải thiện phần chia chunk.

## Vấn Đề Của Chunking

Chunking quyết định đơn vị nội dung được đưa vào embedding và vector store. Nếu chunking kém, retrieval có thể lấy sai hoặc thiếu ngữ cảnh dù embedding model tốt.

Các vấn đề thường gặp và cách xử lý:

| Triệu chứng | Nguyên nhân thường gặp | Hướng xử lý |
| --- | --- | --- |
| Chunk kết thúc giữa câu hoặc ý | Dùng `fixed_word`/`sliding_window`, hoặc một câu dài vượt `CHUNK_SIZE` nên phải fallback sang tách theo từ | Thử `recursive`, `sentence` hoặc `paragraph`; kiểm tra lại chunk dài bất thường sau khi trích xuất PDF |
| Kết quả tìm kiếm thiếu phần giải thích cần thiết | Chunk quá nhỏ, hoặc thông tin bổ trợ nằm ở chunk kế tiếp | Tăng `CHUNK_SIZE`, thử tăng `CHUNK_OVERLAP`, hoặc dùng `parent_child` để retrieve child nhưng đưa thêm parent vào context |
| Kết quả tìm kiếm trộn nhiều chủ đề | Chunk quá lớn hoặc ranh giới đoạn/câu trong text không còn rõ | Giảm `CHUNK_SIZE`; kiểm tra text đã trích xuất; cân nhắc `semantic` khi tài liệu đổi chủ đề thường xuyên |
| Có nhiều kết quả gần như trùng nhau | `CHUNK_OVERLAP` lớn hoặc dùng sliding window trên tài liệu dài | Giảm overlap; kiểm tra số chunk và chất lượng top-k sau khi đổi cấu hình |
| Tiếng Việt bị chia câu không hợp lý | Sentence splitter hiện dựa vào dấu `.`, `!`, `?` theo sau bởi khoảng trắng; PDF có thể làm mất dấu câu hoặc ngắt dòng | Kiểm tra text đầu ra từ loader trước; chọn strategy phù hợp với cấu trúc thực tế thay vì chỉ dựa vào dấu câu |
| Heading, bảng, công thức hoặc code bị tách khỏi nội dung liên quan | Các cấu trúc này không luôn được nhận diện như boundary riêng khi split text | Cải thiện/chuẩn hóa text trước khi chunk; giữ heading hoặc metadata cấu trúc cùng nội dung; kiểm tra thủ công các tài liệu có bảng và công thức |
| Ngữ cảnh bị ngắt ở cuối mỗi trang | `split_documents` tạo chunk bên trong từng trang, không ghép text giữa hai trang | Giữ metadata trang để truy vết; nếu câu hoặc bảng thường xuyên chạy qua trang, xử lý/ghép các trang liên quan trước khi chunk |
| Đổi cấu hình nhưng kết quả không thay đổi | Vector index cũ vẫn chứa chunk đã tạo theo cấu hình trước | Rebuild index sau khi đổi strategy, size hoặc overlap |

Lưu ý: `CHUNK_SIZE` và `CHUNK_OVERLAP` trong cấu hình hiện được tính theo **số từ**, không phải token. Nên đánh giá bằng câu hỏi có ground truth và kiểm tra cả chất lượng retrieval, số chunk, kích thước index, thay vì chọn tham số chỉ theo cảm giác.

## Cấu Hình

Các cấu hình chunking nằm trong `.env`:

```env
CHUNKING_STRATEGY=recursive
CHUNK_SIZE=300
CHUNK_OVERLAP=50
MIN_CHUNK_SIZE=50
```

Sau khi đổi cấu hình chunking, cần rebuild index:

```powershell
python -m src.main --rebuild
```

Nếu không rebuild, hệ thống vẫn dùng `index.faiss` và `metadata.json` đã tạo từ cấu hình cũ.

Ý nghĩa:

- `CHUNKING_STRATEGY`: chiến lược chia chunk.
- `CHUNK_SIZE`: số từ mục tiêu tối đa trong một chunk.
- `CHUNK_OVERLAP`: số từ hoặc đơn vị ngữ cảnh lặp lại giữa hai chunk gần nhau.
- `MIN_CHUNK_SIZE`: kích thước tối thiểu mong muốn trước khi đóng chunk với các strategy có nhận biết boundary.
- `section_title`: lấy từ page metadata nếu loader đoán được heading của trang; nếu không có thì chunker fallback sang dòng đầu phù hợp trong chunk.

## Các Strategy Đã Có

### fixed_word

Chia theo số từ cố định.

Ưu điểm:

- Đơn giản.
- Nhanh.
- Dễ dự đoán số lượng chunk.

Nhược điểm:

- Dễ cắt ngang câu hoặc đoạn.
- Không hiểu cấu trúc tài liệu.
- Có thể làm mất ngữ cảnh quan trọng.

Nên dùng khi:

- Cần baseline đơn giản.
- Tài liệu đã được chuẩn hóa, mỗi đoạn có độ dài đều.
- Muốn benchmark so với các strategy tốt hơn.

### sentence

Chia text thành câu, sau đó gom câu thành chunk gần `CHUNK_SIZE`.

Ưu điểm:

- Ít cắt ngang câu.
- Tốt hơn fixed word cho nội dung giải thích tự nhiên.

Nhược điểm:

- Phụ thuộc dấu câu.
- PDF extract lỗi dấu câu có thể làm split kém.
- Một câu quá dài vẫn phải fallback sang word split.

Nên dùng khi:

- Tài liệu có văn xuôi rõ ràng.
- Câu có dấu chấm/hỏi/cảm tương đối chuẩn.

### paragraph

Chia theo đoạn, sau đó gom đoạn thành chunk.

Ưu điểm:

- Giữ được ý nghĩa tự nhiên theo đoạn.
- Tốt cho tài liệu có format đoạn rõ.

Nhược điểm:

- PDF extract đôi khi mất blank line, khiến paragraph detection yếu.
- Đoạn quá dài vẫn cần fallback.

Nên dùng khi:

- Tài liệu có paragraph rõ.
- Nội dung có nhiều giải thích theo đoạn độc lập.

### recursive

Ưu tiên paragraph, nếu đoạn quá dài thì fallback sang sentence, nếu câu/đơn vị vẫn quá dài thì fallback sang word split.

Ưu điểm:

- Cân bằng giữa giữ ngữ nghĩa và kiểm soát kích thước chunk.
- Phù hợp làm mặc định cho RAG text.
- Ít cắt ngang ý hơn fixed word.

Nhược điểm:

- Phức tạp hơn fixed word.
- Kết quả phụ thuộc chất lượng text extract từ PDF.
- Cần benchmark để chọn `CHUNK_SIZE`/`CHUNK_OVERLAP` tốt nhất.

Nên dùng khi:

- Chưa có benchmark chi tiết.
- Tài liệu gồm nhiều đoạn/câu tự nhiên.
- Muốn mặc định thực dụng cho PDF học thuật.

## Schema Chunk

Mỗi chunk hiện lưu thêm metadata:

```python
{
    "chunk_id": "...",
    "document_id": "...",
    "text": "...",
    "source": "document.pdf",
    "page": 1,
    "chunk_index": 0,
    "start_word": 0,
    "end_word": 300,
    "token_count": 300,
    "char_count": 1800,
    "section_title": "...",
    "chunking_strategy": "recursive",
    "image_refs": [
        {
            "image_index": 0,
            "image_path": "storage/images/.../page_0001_image_0000_xref_123.png",
            "ocr_text": "...",
            "caption": None,
        }
    ],
}
```

Các trường này giúp:

- Truy vết chunk trong retrieval.
- Debug chunk quá ngắn/quá dài.
- So sánh các strategy.
- Lưu strategy vào `metadata.json` của vector store.
- Truy vết chunk về ảnh trên cùng page thông qua `image_refs`.

## Luồng Hiện Tại

```text
DocumentPage.text + DocumentPage.images
  -> split_documents(strategy=CHUNKING_STRATEGY)
  -> TextChunk + metadata + image_refs
  -> EmbeddingService
  -> VectorStore(metadata.json)
  -> SearchResult
```

## Test Retrieval Với Strategy Khác

Dùng script retrieval test:

```powershell
python -m src.retrieval_test --chunking-strategy recursive
```

Thử baseline fixed word:

```powershell
python -m src.retrieval_test --chunking-strategy fixed_word
```

Thử sentence:

```powershell
python -m src.retrieval_test --chunking-strategy sentence
```

Thử paragraph:

```powershell
python -m src.retrieval_test --chunking-strategy paragraph
```

Thử semantic:

```powershell
python -m src.retrieval_test --chunking-strategy semantic
```

Thử parent-child:

```powershell
python -m src.retrieval_test --chunking-strategy parent_child
```

Thử sliding window:

```powershell
python -m src.retrieval_test --chunking-strategy sliding_window
```

Kết quả retrieval sẽ in thêm:

- `chunk_id`
- `chunking_strategy`
- `token_count`

Lệnh này build index tạm trong `storage/retrieval_test`, không ghi đè index chính của app.

### semantic

Gom câu/đoạn dựa trên độ giống nhau của từ khóa nội dung. Nếu độ tương đồng giữa phần đang gom và đơn vị tiếp theo thấp hơn ngưỡng, hệ thống đóng chunk và bắt đầu chunk mới.

Ưu điểm:

- Cố gắng cắt ở ranh giới đổi chủ đề.
- Giảm khả năng trộn nhiều ý khác nhau vào một chunk.

Nhược điểm:

- Bản hiện tại là heuristic lexical nhẹ, chưa dùng embedding similarity.
- Phụ thuộc chất lượng text extract và từ khóa trùng nhau.

Nên dùng khi:

- Tài liệu có nhiều đoạn đổi chủ đề nhanh.
- Muốn thử semantic boundary trước khi đầu tư semantic chunking bằng embedding.

### parent_child

Tạo child chunk nhỏ để embed/retrieve, nhưng lưu parent text lớn hơn để đưa vào context khi trả lời.

Ưu điểm:

- Retrieval vẫn sắc vì child chunk nhỏ.
- LLM nhận nhiều ngữ cảnh hơn nhờ parent chunk.

Nhược điểm:

- Metadata lớn hơn vì lưu `parent_text`.
- Context đưa vào LLM có thể dài hơn.

Nên dùng khi:

- Câu hỏi cần chi tiết cục bộ nhưng câu trả lời cần thêm đoạn xung quanh.
- Tài liệu có giải thích dài theo mạch.

### sliding_window

Chia cửa sổ từ cố định và trượt theo overlap. Đây là biến thể rõ tên của fixed word với overlap.

Ưu điểm:

- Baseline mạnh hơn fixed split không overlap.
- Ít bỏ lỡ thông tin nằm ở ranh giới chunk.

Nhược điểm:

- Tạo nhiều chunk trùng lặp.
- Tăng dung lượng index và có thể trả kết quả lặp.

Nên dùng khi:

- Cần baseline có overlap rõ ràng.
- Nội dung quan trọng hay nằm giữa hai chunk.

## Đánh Giá Định Lượng Cần Bổ Sung

Project có benchmark runner ở `src/chunking_benchmark.py`. Benchmark đọc câu hỏi từ file trong biến `EVALUATION_QUESTIONS` của `.env`, mặc định là `evaluation/kinhtevn_questions.jsonl` cho `KinhteVN.pdf`. Bộ `evaluation/chunking_questions.jsonl` là câu hỏi về tài liệu học máy cũ, chỉ dùng khi PDF đầu vào là tài liệu đó. Benchmark đo:

- `Recall@k`: câu hỏi có retrieve được trang/chunk đúng trong top-k không.
- `MRR`: chunk đúng xuất hiện ở rank bao nhiêu.
- `nDCG@k`: chất lượng thứ hạng top-k.
- Số chunk tạo ra.
- Kích thước index.
- Thời gian build index.
- Thời gian retrieval.

Bảng benchmark mong muốn:

```text
strategy    chunk_size overlap chunks recall@5 mrr   latency
fixed_word  300        50      720    0.62     0.41  1.2s
sentence    300        50      690    0.68     0.45  1.3s
paragraph   300        50      610    0.66     0.44  1.1s
recursive   300        50      675    0.71     0.48  1.4s
```
Ý nghĩa từng cột:
- strategy: thuật toán chia chunk đang test.
- chunk_size: số từ tối đa mục tiêu trong một chunk.
- overlap: số từ/ngữ cảnh lặp lại giữa hai chunk liền nhau.
- chunks: tổng số chunk tạo ra từ tài liệu.
- recall@5: trong top 5 kết quả retrieve, có tìm được chunk/trang đúng không. Càng cao càng tốt.
- mrr: chunk đúng xuất hiện ở vị trí càng cao thì điểm càng cao. Càng cao càng tốt.
- latency: thời gian retrieval hoặc build/test. Càng thấp càng tốt.

Các số trên là ví dụ định dạng, chưa phải kết quả thực nghiệm.

Chạy benchmark:

```powershell
python -m src.chunking_benchmark
```

Chạy một nhóm strategy cụ thể:

```powershell
python -m src.chunking_benchmark --strategies recursive semantic parent_child
```

Format câu hỏi benchmark:

```json
{"question": "Gradient descent là gì?", "expected_pages": [120, 121], "expected_keywords": ["gradient descent"]}
```

Quy tắc tính một chunk là liên quan (`src/benchmarks/metrics.py`):

- Có cả `expected_pages` và `expected_keywords`: chunk phải nằm ở trang nhãn **và** chứa ít nhất một từ khóa. Chunk đúng trang nhưng nói chuyện khác không được tính.
- Chỉ có `expected_pages`: chunk phải nằm ở trang nhãn.
- Chỉ có `expected_keywords`: chunk phải chứa từ khóa. Đây chỉ là đánh giá gần đúng.

`recall@k` là tỷ lệ trang nhãn được phủ bởi ít nhất một chunk liên quan. `nDCG@k` dùng số trang nhãn làm số kết quả lý tưởng, nên bỏ sót trang nhãn sẽ làm giảm điểm.

## Kết Luận

Hiện tại đã đạt phần:
- Có thuật toán.
- Có ưu/nhược điểm.
- Có khi nào sử dụng loại nào.
- Có lệnh test thủ công.
- Có benchmark runner tự động.

Chưa đạt hoàn toàn phần:
- Dataset câu hỏi chuẩn đã được gắn ground truth đầy đủ.
- Bảng kết quả thực nghiệm thật sau khi chạy benchmark trên máy.
- Bảng kết quả thực nghiệm thật.

Project đã cải thiện chunking từ fixed word đơn giản sang hệ thống có nhiều strategy và metadata truy vết. Strategy mặc định hiện là `recursive` vì cân bằng tốt giữa giữ ngữ nghĩa và kiểm soát kích thước chunk.
