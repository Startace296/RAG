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
python -m src.app.main --rebuild
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
python -m src.scripts.retrieval_test --chunking-strategy recursive
```

Thử baseline fixed word:

```powershell
python -m src.scripts.retrieval_test --chunking-strategy fixed_word
```

Thử sentence:

```powershell
python -m src.scripts.retrieval_test --chunking-strategy sentence
```

Thử paragraph:

```powershell
python -m src.scripts.retrieval_test --chunking-strategy paragraph
```

Thử semantic:

```powershell
python -m src.scripts.retrieval_test --chunking-strategy semantic
```

Thử parent-child:

```powershell
python -m src.scripts.retrieval_test --chunking-strategy parent_child
```

Thử sliding window:

```powershell
python -m src.scripts.retrieval_test --chunking-strategy sliding_window
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

Project có benchmark runner ở `src/benchmarks/chunking_benchmark.py`. Benchmark đọc câu hỏi từ file trong biến `EVALUATION_QUESTIONS` của `.env`, mặc định là `evaluation/kinhtevn_questions.jsonl` cho `KinhteVN.pdf`. Benchmark đo:

- `Recall@k`: câu hỏi có retrieve được trang/chunk đúng trong top-k không.
- `MRR`: chunk đúng xuất hiện ở rank bao nhiêu.
- `nDCG@k`: chất lượng thứ hạng top-k.
- Số chunk tạo ra.
- Kích thước index.
- Thời gian build index.
- Thời gian retrieval.

### Kết Quả Thực Nghiệm

Cấu hình: `KinhteVN.pdf` (31 trang), 29 câu hỏi trong `evaluation/kinhtevn_questions.jsonl`, embedding `intfloat/multilingual-e5-small` trên CPU, FAISS `flat`, dense retrieval (không hybrid), `top_k=5`.

```text
strategy        chunk_size  overlap  chunks  queries   hit@5  precision@5  recall@5     mrr  ndcg@5  build_s  retrieval_ms
--------------  ----------  -------  ------  -------  ------  -----------  --------  ------  ------  -------  ------------
fixed_word             300       50      84       29  0.8966       0.2345    0.8793  0.7379  0.7631   8.9217        0.1514
sentence               300       50      85       29  0.9310       0.2345    0.9138  0.7500  0.7779   7.9184        0.1653
paragraph              300       50     108       29  0.9655       0.3034    0.9655  0.8575  0.8660  10.0416        0.1529
recursive              300       50     107       29  0.9655       0.3034    0.9655  0.8575  0.8660   9.7645        0.1591
semantic               300       50     185       29  0.9310       0.2483    0.9138  0.8017  0.8231   7.9372        0.1901
parent_child           300       50      95       29  0.8966       0.2552    0.8793  0.7580  0.7765   8.3771        0.1742
sliding_window         300       50      84       29  0.8966       0.2345    0.8793  0.7379  0.7631   7.8338        0.1594
```

Ý nghĩa từng cột:
- `strategy`: thuật toán chia chunk đang test.
- `chunk_size`: số từ tối đa mục tiêu trong một chunk.
- `overlap`: số từ/ngữ cảnh lặp lại giữa hai chunk liền nhau.
- `chunks`: tổng số chunk tạo ra từ tài liệu.
- `queries`: số câu hỏi đánh giá.
- `hit@5`: tỷ lệ câu hỏi có ít nhất một chunk liên quan trong top 5.
- `precision@5`: tỷ lệ chunk liên quan trong 5 chunk trả về.
- `recall@5`: tỷ lệ trang nhãn được phủ bởi ít nhất một chunk liên quan trong top 5.
- `mrr`: chunk liên quan đầu tiên đứng càng cao thì điểm càng cao (rank 1 = 1.0, rank 2 = 0.5, ...).
- `ndcg@5`: chất lượng thứ hạng của cả top 5.
- `build_s`: thời gian chunk + embed + build index (giây).
- `retrieval_ms`: thời gian tìm kiếm trung bình mỗi câu hỏi (mili giây).

Quy đổi ra số câu hỏi (mỗi câu ≈ 0.0345 `hit@5`):

| Strategy | Trúng top 5 / 29 | MRR |
| --- | --- | --- |
| recursive | 28 | 0.858 |
| paragraph | 28 | 0.858 |
| semantic | 27 | 0.802 |
| sentence | 27 | 0.750 |
| parent_child | 26 | 0.758 |
| fixed_word | 26 | 0.738 |
| sliding_window | 26 | 0.738 |

Nhận xét:

- **Tôn trọng ranh giới đoạn văn giúp ích rõ rệt.** `recursive`/`paragraph` nâng MRR từ 0.74 lên 0.86 so với baseline chia theo từ: chunk đúng thường lên rank 1 thay vì rank 2–3. `hit@5` chỉ chênh 2 câu, nhưng MRR và nDCG cùng chỉ một hướng.
- **`recursive` bằng `paragraph`** vì recursive ưu tiên chia theo đoạn và chỉ chia nhỏ khi đoạn vượt `CHUNK_SIZE`. Với PDF này chỉ có khoảng một đoạn như vậy (107 so với 108 chunk). `recursive` vẫn an toàn hơn vì `paragraph` có thể tạo chunk vượt giới hạn (ví dụ chunk 411 từ ở trang 9).
- **`fixed_word` và `sliding_window` giống nhau hoàn toàn** vì hiện dùng chung một nhánh code (`src/chunking/strategies.py`) với cùng `CHUNK_OVERLAP`.
- **`parent_child` gần như không hơn baseline** vì benchmark chỉ đo retrieval trên child chunk, vốn được chia theo từ. Lợi ích đưa parent vào context cho LLM không được đo ở đây.
- **`semantic`** có MRR tốt thứ hai nhưng tạo nhiều chunk nhất (185), dễ sinh chunk vụn như mục lục.
- **`precision@5` thấp (0.23–0.30) là bình thường**: mỗi câu hỏi chỉ có 1–2 trang nhãn, nên phần lớn top 5 luôn bị tính là không liên quan. Chỉ dùng để so sánh tương đối.
- **Tốc độ không phải yếu tố quyết định**: retrieval ~0.15–0.19 ms; `build_s` chủ yếu là thời gian embed, chênh lệch giữa các strategy nằm trong nhiễu đo.
- **Độ tin cậy**: với 29 câu hỏi, chênh 1–2 câu chưa có ý nghĩa thống kê. Kết luận "recursive/paragraph tốt hơn nhóm chia theo từ" là đáng tin; thứ tự giữa `semantic` và `sentence` thì chưa.

Chạy benchmark:

```powershell
python -m src.benchmarks.chunking_benchmark
```

Chạy một nhóm strategy cụ thể:

```powershell
python -m src.benchmarks.chunking_benchmark --strategies recursive semantic parent_child
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
- Có bảng kết quả thực nghiệm thật trên `KinhteVN.pdf` với 29 câu hỏi có ground truth.

Chưa đạt hoàn toàn phần:
- Bộ câu hỏi còn nhỏ (29 câu) và chỉ cho một tài liệu, nên chênh lệch nhỏ giữa các strategy chưa có ý nghĩa thống kê.
- Chưa benchmark các giá trị `CHUNK_SIZE`/`CHUNK_OVERLAP` khác, và chưa đo với hybrid retrieval.
- `sliding_window` chưa khác biệt thực sự với `fixed_word`.

Project đã cải thiện chunking từ fixed word đơn giản sang hệ thống có nhiều strategy và metadata truy vết. Strategy mặc định là `recursive`. Benchmark cho thấy nó đạt kết quả tốt nhất (hit@5 0.9655, MRR 0.8575, bằng `paragraph`) và kiểm soát kích thước chunk tốt hơn `paragraph`.
