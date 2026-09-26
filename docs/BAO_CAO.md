# BÁO CÁO TIẾN ĐỘ: HỆ THỐNG RAG HỎI ĐÁP TÀI LIỆU PDF TIẾNG VIỆT

Ngày: 26/09/2026

Báo cáo chỉ trình bày những phần đã cài đặt và đã đo. Mọi số liệu trong báo cáo được chạy lại vào ngày 26/09/2026 trên máy cá nhân (CPU), kết quả thô lưu trong `evaluation/results/`.

## 1. Tổng quan

Ứng dụng chạy trên console, không có giao diện đồ họa. Mỗi tầng của pipeline được cài đặt thành một package riêng trong `src/`:

```text
PDF
 -> src/documents   : đọc PDF, tạo Document / DocumentPage / DocumentImage
 -> src/chunking    : chia trang thành TextChunk (7 thuật toán)
 -> src/embeddings  : mã hóa chunk và câu hỏi thành vector
 -> src/retrieval   : lưu vector vào FAISS, tìm top-k
 -> src/app         : lọc, loại trùng, ghép context, gọi LLM local (Qwen3-0.6B)
 -> src/benchmarks  : đo chunking, embedding, vector store
```

Cách chạy:

```powershell
python -m src.main --rebuild                 # build index
python -m src.main "Câu hỏi..."              # hỏi một câu
python -m src.main                           # hỏi liên tục trong console
```

Tài liệu dùng để thử nghiệm là `data/KinhteVN.pdf` (Báo cáo Cập nhật Tình hình Kinh tế Việt Nam tháng 5/2026):

| Thuộc tính | Giá trị |
|---|---:|
| Số trang | 33 |
| Tổng số từ (tách theo khoảng trắng) | 18.303 |
| Ảnh được giữ lại | 63 |
| Ảnh bị bỏ qua (nhỏ hơn 32 px hoặc trùng) | 99 |
| Bảng phát hiện được | 4 bảng trên 4 trang |

Bộ câu hỏi đánh giá `evaluation/kinhtevn_questions.jsonl` gồm 29 câu có nhãn trang và từ khóa đúng. Bộ này chia thành 7 loại: fact (17), summary (4), forecast (2), comparison (2), policy (2), reasoning (1), risk (1). Ngoài ra có 4 câu kiểm tra hành vi: 2 câu không có đáp án trong tài liệu và 2 câu prompt injection.

## 2. Tổ chức Document

### 2.1. Các lớp dữ liệu

Tài liệu được biểu diễn theo ba cấp, cài đặt trong `src/documents/schemas.py`:

```text
Document                 toàn bộ file
 └── DocumentPage        một trang
      └── DocumentImage  một ảnh trong trang
```

| Lớp | Trường chính | Vai trò |
|---|---|---|
| `Document` | `document_id`, `source_path`, `source_name`, `document_type`, `title`, `author`, `created_at`, `metadata`, `pages` | Đại diện một file đầu vào |
| `DocumentPage` | `document_id`, `page_number`, `text`, `source_name`, `images`, `metadata` | Văn bản và thông tin của một trang |
| `DocumentImage` | `document_id`, `page_number`, `image_index`, `image_path`, `ocr_text`, `caption`, `metadata` | Một ảnh và phần chữ tìm kiếm được từ ảnh |

Các lớp là `dataclass(frozen=True)`, nghĩa là không sửa được sau khi tạo. Metadata khai báo bằng `TypedDict` để rõ các trường có thể có.

Quyết định thiết kế:

- **`document_id` lấy từ nội dung file.** Đó là 16 ký tự đầu của SHA-256 nội dung. Đổi tên hay di chuyển file không làm đổi ID, còn sửa nội dung thì ID đổi.
- **Metadata có ba cấp.** Cấp tài liệu lưu số trang, số ảnh, dung lượng và hash. Cấp trang lưu số ký tự, số từ, kích thước trang, góc xoay, tiêu đề đoán được, danh sách layout block và bảng. Cấp ảnh lưu kích thước, colorspace, định dạng, trạng thái xuất file và trạng thái OCR.
- **Chunk được tạo trong phạm vi từng trang.** Nhờ vậy mỗi câu trả lời trích được đúng số trang.

### 2.2. Chuyển đổi tài liệu đầu vào

Hàm `load_pdf_document()` trong `src/documents/pdf_loader.py` dùng PyMuPDF và làm các bước sau cho mỗi trang:

1. Lấy văn bản bằng `page.get_text("text")`.
2. Lấy layout block (vị trí, số dòng, loại block) bằng `page.get_text("dict")`.
3. Đoán tiêu đề trang từ block đầu tiên ngắn, có dạng tiêu đề.
4. Phát hiện bảng bằng `page.find_tables()`, lưu vị trí và số hàng, số cột.
5. Phát hiện ảnh bằng `page.get_images(full=True)` và xử lý như mục 2.4.
6. Ghép văn bản trang với chữ OCR hoặc caption của ảnh, nếu có, thành `chunk_text`.

Sau đó `Document.to_chunk_inputs()` chuyển mỗi trang thành một bản ghi `PageRecord` để đưa vào bước chunking.

### 2.3. Tổ chức Document Chunk

Mỗi chunk là một `TextChunk` mang đủ thông tin để truy ngược nguồn:

| Nhóm | Trường | Ý nghĩa |
|---|---|---|
| Định danh | `chunk_id`, `document_id`, `page_chunk_index`, `document_chunk_index` | ID chunk là hash của tài liệu, trang, vị trí và nội dung |
| Nguồn | `source`, `page` | Dùng để trích dẫn [tên file, trang X] |
| Nội dung | `text`, `word_count`, `char_count`, `start_word`, `end_word` | Văn bản được embed và vị trí của nó trong trang |
| Cấu trúc | `section_title`, `chunking_strategy` | Tiêu đề mục và thuật toán đã tạo chunk |
| Cha con | `parent_id`, `parent_text` | Dùng cho thuật toán parent_child: embed chunk con, đưa đoạn cha vào context |
| Ảnh | `image_refs` | Tham chiếu tới ảnh trên cùng trang |

Toàn bộ metadata chunk được lưu vào `storage/metadata.json`, song song với vector trong `storage/index.faiss`. Vector thứ i ứng với chunk thứ i.

### 2.4. Xử lý tài liệu có hình

Luồng xử lý ảnh:

```text
page.get_images()
 -> bỏ ảnh nhỏ hơn IMAGE_MIN_SIDE (mặc định 32 px) và ảnh lặp trên cùng trang
 -> xuất file ra storage/images/<document_id>/page_XXXX_image_YYYY_xref_Z.png
 -> (tùy chọn) OCR bằng pytesseract -> DocumentImage.ocr_text
 -> ghép "[Image n OCR] ..." vào text của trang trước khi chunking
 -> image_refs đi theo chunk vào vector store và phần nguồn trả về
```

Kết quả trên `KinhteVN.pdf` là 63 ảnh được xuất ra file và 99 ảnh bị bỏ qua. Ảnh bị bỏ qua chủ yếu là mask, đường kẻ và ảnh trang trí nhỏ.

OCR là tùy chọn, bật bằng `DOCUMENT_IMAGE_OCR=true`. Khi thiếu thư viện hoặc Tesseract, pipeline vẫn chạy và ghi trạng thái vào metadata ảnh. Các trạng thái là `disabled`, `missing_dependency`, `failed`, `empty` và `extracted`. Trong các lần đo của báo cáo này OCR đang tắt, nên ảnh chưa đóng góp chữ vào retrieval.

## 3. Chunking

### 3.1. Các vấn đề của chunking

| Vấn đề | Nguyên nhân | Cách xử lý trong project |
|---|---|---|
| Cắt ngang câu hoặc ý | Chia theo số từ cố định | Thuật toán sentence, paragraph, recursive |
| Chunk thiếu ngữ cảnh | Chunk nhỏ, phần giải thích nằm ở chunk kế tiếp | Overlap, hoặc parent_child đưa đoạn cha vào context |
| Chunk trộn nhiều chủ đề | Chunk quá lớn | Giảm kích thước, thuật toán semantic |
| Nhiều kết quả gần trùng nhau | Overlap lớn | Bước loại trùng theo `chunk_id` và nội dung trong `RagService` |
| Tách câu tiếng Việt sai | PDF làm mất dấu câu hoặc ngắt dòng | Recursive tự lùi về tách theo từ khi câu quá dài |
| Nội dung bị ngắt ở cuối trang | Chunk tạo trong từng trang | Chấp nhận để giữ số trang chính xác khi trích dẫn |
| Đổi cấu hình nhưng index cũ vẫn dùng | Index build từ cấu hình trước | `index_config.json` lưu cấu hình, app tự rebuild khi khác `.env` |

Kích thước chunk được tính theo số từ tách bằng khoảng trắng. Với tiếng Việt, đơn vị này gần với âm tiết, không phải token của model.

### 3.2. Bảy thuật toán đã cài đặt

Tất cả nằm trong `src/chunking/strategies.py`.

| Thuật toán | Cách hoạt động | Ưu điểm | Nhược điểm | Khi nào dùng |
|---|---|---|---|---|
| `fixed_word` | Cắt mỗi N từ, lặp lại overlap từ | Đơn giản, nhanh, dễ đoán số chunk | Cắt ngang câu và đoạn | Làm baseline |
| `sliding_window` | Cửa sổ N từ trượt với bước N trừ overlap | Ít bỏ sót thông tin nằm ở ranh giới | Nhiều chunk trùng nội dung | Baseline có overlap rõ ràng |
| `sentence` | Tách câu theo dấu `.!?`, gom câu đến N từ | Ít cắt ngang câu | Phụ thuộc dấu câu | Văn xuôi có dấu câu chuẩn |
| `paragraph` | Tách theo dòng trống, gom đoạn đến N từ | Giữ nguyên ý theo đoạn | PDF hay mất dòng trống | Tài liệu có đoạn rõ ràng |
| `recursive` | Ưu tiên đoạn, đoạn quá dài thì tách câu, câu quá dài thì tách từ | Cân bằng giữa giữ ý và kiểm soát kích thước | Phức tạp hơn, phụ thuộc chất lượng trích xuất | Mặc định thực dụng cho văn bản |
| `semantic` | Gom câu, cắt khi độ trùng từ khóa (Jaccard) với phần trước dưới 0,12 | Cắt ở chỗ đổi chủ đề | Heuristic theo từ vựng, chưa dùng embedding; tạo nhiều chunk | Tài liệu đổi chủ đề nhanh |
| `parent_child` | Chunk cha 2N từ, chia thành chunk con N từ; embed con, đưa cha vào context | Tìm kiếm sắc nét, LLM nhận nhiều ngữ cảnh | Metadata lớn, context dài hơn | Câu hỏi chi tiết nhưng cần đoạn xung quanh để trả lời |

### 3.3. Đánh giá bằng số

Điều kiện đo: model `multilingual-e5-small`, FAISS Flat, top-k bằng 5, overlap 50 từ, 29 câu hỏi.

Một chunk được tính là đúng khi nằm ở trang có nhãn **và** chứa ít nhất một từ khóa có nhãn. Các chỉ số:

- **Hit@5**: tỷ lệ câu hỏi có ít nhất một chunk đúng trong top 5.
- **Precision@5**: tỷ lệ chunk đúng trong 5 chunk trả về.
- **Recall@5**: tỷ lệ trang có nhãn được phủ bởi chunk đúng.
- **MRR**: trung bình của 1 chia cho thứ hạng chunk đúng đầu tiên.
- **nDCG@5**: chất lượng thứ hạng, phạt khi chunk đúng nằm thấp.

**Bảng 1. So sánh 7 thuật toán, kích thước 300 từ**

| Thuật toán | Số chunk | Hit@5 | Precision@5 | Recall@5 | MRR | nDCG@5 | Build (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_word | 88 | 0,862 | 0,228 | 0,845 | 0,644 | 0,681 | 8,4 |
| sliding_window | 88 | 0,862 | 0,228 | 0,845 | 0,644 | 0,681 | 8,2 |
| sentence | 88 | **0,931** | 0,221 | **0,914** | 0,702 | 0,759 | 8,3 |
| paragraph | 106 | 0,897 | 0,248 | 0,879 | 0,725 | 0,761 | 9,2 |
| recursive | 90 | **0,931** | 0,248 | **0,914** | 0,708 | 0,748 | 8,9 |
| semantic | 191 | 0,897 | 0,241 | 0,897 | **0,759** | **0,790** | 7,5 |
| parent_child | 100 | 0,897 | **0,276** | 0,879 | 0,666 | 0,707 | 8,4 |

Thời gian tìm kiếm của mọi thuật toán đều khoảng 0,12 ms mỗi câu, nên không khác biệt.

**Bảng 2. Ảnh hưởng của kích thước chunk**

| Thuật toán | Kích thước | Số chunk | Hit@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_word | 150 | 186 | 0,862 | 0,290 | 0,734 | 0,765 |
| fixed_word | 300 | 88 | 0,862 | 0,228 | 0,644 | 0,681 |
| fixed_word | 500 | 57 | 0,897 | 0,214 | 0,748 | 0,782 |
| recursive | 150 | 217 | 0,897 | 0,331 | **0,771** | **0,797** |
| recursive | 300 | 90 | **0,931** | 0,248 | 0,708 | 0,748 |
| recursive | 500 | 59 | 0,862 | 0,207 | 0,727 | 0,762 |
| parent_child | 150 | 215 | 0,862 | **0,338** | 0,749 | 0,762 |
| parent_child | 300 | 100 | 0,897 | 0,276 | 0,666 | 0,707 |
| parent_child | 500 | 61 | 0,897 | 0,228 | 0,748 | 0,774 |

Nhận xét:

- **Tôn trọng ranh giới câu có lợi.** Sentence và recursive đạt Hit@5 cao nhất (0,931), hơn fixed_word 7 điểm phần trăm với cùng số chunk.
- **Semantic xếp hạng tốt nhất nhưng tốn chunk nhất.** MRR và nDCG cao nhất, nhưng số chunk gấp đôi các thuật toán khác. Nguyên nhân là ngưỡng Jaccard cắt nhiều chunk nhỏ.
- **Chunk nhỏ tăng Precision, chunk vừa tăng Hit.** Với recursive, 150 từ cho MRR và nDCG tốt nhất, còn 300 từ cho Hit@5 tốt nhất.
- **Parent_child có Precision cao nhất ở mỗi kích thước.** Lợi thế thật của nó là context đầy đủ hơn cho LLM, và điều đó thể hiện ở đánh giá end-to-end tại mục 7.
- **Fixed_word và sliding_window cho kết quả giống hệt.** Trong code, hai thuật toán dùng cùng một hàm cắt từ có overlap, chỉ khác tên.

Hạn chế: bộ đánh giá chỉ có 29 câu, nên một câu đúng hay sai làm thay đổi khoảng 3,4 điểm phần trăm. Các chênh lệch nhỏ hơn mức này chưa đủ để kết luận.

Lựa chọn hiện tại: app dùng `parent_child` với 300 từ, vì kết quả end-to-end tốt hơn recursive (mục 7). Nếu chỉ xét retrieval, `recursive` 300 từ là lựa chọn cân bằng nhất.

## 4. Embedding

### 4.1. Các model đã đánh giá

| Model | Số chiều | Độ dài tối đa | Prefix | Đặc điểm |
|---|---:|---:|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 token | không | Nhẹ, train cho paraphrase, không chuyên retrieval |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 128 token | không | Lớn hơn MiniLM, cũng train cho paraphrase |
| `intfloat/multilingual-e5-small` | 384 | 512 token | `query:` / `passage:` | Train riêng cho retrieval, nhỏ |
| `intfloat/multilingual-e5-base` | 768 | 512 token | `query:` / `passage:` | Train riêng cho retrieval, lớn hơn |

Tất cả vector được chuẩn hóa về độ dài 1 trong `EmbeddingService`, nên tích vô hướng trong FAISS bằng cosine similarity.

### 4.2. Kết quả đo

Điều kiện đo: chunking parent_child 300 từ (100 chunk), FAISS Flat, top-k bằng 5, 29 câu hỏi, chạy trên CPU.

**Bảng 3. So sánh 4 model embedding**

| Model | Hit@5 | Precision@5 | Recall@5 | MRR | nDCG@5 | Tải model (s) | Embed 100 chunk (s) | Encode 1 câu hỏi (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM-L12 | 0,793 | 0,214 | 0,776 | 0,581 | 0,616 | 2,9 | **2,9** | **19** |
| mpnet-base | 0,828 | 0,221 | 0,810 | 0,582 | 0,626 | 3,9 | 9,4 | 53 |
| e5-small | 0,897 | 0,276 | 0,879 | 0,666 | 0,707 | **2,7** | 9,1 | 24 |
| e5-base | **0,966** | **0,310** | **0,931** | **0,685** | **0,725** | 4,4 | 26,2 | 49 |

Nhận xét:

- **Hai model E5 tốt hơn rõ rệt cho tiếng Việt.** e5-small hơn MiniLM cùng kích thước 10 điểm phần trăm Hit@5. Lý do là E5 được train cho bài toán tìm kiếm câu hỏi và đoạn văn, còn MiniLM và mpnet được train cho câu gần nghĩa.
- **MiniLM và mpnet bị cắt chunk.** Hai model này chỉ đọc 128 token, trong khi chunk 300 từ tiếng Việt dài hơn nhiều. Phần sau của chunk bị bỏ khi embed.
- **e5-base tốt nhất nhưng chậm gần 3 lần e5-small khi build index.** Encode một câu hỏi mất khoảng 49 ms so với 24 ms.

Kết luận cho tiếng Việt: chọn **e5-base** khi ưu tiên chất lượng. Chọn **e5-small** khi cần nhanh và nhẹ, chất lượng chỉ kém khoảng 7 điểm Hit@5. App hiện dùng e5-small vì chạy trên CPU cùng LLM local.

Lưu ý khi dùng E5: bắt buộc thêm prefix `query: ` cho câu hỏi và `passage: ` cho chunk. Điểm cosine của E5 dồn trong khoảng 0,77 đến 0,92 ngay cả với câu hỏi không liên quan, nên ngưỡng lọc phải chỉnh theo model. App dùng 0,795 cho E5.

## 5. Vector Store

### 5.1. Tổ chức lưu trữ

Vector store dùng FAISS, cài đặt trong `src/retrieval/vector_store.py`. Mỗi index gồm ba file:

| File | Nội dung |
|---|---|
| `index.faiss` | Vector đã chuẩn hóa |
| `metadata.json` | Metadata của từng chunk, cùng thứ tự với vector |
| `index_config.json` | Loại index, tham số FAISS, hash PDF, model embedding, prefix, tham số chunking |

Khi chạy app, cấu hình trong `index_config.json` được so với `.env`. Nếu khác nhau, index tự build lại. Nhờ vậy không bao giờ tìm kiếm bằng vector của model cũ.

### 5.2. Ba thuật toán index

| Index | Cơ chế | Ưu điểm | Nhược điểm | Khi nào dùng |
|---|---|---|---|---|
| Flat (`IndexFlatIP`) | So câu hỏi với mọi vector | Chính xác tuyệt đối, không cần train | Thời gian tăng tuyến tính theo số vector | Dưới vài trăm nghìn vector, làm mốc so sánh |
| HNSW (`IndexHNSWFlat`, M=32) | Duyệt đồ thị nhiều tầng | Nhanh ở quy mô lớn, không cần train | Tốn RAM cho đồ thị, kết quả gần đúng | Nhiều vector, cần độ trễ thấp |
| IVF (`IndexIVFFlat`, nlist=64, nprobe=8) | Phân cụm k-means, chỉ tìm trong nprobe cụm | Chỉnh được tốc độ và độ chính xác qua nprobe | Cần train, có thể bỏ sót cụm đúng | Dữ liệu rất lớn, có đủ dữ liệu để train |

### 5.3. Kết quả đo

Điều kiện đo: e5-small, parent_child 300 từ, 100 vector 384 chiều, 29 câu hỏi.

**Bảng 4. So sánh ba index FAISS**

| Index | Hit@5 | Recall@5 | MRR | nDCG@5 | Build (ms) | Tìm kiếm (ms/câu) |
|---|---:|---:|---:|---:|---:|---:|
| Flat | **0,897** | **0,879** | **0,666** | **0,707** | **0,9** | **0,053** |
| HNSW | **0,897** | **0,879** | **0,666** | **0,707** | 8,9 | 0,090 |
| IVF | 0,862 | 0,845 | 0,651 | 0,688 | 5,6 | 0,074 |

Nhận xét:

- **HNSW cho kết quả giống hệt Flat.** Với 100 vector, đồ thị HNSW gần như duyệt hết các điểm.
- **IVF mất một câu hỏi đúng.** FAISS cảnh báo cần ít nhất 2.496 điểm để train 64 cụm, trong khi chỉ có 100 điểm. Cụm bị train kém nên có lúc bỏ sót chunk đúng.
- **Flat nhanh nhất ở quy mô này.** HNSW và IVF chỉ có lợi khi số vector lên tới hàng trăm nghìn.

Kết luận: với tài liệu vài trăm chunk, **Flat** là lựa chọn đúng. App đang dùng Flat.

## 6. Retrieval trong ứng dụng

Luồng trả lời trong `src/app/rag_service.py`:

1. Encode câu hỏi với prefix `query: `.
2. Tìm top 5 chunk trên FAISS bằng cosine similarity.
3. Loại chunk có điểm dưới `SIMILARITY_THRESHOLD`. Nếu không còn chunk nào, trả lời "Tôi không tìm thấy đủ thông tin trong tài liệu để trả lời câu hỏi này." mà không gọi LLM.
4. Loại chunk trùng theo `chunk_id` và theo nội dung đoạn cha.
5. Ghép context, mỗi khối có tiêu đề `[tên file, trang X]` và dùng `parent_text` nếu có. Tổng context giới hạn 12.000 ký tự.
6. Gọi Qwen3-0.6B chạy local với prompt tiếng Việt. Prompt yêu cầu chỉ dùng context, trích nguồn theo trang, và từ chối khi thiếu thông tin.
7. Trả về câu trả lời kèm danh sách nguồn: trang, điểm similarity, chunk_id, ảnh liên quan.

## 7. Đánh giá độ chính xác của hệ thống RAG

### 7.1. Phương pháp

Hệ thống được đánh giá ở hai tầng:

| Tầng | Câu hỏi đánh giá | Chỉ số |
|---|---|---|
| Retrieval | Có tìm đúng đoạn văn không? | Hit@k, Precision@k, Recall@k, MRR, nDCG@k (mục 3 đến 5) |
| End-to-end | Câu trả lời cuối có đúng và an toàn không? | Page hit, tỷ lệ từ khóa đúng trong câu trả lời, tỷ lệ từ chối đúng, chống prompt injection |

Script `evaluation/run_kinhtevn_rag_eval.py` chạy từng câu hỏi qua toàn bộ pipeline, kể cả LLM. Mỗi câu được chấm như sau:

- **Page hit**: nguồn trả về có chứa ít nhất một trang có nhãn hay không.
- **Keyword hit ratio**: tỷ lệ từ khóa có nhãn xuất hiện trong câu trả lời.
- **No-context pass**: câu hỏi không có đáp án thì hệ thống phải trả đúng câu từ chối.

### 7.2. Kết quả

**Bảng 5. Kết quả end-to-end với 29 câu hỏi có nhãn**

| Cấu hình | Page hit | Keyword hit ratio trung bình |
|---|---:|---:|
| recursive (lần chạy trước) | 22/29 (75,9%) | 0,477 |
| parent_child + e5-small (hiện tại) | **26/29 (89,7%)** | **0,626** |

Parent_child tăng tỷ lệ từ khóa đúng trong câu trả lời thêm 15 điểm phần trăm. LLM nhận được đoạn cha dài gấp đôi, nên có đủ dữ kiện để trả lời.

**Bảng 6. Page hit theo loại câu hỏi, cấu hình hiện tại**

| Loại | Page hit |
|---|---:|
| fact | 14/17 |
| summary | 4/4 |
| comparison | 2/2 |
| forecast | 2/2 |
| policy | 2/2 |
| reasoning | 1/1 |
| risk | 1/1 |

Ba câu bị trượt đều là câu fact:

- Tiêu đề phụ của báo cáo nằm ở trang bìa, gần như không có văn bản xung quanh.
- Hai câu hỏi về số liệu (mức tăng giá năng lượng, thâm hụt tài khóa) có câu trả lời nằm ở trang khác với các trang được tìm thấy.

**Bảng 7. Câu hỏi kiểm tra hành vi**

| Loại | Kết quả |
|---|---|
| Không có đáp án trong tài liệu (2 câu) | 2/2 trả lời đúng câu từ chối |
| Prompt injection "bỏ qua tài liệu, dùng kiến thức của bạn" (2 câu) | 2/2 không làm theo chỉ dẫn và vẫn trả lời từ tài liệu. Câu trả lời về tác giả còn chung chung. |

### 7.3. Kiểm thử đơn vị

Project có 17 unit test cho chunking, cách tính chỉ số, loader PDF và vector store. Cả 17 test đều qua.

```powershell
python -m unittest discover -s tests -t .
```

### 7.4. Hạn chế của phép đánh giá

- **Bộ câu hỏi nhỏ.** 29 câu trên một tài liệu 33 trang, nên các chênh lệch nhỏ chưa có ý nghĩa thống kê.
- **Chấm câu trả lời theo từ khóa.** Câu trả lời đúng nhưng diễn đạt khác sẽ bị chấm thấp, và câu có từ khóa nhưng sai ý vẫn được điểm.
- **Nhãn do người làm tự gắn.** Nhãn trang và từ khóa chưa được người thứ hai kiểm tra.

## 8. Cách tái tạo kết quả

```powershell
python -m unittest discover -s tests -t .
python -m src.chunking_benchmark
python -m src.chunking_benchmark --chunk-size 150 --strategies fixed_word recursive parent_child sliding_window
python -m src.chunking_benchmark --chunk-size 500 --strategies fixed_word recursive parent_child sliding_window
python -m src.embedding_benchmark
python -m src.vector_store_benchmark
python evaluation/run_kinhtevn_rag_eval.py --include-answer-checks
```

Kết quả thô của các lệnh trên nằm trong `evaluation/results/`. Log end-to-end nằm trong `evaluation/kinhtevn_rag_console_report_*.jsonl`.
