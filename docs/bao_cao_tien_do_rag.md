# BÁO CÁO TIẾN ĐỘ XÂY DỰNG HỆ THỐNG RAG

Ngày tổng hợp: 22/09/2026.

Báo cáo mô tả hệ thống hiện tại, kết quả đo trực tiếp bước chia chunk trên `KinhteVN.pdf` và các log đánh giá đã lưu.

## 1. Mục tiêu hệ thống RAG

Mục tiêu của hệ thống là xây dựng một pipeline hỏi đáp trên tài liệu PDF tiếng Việt, trong đó câu trả lời có thể truy ngược về trang và chunk nguồn. Hệ thống ưu tiên khả năng chạy local, dễ kiểm thử và đo được ảnh hưởng của từng lựa chọn chunking, embedding và vector index.

Tiêu chí đánh giá được chia thành bốn nhóm: chất lượng trích xuất tài liệu, chất lượng retrieval, độ đúng của câu trả lời và chi phí vận hành. Các số liệu được ghi theo corpus và bộ câu hỏi đang sử dụng.

Phạm vi hiện tại tập trung vào PDF `KinhteVN.pdf`, dense retrieval bằng Sentence Transformers và FAISS local. OCR, bảng có cấu trúc, truy xuất đa phương thức, hybrid search và reranker đang được khảo sát trong quá trình phát triển.

## 2. Kiến trúc pipeline tổng thể

```text
PDF
  -> kiểm tra file, hash, metadata
  -> trích xuất page text, layout, bảng, ảnh
  -> Document / DocumentPage / DocumentImage
  -> PageRecord và làm giàu OCR/caption
  -> TextChunk theo strategy
  -> embedding chunk và query
  -> FAISS + metadata JSON
  -> top-k retrieval
  -> context assembly -> local Qwen3 -> câu trả lời + nguồn
```

Pipeline được tách thành các lớp xử lý tài liệu, chunking, embedding, retrieval, ứng dụng và đánh giá. Hiện ứng dụng chính dùng dense retrieval; kết quả top-k được gom context trước khi gửi cho LLM.

## 3. Tổ chức Document, Page, Image và Chunk

Phần này mô tả cấu trúc dữ liệu và cách tài liệu được chuyển qua pipeline; không đặt các mô hình Document, Page và Image lên bàn cân so sánh.

### 3.1. Document, Page và Image

Hệ thống đã xây dựng mô hình tài liệu theo ba cấp: `Document -> DocumentPage -> DocumentImage`. Cấu trúc này tách thông tin toàn tài liệu, nội dung từng trang và thông tin hình ảnh, đồng thời giữ liên kết nguồn để truy vết kết quả trả lời.

```text
Document: định danh, nguồn, thuộc tính tài liệu
  +-- DocumentPage: số trang, văn bản, bố cục, thông tin bảng
        +-- DocumentImage: tham chiếu ảnh, đường dẫn, OCR, caption
```

| Thành phần | Các trường chính | Vai trò |
|---|---|---|
| `Document` | `document_id`, `source_path`, `source_name`, `document_type`, `title`, `author`, `created_at`, `metadata`, `pages` | Đại diện tài liệu đầu vào và quản lý danh sách trang |
| `DocumentPage` | `document_id`, `page_number`, `text`, `source_name`, `images`, `metadata` | Lưu văn bản và thông tin của một trang |
| `DocumentImage` | `document_id`, `page_number`, `image_index`, `image_path`, `ocr_text`, `caption`, `metadata` | Lưu hình ảnh và phần văn bản có thể tìm kiếm từ ảnh |

Ba thành phần được tổ chức theo mô hình dữ liệu bất biến ở cấp đối tượng sau khi khởi tạo; các danh sách và metadata bên trong vẫn được lưu dưới dạng cấu trúc dữ liệu thông thường.

Định danh tài liệu lấy 16 ký tự đầu của SHA-256 nội dung file, đồng thời lưu hash đầy đủ trong metadata. Cùng nội dung file sẽ tạo cùng định danh khi đổi tên hoặc di chuyển file. Đây là cơ sở nhận diện nội dung; hệ thống chưa có quy trình quản lý phiên bản tài liệu hoàn chỉnh.

Metadata gồm ba cấp: cấp tài liệu lưu số trang, số ảnh, dung lượng và hash; cấp trang lưu số ký tự, số đơn vị tách bằng khoảng trắng, kích thước, góc xoay, tiêu đề dự đoán, layout block và thông tin bảng; cấp ảnh lưu kích thước, định dạng, trạng thái xuất ảnh và OCR.

Mô hình tài liệu được chuyển thành bản ghi theo trang trước khi đưa vào bước chunking.

### 3.3. Document Chunk

#### Kết quả đã thực hiện

Đơn vị đưa vào embedding là chunk văn bản. Mỗi chunk có nội dung cùng metadata để truy ngược về tài liệu, trang và chiến lược chia.

| Nhóm thông tin | Các trường | Ý nghĩa |
|---|---|---|
| Định danh | `chunk_id`, `document_id`, `chunk_index` | Nhận diện chunk và tài liệu nguồn; chỉ số chunk được tạo trong từng trang |
| Nội dung và nguồn | `text`, `source`, `page` | Văn bản được embedding và vị trí trích dẫn |
| Vị trí, độ dài | `start_word`, `end_word`, `token_count`, `char_count` | Theo dõi phạm vi và kích thước chunk |
| Cấu trúc | `section_title`, `chunking_strategy` | Ghi nhận tiêu đề suy đoán và phương pháp chia |
| Quan hệ cha/con | `parent_id`, `parent_text` | Truy xuất chunk con và sử dụng đoạn cha làm ngữ cảnh |
| Hình ảnh | `image_refs` | Giữ tham chiếu ảnh từ trang nguồn |

Chunk hiện được tạo riêng trong từng trang. Cách này giúp trích dẫn rõ ràng nhưng có thể làm đứt nội dung nối tiếp qua hai trang. Với `parent_child`, hệ thống embedding nội dung con và có thể dùng `parent_text` khi tạo context cho LLM; ứng dụng cũng có bước loại bớt kết quả trùng.

#### Đặc điểm hiện tại

Trường `token_count` thực chất đang dùng `len(text.split())`, tức đếm đơn vị phân cách bằng khoảng trắng. Với tiếng Việt, đơn vị này thường gần âm tiết và không tương đương từ đã phân đoạn hoặc token của embedding model. Vì vậy, báo cáo này gọi phép đo đó là “đơn vị khoảng trắng”, viết gọn là “từ” trong bảng đo.

Trong phiên bản hiện tại, `image_refs` lấy các ảnh của cả trang và chưa xác định ảnh nào thực sự liên quan đến từng chunk.

## 4. Chuyển đổi tài liệu đầu vào

### 4.1. Luồng đã xây dựng

```text
File PDF
  -> Kiểm tra file, tính hash, đọc metadata
  -> Trích xuất text, layout, thông tin bảng và ảnh theo trang
  -> Document / DocumentPage / DocumentImage
  -> PageRecord: text trang + OCR/caption nếu có
  -> TextChunk theo chiến lược được chọn
  -> embedding: vector float32 được chuẩn hóa
  -> FAISS + metadata JSON
  -> Truy xuất top-k -> tạo context -> LLM trả lời kèm nguồn
```

Tài liệu được chuyển thành mô hình gồm tài liệu, trang và ảnh, sau đó thành các bản ghi theo trang. Trang có ảnh nhưng không có text vẫn có thể được giữ trong mô hình tài liệu. Tuy nhiên, nếu không có text, OCR hoặc caption để tìm kiếm, trang đó không được đưa vào danh sách đầu vào chunking.

Kết quả kiểm tra trực tiếp tài liệu `KinhteVN.pdf` trong lần tổng hợp này:

| Chỉ tiêu | Kết quả |
|---|---:|
| Dung lượng file | 2.264.229 byte |
| Trang được loader giữ lại | 33 |
| Trang có nội dung để đưa vào chunking khi tắt OCR | 33 |
| Tổng đơn vị khoảng trắng trong văn bản | 18.303 |
| Bản ghi ảnh do loader thu thập | 162 |
| Bảng được bộ phát hiện ghi nhận | 4 |

162 là số bản ghi ảnh được thu thập, có thể bao gồm ảnh lặp hoặc dùng chung tài nguyên; không đồng nghĩa với 162 hình minh họa độc lập. Bốn bảng là kết quả phát hiện tự động, chưa phải số bảng đã kiểm chứng thủ công.

### 4.2. Xử lý ảnh trong quá trình chuyển đổi

#### Kết quả đã thực hiện

Hệ thống liệt kê ảnh theo trang, lấy metadata và xuất ảnh ra thư mục theo định danh tài liệu khi bật chức năng xuất ảnh. Chức năng xuất ảnh và OCR được điều khiển bằng cấu hình; mặc định xuất ảnh được bật và OCR được tắt.

Khi có văn bản OCR, nội dung này được đưa vào văn bản trang trước khi chia chunk. Trạng thái thiếu thư viện, lỗi OCR hoặc không có văn bản được ghi vào metadata. Tham chiếu ảnh được mang tiếp qua chunk, vector store và kết quả tìm kiếm.

#### Trạng thái hiện tại

- Có trường `caption` và logic sử dụng caption, nhưng loader chưa tự sinh mô tả hình bằng mô hình thị giác.
- Chưa embedding trực tiếp ảnh hoặc thực hiện truy xuất đa phương thức.
- OCR hiện xử lý ảnh trích xuất; chưa có quy trình render toàn trang và OCR toàn trang để xử lý các trường hợp PDF scan phức tạp.
- Lệnh OCR chưa chỉ định `lang="vie"` hoặc `vie+eng`; chưa thể xác nhận OCR đã được tối ưu cho tiếng Việt.
- Chưa có benchmark độ chính xác OCR, đọc biểu đồ, liên kết ảnh với caption hoặc bảo toàn cấu trúc bảng trong ảnh.
- OCR hiện chỉ chuyển chữ trong ảnh thành text; chưa có bước hiểu trực tiếp ý nghĩa biểu đồ.

## 5. Chunking: so sánh thuật toán và đánh giá bằng số

### 5.1. Các vấn đề đang gặp

Chunk nhỏ dễ thiếu ngữ cảnh; chunk lớn có thể trộn nhiều ý và vượt giới hạn token của model. Overlap giữ thông tin ở ranh giới nhưng tăng dữ liệu lặp và số vector. Với PDF tiếng Việt, việc chia còn chịu ảnh hưởng của lỗi xuống dòng, nhiều cột, dấu câu, bảng và header/footer.

Hệ thống hiện có bảy chiến lược. `fixed_word` và `sliding_window` cho cùng nội dung chunk khi dùng cùng tham số. `semantic` hiện dùng độ tương đồng từ vựng, chưa phải semantic chunking bằng embedding và chưa sử dụng overlap.

### 5.2. Các thuật toán được so sánh

| Phương pháp | Cách thực hiện hiện tại | Ưu điểm | Hạn chế | Phạm vi sử dụng |
|---|---|---|---|---|
| `fixed_word` | Cửa sổ theo số từ, có overlap | Đơn giản, nhanh, kiểm soát độ dài | Có thể cắt ngang câu/ý | Baseline và dữ liệu ít cấu trúc |
| `sentence` | Tách theo dấu câu, gom câu; câu quá dài chia theo từ | Thường giữ trọn câu | Regex đơn giản, nhạy với dấu câu lỗi | Văn xuôi có câu rõ ràng |
| `paragraph` | Tách đoạn; fallback sang câu khi không có ranh giới đoạn | Giữ mạch đoạn khi trích xuất tốt | PDF có đoạn lỗi; có thể tạo chunk quá ngắn hoặc quá dài | Tài liệu có đoạn được chuẩn hóa |
| `recursive` | Ưu tiên đoạn, rồi câu, rồi từ | Kết hợp cấu trúc và độ dài mục tiêu | Logic gom hiện chưa bảo đảm trần độ dài | Baseline cấu trúc cho PDF, sau khi kiểm tra đầu ra |
| `semantic` | Ngắt theo độ trùng từ vựng, ngưỡng mặc định 0,12 | Chi phí thấp hơn việc thêm model cho ranh giới | Không hiểu đồng nghĩa; dễ chia vụn; chưa có overlap | Thử nghiệm heuristic, cần đối chứng |
| `parent_child` | Parent có kích thước mục tiêu gấp đôi child | Truy xuất chi tiết, cung cấp ngữ cảnh rộng hơn | Lặp parent text trong metadata, tăng context | Câu hỏi cần chi tiết kèm giải thích xung quanh |
| `sliding_window` | Dùng cùng hàm với `fixed_word` | Giữ ngữ cảnh biên nhờ overlap | Không phải thuật toán độc lập trong bản hiện tại | Dùng như tên khác của baseline cửa sổ |

Recursive chia theo thứ tự ranh giới đoạn, câu và từ để giữ các phần liên quan khi có thể.

### 5.3. Số đo trực tiếp trên tài liệu dự án

Thực hiện ngày 22/09/2026 trên cùng tài liệu và cùng bộ tham số: `chunk_size=300`, `chunk_overlap=50`, `min_chunk_size=50`. Tắt xuất ảnh và OCR trong tiến trình đo. Mỗi chiến lược chạy 5 lần; thời gian dưới đây là trung vị riêng của bước chia chunk, không bao gồm embedding, xây index hoặc gọi LLM.

| Chiến lược | Số chunk | Từ/chunk TB | Min / Max | Chunk >300 từ | Chunk <50 từ | Hệ số lượng text | Trung vị chia (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fixed_word` | 88 | 239,24 | 12 / 300 | 0 | 5 | 1,150 | 6,269 |
| `sentence` | 88 | 235,11 | 12 / 300 | 0 | 6 | 1,130 | 15,444 |
| `paragraph` | 106 | 202,25 | 1 / 463 | 6 | 21 | 1,171 | 8,121 |
| `recursive` | 90 | 242,06 | 12 / 463 | 7 | 5 | 1,190 | 17,881 |
| `semantic` | 191 | 95,83 | 4 / 300 | 0 | 27 | 1,000 | 34,007 |
| `parent_child` | 100 | 216,53 | 12 / 300 | 0 | 5 | 1,183 | 8,477 |
| `sliding_window` | 88 | 239,24 | 12 / 300 | 0 | 5 | 1,150 | 6,048 |

Hệ số lượng text = tổng số từ trong `chunk.text` / 18.303 từ đầu vào. Hệ số này chưa tính phần `parent_text` lưu thêm và không phải phép đo dung lượng index. Thời gian chỉ mô tả lần đo cục bộ; chênh lệch nhỏ giữa `fixed_word` và `sliding_window` là dao động đo, vì chúng dùng cùng logic.

Các nhận xét có thể rút ra từ số liệu:

- `semantic` tạo 191 chunk, khoảng 2,17 lần `fixed_word`, đồng thời có 27 chunk dưới 50 từ.
- `recursive` có 7/90 chunk vượt 300 từ, tương đương 7,78%; độ dài lớn nhất là 463.
- `paragraph` tạo 21 chunk dưới 50 từ và có chunk chỉ một từ.
- `min_chunk_size=50` không bảo đảm mọi chunk đạt 50 từ; trang ngắn và phần dư vẫn có thể nhỏ hơn.
- Chưa đo Recall/MRR cho từng chiến lược trong lần này. Bảng trên đánh giá cấu trúc và chi phí chia, chưa xếp hạng chất lượng retrieval.

### 5.4. Kết quả từ log RAG đã có

Tệp `evaluation/kinhtevn_rag_console_report_recursive.jsonl` lưu 29 câu hỏi, mỗi câu có 5 nguồn truy xuất. Tổng hợp lại với nhãn trang trong `kinhtevn_questions.jsonl` cho kết quả:

| Chỉ số | Kết quả | Diễn giải |
|---|---:|---|
| Page Hit@5 | 22/29 = 75,86% | Có ít nhất một trang nhãn trong 5 nguồn |
| Page Recall@5 trung bình | 72,41% | Trung bình tỷ lệ trang nhãn khác nhau được tìm thấy |
| MRR@5 theo trang | 0,5460 | Trung bình nghịch đảo thứ hạng nguồn đầu tiên thuộc trang nhãn |
| Tỷ lệ khớp từ khóa câu trả lời TB | 47,70% | Trung bình tỷ lệ từ khóa nhãn xuất hiện trong câu trả lời |

| Loại câu hỏi | Số câu | Câu tìm được trang nhãn trong top-5 |
|---|---:|---:|
| Sự kiện | 17 | 12 |
| Tóm tắt | 4 | 4 |
| Dự báo | 2 | 2 |
| So sánh | 2 | 0 |
| Suy luận | 1 | 1 |
| Chính sách | 2 | 2 |
| Rủi ro | 1 | 1 |

Đây là kết quả tổng hợp lại log cũ, không phải chạy mới toàn bộ RAG. Log cho biết nguồn dùng `recursive` nhưng không lưu đầy đủ model/revision và toàn bộ tham số chạy, nên chưa đủ để so sánh nhân quả giữa các cấu hình. Nhãn trang cũng cần kiểm chứng thủ công. Khớp từ khóa không tương đương độ chính xác nội dung; tìm đúng trang không bảo đảm tìm đúng đoạn hoặc trả lời đúng.

Ví dụ câu đầu tiên tìm được trang nhãn nhưng câu trả lời lấy nhầm dòng đầu trang làm tiêu đề. Hai câu so sánh chưa tìm được trang nhãn, là nhóm nên ưu tiên kiểm tra. Bộ kiểm tra bổ sung có hai câu ngoài tài liệu đều nhận câu từ chối theo mẫu; cỡ mẫu này chưa đủ đánh giá tổng quát. Hai log prompt injection cũng chưa có chấm chất lượng chuẩn để kết luận độ vững của hệ thống.

Nguồn nội bộ: [log recursive](../evaluation/kinhtevn_rag_console_report_recursive.jsonl), [nhãn câu hỏi](../evaluation/kinhtevn_questions.jsonl), [log kiểm tra bổ sung](../evaluation/kinhtevn_rag_answer_checks_recursive.jsonl).

### 5.5. Phạm vi đánh giá chunking

Việc so sánh các chiến lược hiện dựa trên số chunk, độ dài chunk, số chunk vượt hoặc thấp hơn ngưỡng, hệ số lượng text và thời gian chia. Công cụ đánh giá cũng hỗ trợ Hit, Precision, Recall, MRR và nDCG cho retrieval. Tuy nhiên, phép đo hiện chấp nhận kết quả liên quan nếu khớp trang HOẶC khớp một từ khóa; điều này có thể làm điểm cao dù đoạn không trả lời câu hỏi. Khi không có nhãn trang, Recall đang trở thành chỉ báo hit. IDCG của nDCG đang dựa trên số kết quả liên quan đã tìm thấy trong top-k, thay vì tập nhãn liên quan đầy đủ.

Bộ mặc định `chunking_questions.jsonl` có 33 câu về học máy; tài liệu hiện có trong `data` là `KinhteVN.pdf`. Khi đánh giá PDF kinh tế phải chỉ định bộ `kinhtevn_questions.jsonl`, tránh dùng câu hỏi và nhãn trang của tài liệu khác.

## 6. Embedding: so sánh nhiều model và đánh giá cho tiếng Việt

### 6.1. Kết quả đã thực hiện

Hệ thống sử dụng Sentence Transformers để embedding tài liệu và câu hỏi, hỗ trợ prefix, chuẩn hóa vector và lưu vector dạng số thực 32-bit. Hiện có bốn cấu hình model và công cụ benchmark riêng. Mặc định là multilingual MiniLM.

| Model | Chiều vector | Giới hạn được công bố trong cấu hình/model card | Vai trò và lưu ý |
|---|---:|---:|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 128 token trong cấu hình Sentence Transformers | Baseline gọn; đặc biệt chú ý truncation với chunk dài |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | 128 token trong cấu hình Sentence Transformers | Baseline multilingual; số chiều lớn hơn không tự bảo đảm retrieval tốt hơn |
| `multilingual-e5-small` | 384 | 512 token | Ứng viên retrieval; dùng prefix `query:` và `passage:` |
| `multilingual-e5-base` | 768 | 512 token | Ứng viên đối chứng với E5-small; cần đo chất lượng và tài nguyên |

Thông số tham khảo: [MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [MPNet](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2), [E5-small](https://huggingface.co/intfloat/multilingual-e5-small), [E5-base](https://huggingface.co/intfloat/multilingual-e5-base). Cần ghi lại giới hạn thực tế của model được tải trong mỗi lần benchmark; các giá trị trên không phải phép đo model runtime trên máy trong báo cáo này.

Một rủi ro cần ưu tiên là chunk 300 đơn vị khoảng trắng có thể vượt 128 hoặc 512 token, nhất là với tiếng Việt. Khi đó vector chỉ biểu diễn phần được model giữ lại dù metadata vẫn lưu toàn chunk. Việc sửa token counting và đo tỷ lệ truncation cần đi trước kết luận model nào tốt hơn.

Với các model E5, embedding tài liệu và câu hỏi sử dụng prefix tương ứng. Khi thay model, tài liệu phải được embedding lại và vector store phải được xây dựng lại, kể cả khi hai model có cùng số chiều.

### 6.2. Số liệu tham khảo cho tiếng Việt

Model card của AITeamVN công bố kết quả trên tập training Legal Zalo 2021 dùng làm tập đánh giá; tác giả cho biết model của họ không được huấn luyện trên tập này. Bảng sau trích kết quả do nhà phát triển công bố, chưa được tái lập trên máy và không phải kết quả trên tài liệu kinh tế của dự án:

| Model | Accuracy@1 | Accuracy@5 | MRR@10 |
|---|---:|---:|---:|
| `Vietnamese_Embedding` | 0,7274 | 0,9305 | 0,8181 |
| `Vietnamese_Embedding_v2` | 0,7262 | 0,9268 | 0,8149 |
| `Vietnamese-bi-encoder` của BKAI | 0,7109 | 0,9014 | 0,7951 |
| `BGE-M3` | 0,5682 | 0,8382 | 0,6822 |

Giữ nguyên tên chỉ số Accuracy của nguồn, không đổi thành Recall. Nguồn: [AITeamVN/Vietnamese_Embedding_v2, mục Evaluation](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2#evaluation).

Trên phép đo này, các model chuyên tiếng Việt là ứng viên đáng thử; phiên bản v2 không cao hơn phiên bản đầu ở mọi chỉ số. Không thể suy rộng thứ hạng của miền pháp lý sang miền kinh tế hoặc khẳng định một model tốt nhất cho toàn bộ tiếng Việt.

Hai ứng viên mở rộng chưa có trong registry chuẩn của dự án là `BAAI/bge-m3` và `AITeamVN/Vietnamese_Embedding_v2`. BGE-M3 có vector 1.024 chiều, hỗ trợ đầu vào tới 8.192 token và nhiều chế độ retrieval; pipeline hiện tại chỉ khai thác dense embedding. Nguồn: [BGE-M3](https://huggingface.co/BAAI/bge-m3). Model card AITeamVN v2 công bố vector 1.024 chiều và giới hạn 2.048 token. Nguồn: [thông số AITeamVN v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2#model-details).

### 6.3. Phạm vi đánh giá embedding

Các model được so sánh trên cùng corpus, cùng bộ câu hỏi, cùng chiến lược chunking và cùng loại index. Các chỉ số cần ghi nhận gồm Hit@5, Recall@5, MRR@5, nDCG@5, thời gian embedding, thời gian encode query, latency retrieval, dung lượng vector và tỷ lệ truncation. Hiện chưa có bảng kết quả đối chứng có thể tái lập cho cả bốn model trong các tệp kết quả đã kiểm tra; vì vậy báo cáo chưa kết luận model nào tốt hơn trên corpus của project.

## 7. Vector Store: tổ chức và đánh giá

### 7.1. Kết quả đã thực hiện

Hệ thống đã tích hợp FAISS để lưu và tìm kiếm vector. Vector, văn bản nguồn và cấu hình index được lưu riêng để có thể nạp lại và truy xuất top-k. Hệ thống có kiểm tra sự phù hợp giữa số chiều query và số lượng metadata với index.

Vector được chuẩn hóa trước khi lưu, vì vậy inner product tương đương cosine similarity khi cả tài liệu và query đều qua bước chuẩn hóa. Kết quả tìm kiếm giữ định danh chunk, nguồn, trang, score, parent text và tham chiếu ảnh.

Hiện có ba loại index FAISS. Đây là ba thuật toán index trong một thư viện; dự án chưa tích hợp ba hệ quản trị vector database khác nhau.

| Index | Cơ chế | Ưu điểm | Nhược điểm | Phạm vi sử dụng |
|---|---|---|---|---|
| Flat | So sánh với toàn bộ vector | Tìm kiếm chính xác, không cần train; làm mốc đối chứng | Chi phí tìm kiếm tăng theo số vector | Corpus hiện tại và baseline chất lượng |
| HNSW | Tìm kiếm gần đúng trên đồ thị | Có thể giảm latency ở quy mô lớn | Tốn bộ nhớ đồ thị; đánh đổi recall với tham số tìm kiếm | Khi Flat không đáp ứng latency và đủ RAM |
| IVF | Phân cụm vector, tìm trong một số cụm | Điều chỉnh phạm vi tìm qua số cụm được duyệt | Cần train; có thể bỏ sót cụm chứa kết quả đúng | Corpus đủ lớn và dữ liệu train đại diện |

Nguồn về cơ chế và trade-off: [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes). Với khoảng 90 chunk ở phép đo hiện tại, chưa có bằng chứng cần thay Flat bằng ANN.

### 7.2. So sánh bằng số: phần có thể xác định hiện tại

Chưa có bảng chạy đối chứng Flat/HNSW/IVF được lưu trong các tệp kết quả đã kiểm tra. Không công bố latency hoặc Recall giả định. Có thể tính phần dung lượng vector float32 thô theo `N * d * 4` byte; bảng dưới là tính toán lý thuyết, chưa bao gồm metadata, đồ thị HNSW, ID, centroid hoặc bộ nhớ tiến trình:

| Số vector | 384 chiều | 768 chiều | 1.024 chiều |
|---|---:|---:|---:|
| 90 | 135 KiB | 270 KiB | 360 KiB |
| 100.000 | 146,48 MiB | 292,97 MiB | 390,63 MiB |

Tăng 384 lên 768 chiều làm gấp đôi phần vector thô. Nhiều chunk hơn cũng tăng dung lượng và chi phí embedding.

### 7.3. Trạng thái hiện tại

- Chưa có API ở lớp ứng dụng cho cập nhật/xóa từng tài liệu, lọc metadata hoặc quản lý nhiều người dùng.
- Chưa có hybrid search BM25 + dense và reranker trong pipeline đang chạy.
- Chưa benchmark đồng thời nhiều query, p95/p99, throughput, peak RAM hoặc kích thước file index thực tế.
- HNSW mới đưa `M` vào cấu hình; chưa có cấu hình và khảo sát `efSearch`, `efConstruction` đầy đủ.
- File cấu hình index chưa lưu model/revision, prefix, tokenizer và fingerprint corpus/chunking đầy đủ. Kiểm tra số chiều không phát hiện được đổi sang model khác cùng số chiều.
- Runner hiện so sánh ba loại index FAISS; chưa so sánh các backend vector database.

## 8. Retrieval và hướng Re-ranker

### 8.1. Retrieval hiện tại

Retriever hiện encode câu hỏi bằng cùng hệ embedding với chunk, tìm `top-k` trên FAISS và trả về score cùng metadata nguồn. Vì vector đã normalize, index Flat đo inner product tương đương cosine similarity. Context được tạo từ các kết quả này, có thể ưu tiên parent text khi chunk thuộc chiến lược cha-con, rồi chuyển sang LLM.

Phương án hiện tại là dense retrieval. Ưu điểm là kiến trúc đơn giản, phù hợp corpus nhỏ và có thể đo exact với Flat. Nhược điểm là dễ bỏ sót từ khóa chính xác, số hiệu, tên riêng hoặc câu hỏi so sánh; score cũng chưa phải xác suất đúng. Kết quả log hiện tại cho thấy Page Hit@5 đạt `75,86%`, nhưng hai câu hỏi so sánh không tìm được trang nhãn trong top-5, nên dense retrieval cần được kiểm tra riêng trên nhóm câu hỏi này.

### 8.2. Trạng thái reranker

Pipeline hiện tại chưa tích hợp hybrid search hoặc reranker. Kết quả đưa vào context hiện được lấy trực tiếp từ top-k dense retrieval của FAISS.

## 9. Đánh giá hệ thống RAG

Đánh giá hiện được theo dõi ở ba tầng: (1) ingestion/chunking, (2) retrieval, và (3) generation. Các chỉ số retrieval đang dùng là Page Hit@k, Recall@k, MRR@k và nDCG@k. Log cũng lưu câu trả lời, nguồn truy xuất và kết quả kiểm tra bổ sung.

Kết quả đã có trên log recursive gồm 29 câu hỏi, 5 nguồn mỗi câu: Page Hit@5 `22/29 = 75,86%`, Page Recall@5 trung bình `72,41%`, MRR@5 `0,5460` và tỷ lệ khớp từ khóa câu trả lời trung bình `47,70%`. Đây là tín hiệu baseline, chưa phải độ chính xác end-to-end: khớp trang không bảo đảm đúng đoạn, còn khớp từ khóa không bảo đảm đúng nội dung.

Trạng thái phép đo hiện tại:

- Metric còn cho phép liên quan nếu khớp trang **hoặc** từ khóa, nên có thể đánh giá cao một kết quả không trả lời đúng câu hỏi.
- nDCG chưa dùng đầy đủ tập nhãn liên quan để tính IDCG.
- Chưa có evidence span, nhãn mức độ liên quan và chấm factual/citation độc lập.
- Dataset mặc định về machine learning không khớp corpus `KinhteVN.pdf`; phải chỉ định `kinhtevn_questions.jsonl`.
- Hai prompt-injection và các câu ngoài tài liệu mới có cỡ mẫu nhỏ, chưa đủ kết luận robustness.

Các số liệu trên là kết quả tổng hợp từ cấu hình và log hiện có; chưa có kết quả benchmark mới cho embedding, index hoặc toàn bộ LLM trong lần tổng hợp này.

## 10. Kết quả hiện tại

| Hạng mục | Trạng thái đã thực hiện và đang triển khai |
|---|---|---|---|
| Document | Đã có mô hình tài liệu/trang/ảnh, định danh theo hash và metadata. Loader hiện tập trung vào PDF. |
| Chuyển đổi PDF | Đã trích xuất text, layout, ảnh và thông tin bảng bằng PyMuPDF. |
| Hình ảnh | Đã có tham chiếu, xuất ảnh và nhánh OCR; caption tự động và truy xuất đa phương thức chưa tích hợp. |
| Chunking | Đã có 7 strategy và số đo cấu trúc/thời gian trên PDF. Benchmark retrieval theo từng strategy chưa có trong log hiện tại. |
| Embedding | Đã có service, 4 cấu hình model và runner benchmark. Kết quả đối chứng nội bộ chưa được lưu trong lần tổng hợp này. |
| Vector store | Đã tích hợp FAISS Flat/HNSW/IVF, lưu/load/search và metadata. Chưa có benchmark đối chứng hoàn chỉnh trong log hiện tại. |
| Retrieval và RAG | Đã có dense retrieval top-k, tạo context và gọi local Qwen3. Log hiện có 29 câu hỏi với Page Hit@5 là 75,86%. |
| Đánh giá | Đã có runner và log đánh giá retrieval, câu trả lời, nguồn truy xuất và kiểm tra bổ sung. |

Trong lần lập báo cáo này đã đo quá trình chuyển đổi tài liệu và chunking, đồng thời tổng hợp lại log. Các phần embedding, index và LLM được báo cáo theo trạng thái và kết quả đã lưu.

## Phụ lục: dữ liệu và khả năng tái lập

Tài liệu đo: `data/KinhteVN.pdf`. SHA-256: `0cd5ed284f50fa938ec67cb0f88e43fd39a9fe1f979fd6b9988436609d987875`.

Số đo bảng chunking được thực hiện 5 lần cho mỗi chiến lược với bộ tham số (300, 50, 50). Trung vị được tính quanh riêng bước chia. Min/max, số chunk vượt/ngắn và hệ số text được tính theo số đơn vị phân cách bằng khoảng trắng. Không thay đổi thuật toán trong lần đo.

Các lệnh benchmark hiện có, chạy từ thư mục `rag`:

```powershell
python -m src.chunking_benchmark --pdf data/KinhteVN.pdf --questions evaluation/kinhtevn_questions.jsonl --top-k 5 --chunk-size 300 --chunk-overlap 50 --min-chunk-size 50
python -m src.embedding_benchmark --pdf data/KinhteVN.pdf --questions evaluation/kinhtevn_questions.jsonl --models miniLM-multilingual mpnet-multilingual e5-small e5-base --top-k 5
python -m src.vector_store_benchmark --pdf data/KinhteVN.pdf --questions evaluation/kinhtevn_questions.jsonl --indexes flat hnsw ivf --top-k 5
```

Khi chạy benchmark cần ghi lại model, prefix, chunking và tham số index hiệu lực. Môi trường đo sử dụng Python 3.14.3.

Không sử dụng các con số minh họa trong `docs/chunking.md` làm kết quả thực nghiệm. Các nguồn ngoài được truy cập khi tổng hợp ngày 22/09/2026; số liệu công bố trên dataset khác được ghi riêng với số liệu của project.
