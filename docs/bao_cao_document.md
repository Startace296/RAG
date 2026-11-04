# BÁO CÁO: TỔ CHỨC VÀ XỬ LÝ TÀI LIỆU TRONG HỆ THỐNG RAG

## Mở đầu

Trong một hệ thống RAG (Retrieval-Augmented Generation), chất lượng câu trả lời phụ thuộc rất lớn vào khâu chuẩn bị dữ liệu. Mô hình ngôn ngữ chỉ trả lời tốt khi được cung cấp đúng đoạn văn bản liên quan. Muốn tìm được đúng đoạn đó, tài liệu gốc phải được đọc chính xác, tổ chức có cấu trúc và chia thành những phần nhỏ hợp lý.

Báo cáo này trình bày cách hệ thống xử lý tài liệu, từ lúc nhận một file PDF đến khi tài liệu sẵn sàng cho việc tìm kiếm. Báo cáo tập trung vào bốn câu hỏi:

1. Document được tổ chức như thế nào?
2. Document Chunk được tổ chức như thế nào?
3. Tài liệu đưa vào được chuyển đổi ra sao?
4. Tài liệu có hình ảnh được xử lý thế nào?

---

## 1. Tổng quan quy trình

Toàn bộ quá trình xử lý tài liệu gồm các giai đoạn nối tiếp nhau:

```text
  File PDF
     │
     ▼
  [1] Đọc và trích xuất ──────► Document (tài liệu → trang → hình ảnh)
     │
     ▼
  [2] Làm sạch và nhận diện cấu trúc
     │
     ▼
  [3] Chia nhỏ (Chunking) ────► Danh sách Chunk
     │
     ▼
  [4] Mã hóa thành vector (Embedding)
     │
     ▼
  [5] Lưu vào Vector Store ───► sẵn sàng cho truy vấn
```

Mỗi giai đoạn do một module riêng đảm nhận:

| Giai đoạn | Module | Vai trò |
| --- | --- | --- |
| Đọc và trích xuất | `src/documents/pdf_loader.py` | Đọc PDF, lấy văn bản, hình ảnh và thông tin cấu trúc |
| Mô hình dữ liệu | `src/documents/schemas.py` | Định nghĩa cách biểu diễn tài liệu |
| Chia nhỏ | `src/chunking/strategies.py` | Chia tài liệu thành các chunk |
| Embedding | `src/embeddings/service.py` | Chuyển chunk thành vector |
| Lưu trữ và tìm kiếm | `src/retrieval/` | Lưu vector và tìm chunk liên quan |

Các module giao tiếp với nhau qua những cấu trúc dữ liệu thống nhất. Vì vậy có thể thay đổi một giai đoạn, ví dụ đổi cách chia chunk, mà không ảnh hưởng đến các giai đoạn khác.

---

## 2. Tổ chức Document

### 2.1. Ý tưởng

Một tài liệu thực tế có cấu trúc phân cấp tự nhiên: một cuốn tài liệu gồm nhiều trang, mỗi trang chứa văn bản và có thể có hình ảnh. Hệ thống giữ nguyên cấu trúc này khi biểu diễn tài liệu trong chương trình, thay vì gộp tất cả thành một chuỗi văn bản dài. Nhờ đó, về sau mọi đoạn văn luôn biết mình thuộc tài liệu nào, trang nào và đi kèm hình ảnh nào.

### 2.2. Cấu trúc ba tầng

```text
Document (Tài liệu)
 │
 ├── Thông tin chung: mã tài liệu, tên file, tác giả, ngày tạo, số trang...
 │
 └── Danh sách DocumentPage (Trang)
        │
        ├── Số trang
        ├── Nội dung văn bản đã làm sạch
        ├── Thông tin trang: tiêu đề mục, số từ, bố cục, bảng...
        │
        └── Danh sách DocumentImage (Hình ảnh)
               ├── Vị trí: thuộc trang nào, là ảnh thứ mấy
               ├── Đường dẫn file ảnh đã lưu
               ├── Chữ đọc được từ ảnh (OCR)
               └── Mô tả ảnh (caption)
```

**Tầng Document** đại diện cho toàn bộ một file tài liệu. Tầng này lưu thông tin nhận dạng và thông tin tổng quát: mã tài liệu, tên và đường dẫn file, loại tài liệu, tiêu đề, tác giả, ngày tạo, cùng các thống kê chung như số trang và số hình ảnh.

**Tầng DocumentPage** đại diện cho một trang và là đơn vị xử lý trung tâm của hệ thống. Mỗi trang chứa văn bản đã làm sạch, danh sách hình ảnh trên trang và các thông tin mô tả trang. Quan trọng nhất trong số đó là **tiêu đề mục** mà trang thuộc về, cho biết nội dung trang nằm trong phần nào của tài liệu.

**Tầng DocumentImage** đại diện cho một hình ảnh trong trang. Tầng này không chứa dữ liệu ảnh trực tiếp mà chỉ lưu đường dẫn tới file ảnh đã được tách ra, kèm phần văn bản có thể khai thác từ ảnh (chữ đọc được, mô tả).

### 2.3. Các nguyên tắc tổ chức

**Định danh theo nội dung.** Mã tài liệu được tạo bằng cách băm (SHA-256) toàn bộ nội dung file. Đổi tên hay di chuyển file thì mã vẫn giữ nguyên; chỉnh sửa nội dung thì mã thay đổi. Cách này tránh hai lỗi thường gặp: cùng một tài liệu bị coi là hai tài liệu khác nhau, và tài liệu đã bị sửa nhưng hệ thống vẫn dùng dữ liệu cũ.

**Mỗi phần tử tự biết nguồn gốc của mình.** Mỗi trang và mỗi hình ảnh đều lưu lại mã tài liệu cha; hình ảnh còn lưu số trang. Khi các phần này được tách ra xử lý riêng ở những bước sau, chúng vẫn truy ngược được về vị trí gốc.

**Dữ liệu không bị thay đổi sau khi tạo.** Các đối tượng Document, Page và Image được khai báo bất biến (immutable). Các bước xử lý phía sau chỉ đọc dữ liệu gốc và tạo dữ liệu mới, không sửa dữ liệu gốc, nên tránh được lỗi do nhiều bước cùng thay đổi một dữ liệu.

**Tách biệt giữa đọc tài liệu và chia nhỏ.** Trước khi chia chunk, mỗi trang được chuyển thành một bản ghi đơn giản gồm văn bản, số trang, tên nguồn, thông tin trang và danh sách ảnh. Bộ chia chunk chỉ làm việc với bản ghi này, không cần biết tài liệu gốc là PDF hay định dạng khác. Muốn hỗ trợ thêm Word hay HTML, chỉ cần viết thêm bộ đọc tương ứng.

---

## 3. Quá trình chuyển đổi tài liệu

### 3.1. Khó khăn khi đọc PDF

PDF là định dạng dùng để **trình bày**, không phải để lưu **cấu trúc nội dung**. Bên trong, một trang PDF chỉ là tập hợp lệnh “vẽ chữ này ở vị trí này, với cỡ chữ này”. PDF không có khái niệm đoạn văn, tiêu đề, header hay bảng. Vì vậy, nếu chỉ lấy chữ ra một cách đơn thuần, sẽ gặp các vấn đề:

- Câu bị ngắt dòng giữa chừng ở lề trang.
- Không biết đâu là ranh giới giữa các đoạn văn.
- Header, footer và số trang lặp lại trên mọi trang, trộn lẫn vào nội dung.
- Không phân biệt được tiêu đề với nội dung thường.

Nhiệm vụ của bước chuyển đổi là **dựng lại cấu trúc ngữ nghĩa** từ những thông tin trình bày đó. Hệ thống dùng thư viện PyMuPDF để đọc PDF. Thư viện này trả về nội dung dưới dạng các khối (block), trong khối có các dòng (line), trong dòng có các đoạn chữ cùng định dạng (span) kèm vị trí và cỡ chữ.

### 3.2. Các bước chuyển đổi

Quá trình đọc được chia làm **hai lượt**. Lượt thứ nhất quét toàn bộ tài liệu để thu thập các thông tin thống kê chung. Lượt thứ hai xử lý chi tiết từng trang, dựa trên các thông tin đó. Cần hai lượt vì một số quyết định chỉ đưa ra được khi đã nhìn toàn bộ tài liệu, ví dụ một dòng có phải header hay không chỉ biết được khi thấy nó lặp lại trên nhiều trang.

**Bước 1: Tạo mã tài liệu.** Hệ thống băm nội dung file để tạo mã định danh, đồng thời xóa các hình ảnh cũ đã tách ra từ lần xử lý trước của cùng tài liệu.

**Bước 2: Trích xuất văn bản theo khối.** Văn bản được lấy theo từng khối mà PyMuPDF nhận diện. Mỗi khối thường tương ứng với một đoạn văn, một tiêu đề hoặc một chú thích. Các khối được ngăn cách bằng một dòng trống. Đây là quy ước chung giữa bộ đọc và bộ chia chunk: **dòng trống nghĩa là ranh giới đoạn văn**.

**Bước 3: Nối lại các dòng bị ngắt.** Trong một đoạn, PDF thường ngắt dòng ở lề phải dù câu chưa kết thúc. Hệ thống phân biệt hai trường hợp: dòng dài bị ngắt giữa chừng thì được nối với dòng sau bằng dấu cách; dòng ngắn (thường là tiêu đề hoặc nhãn) thì giữ nguyên việc xuống dòng vì nó mang ý nghĩa cấu trúc.

**Bước 4: Loại bỏ nội dung lặp (header, footer, số trang).** Hệ thống đếm xem mỗi dòng xuất hiện trên bao nhiêu trang. Dòng lặp lại trên một tỷ lệ đáng kể số trang được coi là header hoặc footer và bị loại bỏ. Khi so sánh, số trang dính ở đầu hoặc cuối dòng được bỏ qua để nhận ra cùng một header dù số trang thay đổi. Ngoài ra, số trang đứng riêng và các dòng đóng dấu lặp nhiều lần ở mép trang cũng bị loại.

Hệ thống chỉ xóa dòng lặp ở **mép trang**. Dòng lặp nằm giữa trang, ví dụ chú thích “Nguồn: …” dưới nhiều biểu đồ, là nội dung thật nên được giữ lại. Bước này quan trọng vì nếu giữ header, đoạn văn nào cũng chứa tên tài liệu và làm sai lệch việc tìm kiếm.

**Bước 5: Nhận diện tiêu đề mục.** Tài liệu được dàn trang thường dùng chữ to hơn rõ rệt cho tiêu đề. Hệ thống xác định cỡ chữ của phần thân bài (cỡ chữ chiếm nhiều ký tự nhất trong tài liệu), rồi coi những dòng có cỡ chữ lớn hơn đáng kể là tiêu đề. Kèm theo đó là các quy tắc bổ sung:

- Loại trừ chú thích hình, bảng, dòng “Nguồn:” và nhãn trục biểu đồ (dòng toàn số).
- Ghép lại tiêu đề dài bị tách thành hai dòng.
- Nếu một trang có nhiều tiêu đề, lấy tiêu đề cụ thể nhất (nằm dưới cùng).
- Trang không có tiêu đề mới **kế thừa tiêu đề của trang trước**, vì một mục thường kéo dài qua nhiều trang.

**Bước 6: Ghi nhận bố cục và bảng.** Hệ thống ghi lại vị trí các khối văn bản, khối hình ảnh và các bảng được phát hiện trên trang. Hiện tại đây là thông tin mô tả phục vụ kiểm tra; nội dung bảng vẫn được lấy như văn bản thường.

**Bước 7: Xử lý hình ảnh.** Hình ảnh trên trang được phát hiện, lọc, lưu ra file và có thể được đọc chữ (chi tiết ở mục 5).

**Bước 8: Tạo đối tượng Document.** Các trang có nội dung được đóng gói thành DocumentPage, gắn với danh sách hình ảnh và thông tin trang, rồi tập hợp thành một Document hoàn chỉnh. Trang trống hoàn toàn bị bỏ qua.

### 3.3. Đảm bảo tính nhất quán

Mỗi khi xây dựng chỉ mục tìm kiếm, hệ thống ghi lại cấu hình đã dùng: mã băm file nguồn, mô hình embedding, cách chia chunk và phiên bản bộ đọc tài liệu. Lần chạy sau, nếu một trong các yếu tố này thay đổi, hệ thống tự động xây dựng lại chỉ mục. Nhờ vậy, dữ liệu dùng để tìm kiếm luôn khớp với tài liệu và cấu hình hiện tại.

---

## 4. Tổ chức Document Chunk

### 4.1. Vai trò của Chunk

Chunk là **đơn vị tìm kiếm** của hệ thống. Mỗi chunk được chuyển thành một vector. Khi người dùng đặt câu hỏi, hệ thống tìm các chunk có vector gần với câu hỏi nhất rồi đưa chúng cho mô hình ngôn ngữ làm ngữ cảnh.

Kích thước chunk là một bài toán cân bằng:

- **Chunk quá nhỏ**: tìm kiếm chính xác nhưng thiếu ngữ cảnh để trả lời.
- **Chunk quá lớn**: chứa nhiều ý khác nhau, vector bị “pha loãng”, khó tìm đúng, và tốn không gian ngữ cảnh của mô hình.

### 4.2. Nguyên tắc chia Chunk

**Chunk không vượt qua ranh giới trang.** Mỗi trang được chia riêng, nên một chunk luôn nằm gọn trong một trang. Nhờ vậy khi trích dẫn nguồn, hệ thống luôn chỉ ra đúng số trang.

**Ưu tiên ranh giới tự nhiên của ngôn ngữ.** Chiến lược mặc định chỉ cắt ở ranh giới đoạn văn hoặc câu, tránh cắt ngang một ý.

**Có phần gối đầu (overlap).** Chunk sau lặp lại một phần cuối của chunk trước, để thông tin nằm ở ranh giới giữa hai chunk không bị mất.

### 4.3. Chiến lược chia mặc định: Recursive

Chiến lược recursive hoạt động theo nguyên tắc “chia theo đơn vị lớn nhất có thể, chỉ chia nhỏ hơn khi cần”:

```text
  Văn bản của trang
        │
        ▼
  Tách thành các đoạn văn (theo dòng trống)
        │
        ├── Đoạn vừa kích thước ──────────► giữ nguyên đoạn
        │
        └── Đoạn quá dài ──► tách thành các câu
                                  │
                                  ├── Câu vừa kích thước ──► giữ nguyên câu
                                  └── Câu quá dài ─────────► cắt theo số từ
        │
        ▼
  Gom các đơn vị liên tiếp thành chunk, cho đến khi gần đạt kích thước tối đa
        │
        ▼
  Khi đóng một chunk, mang phần cuối của nó sang đầu chunk tiếp theo (overlap)
```

Khi gom, hệ thống chỉ đóng chunk nếu chunk hiện tại đã đủ một kích thước tối thiểu, để tránh tạo ra những chunk quá vụn. Các đơn vị trong một chunk được nối bằng dấu xuống dòng, giúp tiêu đề và đoạn văn vẫn tách bạch khi mô hình ngôn ngữ đọc.

Hệ quả tích cực của cách làm này là **tiêu đề phụ thường đi cùng đoạn văn ngay sau nó**, vì tiêu đề ngắn được mang sang chunk mới qua phần overlap. Nội dung trong chunk vì thế có ngữ cảnh rõ ràng hơn.

### 4.4. Các chiến lược khác

Ngoài recursive, hệ thống hỗ trợ thêm các chiến lược khác để so sánh và dùng cho những loại tài liệu khác nhau:

| Chiến lược | Cách hoạt động | Phù hợp khi |
| --- | --- | --- |
| Fixed word / Sliding window | Cắt theo số từ cố định, có gối đầu | Làm mốc so sánh đơn giản |
| Sentence | Tách câu rồi gom câu thành chunk | Văn xuôi có dấu câu chuẩn |
| Paragraph | Tách đoạn rồi gom đoạn thành chunk | Tài liệu có đoạn văn rõ ràng |
| Semantic | Gom câu, cắt khi câu mới ít liên quan đến phần trước | Tài liệu đổi chủ đề thường xuyên |
| Parent–Child | Chunk nhỏ (con) dùng để tìm kiếm, chunk lớn (cha) đưa cho mô hình | Cần tìm chính xác nhưng trả lời cần nhiều ngữ cảnh |

Chiến lược **Parent–Child** tách hai vai trò mà một chunk thông thường phải gánh cùng lúc. Chunk con nhỏ giúp tìm kiếm chính xác. Khi đã tìm được chunk con, hệ thống đưa chunk cha lớn hơn chứa nó cho mô hình, để mô hình có đủ ngữ cảnh trả lời.

### 4.5. Cấu trúc một Chunk

Mỗi chunk không chỉ gồm văn bản mà còn mang theo thông tin mô tả, chia thành các nhóm:

| Nhóm thông tin | Nội dung | Công dụng |
| --- | --- | --- |
| Nội dung | Văn bản của chunk | Dùng để tạo vector và đưa cho mô hình |
| Định danh | Mã chunk | Nhận diện và loại bỏ chunk trùng lặp khi tìm kiếm |
| Vị trí | Mã tài liệu, tên file, số trang, thứ tự chunk trong trang và trong tài liệu, vị trí từ bắt đầu và kết thúc | Trích dẫn nguồn chính xác |
| Ngữ cảnh | Tiêu đề mục | Cho biết chunk thuộc phần nào của tài liệu |
| Quan hệ cha–con | Mã và văn bản của chunk cha | Mở rộng ngữ cảnh khi dùng Parent–Child |
| Hình ảnh | Danh sách tham chiếu tới ảnh cùng trang | Liên kết văn bản với hình minh họa |
| Kích thước và cách chia | Số từ, số ký tự, tên chiến lược đã dùng | Kiểm tra và so sánh các chiến lược |

Mã chunk được tạo bằng cách băm tổ hợp gồm mã tài liệu, số trang, thứ tự và phần đầu nội dung. Mã này có tính **tất định**: xử lý lại cùng một tài liệu với cùng cấu hình sẽ cho ra đúng những mã cũ.

### 4.6. Chunk được sử dụng như thế nào

Sau khi được tạo, chunk đi qua các bước sau:

1. **Mã hóa:** văn bản chunk được thêm tiền tố `passage:` (theo quy ước của mô hình e5) rồi chuyển thành vector và chuẩn hóa độ dài. Nhờ đó, độ tương đồng giữa câu hỏi và chunk được đo bằng góc giữa hai vector (cosine).
2. **Lưu trữ:** vector được lưu trong chỉ mục FAISS, còn thông tin mô tả của chunk được lưu trong một file JSON đi kèm. Hai phần này liên kết với nhau theo thứ tự.
3. **Tìm kiếm:** hệ thống kết hợp hai cách tìm. Tìm theo ngữ nghĩa (vector) giúp hiểu ý câu hỏi; tìm theo từ khóa (BM25) giúp bắt đúng các con số, tên riêng. Kết quả của hai cách được hợp nhất thành một bảng xếp hạng chung.
4. **Tạo câu trả lời:** các chunk tốt nhất được loại trùng lặp và ghép lại thành ngữ cảnh, mỗi chunk kèm nhãn nguồn (tên file, số trang). Mô hình ngôn ngữ đọc ngữ cảnh và trả lời, còn thông tin vị trí, tiêu đề và hình ảnh của chunk được trả về làm nguồn tham khảo.

---

## 5. Xử lý tài liệu có hình ảnh

### 5.1. Đặc điểm của hình ảnh trong PDF

Hình ảnh trong PDF có vài đặc điểm cần chú ý:

- Mỗi ảnh được lưu một lần trong file và có thể được nhiều trang dùng lại.
- Nhiều “ảnh” thực chất là thành phần kỹ thuật như lớp mặt nạ trong suốt, điểm ảnh đệm rất nhỏ hay họa tiết nền, không mang nội dung.
- Nội dung của ảnh, như số liệu trên biểu đồ hay chữ trong ảnh chụp, **không tồn tại dưới dạng văn bản**, nên mô hình embedding văn bản không “nhìn” thấy được.

### 5.2. Quy trình xử lý

```text
  Trang PDF
     │
     ▼
  Phát hiện các hình ảnh trên trang
     │
     ▼
  Lọc bỏ ảnh không có nội dung (quá nhỏ, trùng lặp trên cùng trang)
     │
     ▼
  Lưu ảnh gốc ra thư mục riêng của tài liệu
     │
     ▼
  (Tùy chọn) Đọc chữ trong ảnh bằng OCR
     │
     ▼
  Tạo DocumentImage và gắn vào trang
```

**Phát hiện và lọc.** Hệ thống liệt kê mọi hình ảnh trên trang, bỏ những ảnh quá nhỏ và những ảnh xuất hiện lặp lại trên cùng trang. Bước lọc này rất cần thiết vì phần lớn “ảnh” kỹ thuật trong PDF không có giá trị nội dung.

**Lưu ảnh.** Ảnh hợp lệ được lưu nguyên bản (giữ định dạng và chất lượng gốc) vào thư mục riêng mang mã của tài liệu. Tên file ghi rõ số trang, thứ tự ảnh trong trang và mã tham chiếu nội bộ trong PDF, nên từ tên file có thể biết ngay ảnh nằm ở đâu trong tài liệu.

**Đọc chữ trong ảnh (OCR).** Khi được bật, hệ thống dùng Tesseract để đọc chữ trong ảnh. OCR là tính năng tùy chọn. Nếu thiếu thư viện hoặc OCR gặp lỗi, hệ thống không dừng lại mà ghi nhận trạng thái (đang tắt, thiếu thư viện, lỗi, không đọc được chữ, đọc thành công) và tiếp tục xử lý.

### 5.3. Hình ảnh tham gia vào hệ thống bằng cách nào

Hình ảnh được đưa vào hệ thống theo **hai con đường**.

**Con đường thứ nhất: chuyển ảnh thành văn bản để tìm kiếm được.** Nếu ảnh có chữ đọc được (OCR) hoặc có mô tả (caption), phần văn bản này được đánh dấu rõ là nội dung từ ảnh, rồi nối vào cuối văn bản của trang **trước khi chia chunk**. Như vậy nội dung ảnh đi chung quy trình với văn bản thường: được chia chunk, mã hóa và tìm kiếm. Kể cả trang chỉ có ảnh (ví dụ trang scan) cũng tìm kiếm được nếu OCR đọc ra chữ.

**Con đường thứ hai: giữ liên kết để truy vết.** Mỗi chunk mang theo danh sách tham chiếu tới các ảnh trên cùng trang, gồm đường dẫn file, số trang và thứ tự ảnh. Khi hệ thống trả lời, các tham chiếu này được trả về cùng nguồn, cho phép hiển thị hoặc mở hình gốc bên cạnh đoạn văn được trích dẫn.

Thiết kế này tách **dữ liệu ảnh** (file trên đĩa) khỏi **thông tin về ảnh** (đường dẫn, chữ, mô tả). Thông tin mô tả vì thế gọn nhẹ và dễ lưu trữ, trong khi ảnh gốc vẫn còn nguyên để xử lý thêm khi cần.

---

## 6. Nhận xét và hạn chế

Hệ thống hiện tại đã có một quy trình xử lý tài liệu đầy đủ và có cấu trúc. Tuy vậy, vẫn còn một số điểm cần cải thiện:

- **Hình ảnh chưa được “hiểu”.** OCR mặc định đang tắt và chưa có mô hình thị giác để mô tả ảnh. Vì vậy nội dung biểu đồ và hình minh họa chưa tham gia vào tìm kiếm; hệ thống mới dừng ở việc lưu trữ và liên kết ảnh.
- **Biểu đồ dạng vector chưa được xử lý.** Một số biểu đồ trong PDF được vẽ bằng lệnh đồ họa chứ không phải ảnh, nên không được phát hiện như hình ảnh.
- **Liên kết ảnh ở mức trang.** Mọi chunk trên một trang đều gắn với tất cả ảnh của trang đó, chưa xác định được ảnh nào thuộc đoạn văn nào.
- **Bảng chưa giữ được cấu trúc.** Bảng được phát hiện nhưng nội dung vẫn được đọc như văn bản thường, mất quan hệ hàng–cột.
- **Phần gối đầu có thể quá lớn.** Phần overlap luôn giữ trọn ít nhất một đoạn. Khi đoạn cuối dài, nội dung bị lặp lại nhiều giữa hai chunk và chunk có thể vượt kích thước dự kiến.
- **Chunk không vượt trang.** Đoạn văn kéo dài qua hai trang sẽ bị tách đôi.

Hướng phát triển tiếp theo: bật OCR và bổ sung mô hình thị giác để sinh mô tả cho biểu đồ, chuyển bảng thành dạng có cấu trúc, gắn ảnh với đoạn văn dựa trên vị trí trên trang, và giới hạn chặt hơn kích thước phần overlap.

---

## Kết luận

Hệ thống tổ chức tài liệu theo mô hình phân cấp **Tài liệu → Trang → Hình ảnh**. Mỗi phần tử có định danh rõ ràng và luôn truy ngược được về nguồn gốc. Vì PDF chỉ lưu thông tin trình bày, quá trình chuyển đổi phải dựng lại cấu trúc ngữ nghĩa: nhận diện đoạn văn, nối câu bị ngắt, loại bỏ header và footer, xác định tiêu đề mục theo cỡ chữ.

Tài liệu sau đó được chia thành các chunk theo chiến lược recursive, tôn trọng ranh giới đoạn và câu, không vượt qua ranh giới trang. Mỗi chunk mang đầy đủ thông tin về vị trí, ngữ cảnh và hình ảnh liên quan, phục vụ việc tìm kiếm chính xác và trích dẫn nguồn đáng tin cậy.

Hình ảnh được phát hiện, lọc, lưu trữ và liên kết với văn bản. Kiến trúc đã sẵn sàng để ảnh được chuyển thành văn bản qua OCR hoặc mô tả tự động và đi chung quy trình với văn bản thường. Đây là hướng phát triển quan trọng nhất để hệ thống trả lời được các câu hỏi liên quan đến nội dung hình ảnh.
