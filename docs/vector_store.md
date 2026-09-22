# Vector Store

Vector Store lưu embedding của các chunk và truy vấn lại những chunk gần nhất với câu hỏi. Trong project này Vector Store dùng FAISS, metadata của chunk lưu riêng trong `metadata.json`.

## Cách tổ chức hiện tại

Mỗi vector tương ứng đúng một chunk trong metadata:

- `index.faiss`: FAISS index chứa embedding đã normalize.
- `metadata.json`: thông tin chunk như `chunk_id`, `document_id`, `source`, `page`, `chunk_index`, `section_title`, `parent_text`.
- `index_config.json`: cấu hình index đã build như `index_type`, `hnsw_m`, `ivf_nlist`, `ivf_nprobe`.

Embedding được normalize trong `EmbeddingService`, nên FAISS dùng inner product để tính tương đương cosine similarity.

## Các thuật toán đã hỗ trợ

| Index | Cách hoạt động | Ưu điểm | Nhược điểm | Khi dùng |
| --- | --- | --- | --- | --- |
| `flat` | So sánh query với toàn bộ vector. | Chính xác nhất, đơn giản, ổn định, không cần train. | Chậm khi dữ liệu rất lớn vì quét toàn bộ vector. | Dataset nhỏ/vừa, benchmark baseline, cần độ chính xác cao. |
| `hnsw` | Dùng đồ thị gần đúng để đi tới các vector lân cận. | Truy vấn nhanh trên tập dữ liệu lớn hơn, không cần train. | Kết quả gần đúng, tốn RAM hơn, tham số `M` ảnh hưởng tốc độ/chất lượng. | Nhiều chunk, cần giảm latency nhưng vẫn muốn recall tốt. |
| `ivf` | Chia vector thành nhiều cụm/inverted lists, query chỉ tìm trong một số cụm. | Truy vấn nhanh khi số vector lớn, kiểm soát tốc độ bằng `nprobe`. | Cần train index, có thể mất recall nếu `nprobe` thấp hoặc dữ liệu nhỏ. | Dataset lớn, cần cân bằng tốc độ/độ chính xác. |

## Cấu hình

Trong `.env`:

```makefile
VECTOR_INDEX_TYPE=flat
VECTOR_BENCHMARK_INDEXES=flat,hnsw,ivf
FAISS_HNSW_M=32
FAISS_IVF_NLIST=64
FAISS_IVF_NPROBE=8
```

Ý nghĩa:

- `VECTOR_INDEX_TYPE`: index dùng cho app chính.
- `VECTOR_BENCHMARK_INDEXES`: danh sách index đem đi benchmark.
- `FAISS_HNSW_M`: số liên kết mỗi node trong HNSW. Cao hơn thường recall tốt hơn nhưng tốn RAM/build lâu hơn.
- `FAISS_IVF_NLIST`: số cụm IVF. Cao hơn chia nhỏ không gian hơn nhưng cần đủ dữ liệu để train ổn.
- `FAISS_IVF_NPROBE`: số cụm được duyệt khi search. Cao hơn recall tốt hơn nhưng chậm hơn.

Sau khi đổi `VECTOR_INDEX_TYPE` hoặc tham số FAISS, cần rebuild:

```powershell
python -m src.main --rebuild
```

## Benchmark

Chạy:

```powershell
python -m src.vector_store_benchmark
```

Hoặc chỉ test một vài index:

```powershell
python -m src.vector_store_benchmark --indexes flat hnsw ivf
```

Bảng kết quả gồm:

| Cột | Ý nghĩa |
| --- | --- |
| `index_type` | Loại FAISS index đang test. |
| `chunks` | Số chunk được đưa vào index. |
| `dimension` | Số chiều embedding. |
| `recall@5` | Tỷ lệ câu hỏi có ít nhất một chunk đúng trong top 5. |
| `mrr` | Chunk đúng xuất hiện càng cao thì điểm càng lớn. |
| `ndcg@5` | Đánh giá chất lượng thứ hạng trong top 5. |
| `build_s` | Thời gian build index. |
| `retrieval_ms` | Thời gian truy vấn trung bình mỗi câu hỏi. |

## Cách chọn

Với tài liệu PDF hiện tại, số chunk chỉ ở mức nhỏ/vừa nên `flat` là lựa chọn mặc định tốt nhất: dễ giải thích, recall ổn định, phù hợp làm baseline.

Khi số chunk tăng lên hàng chục nghìn trở lên:

- Dùng `hnsw` nếu muốn truy vấn nhanh và không muốn bước train index.
- Dùng `ivf` nếu dữ liệu rất lớn và cần chỉnh `nprobe` để đổi tốc độ lấy recall.
- Luôn so sánh bằng benchmark nội bộ trước khi đổi index cho app chính.

## Vector database khác

Project hiện dùng FAISS local để đơn giản và dễ benchmark. Nếu triển khai production có nhiều tài liệu/người dùng, có thể cân nhắc:

| Vector Store | Điểm mạnh | Khi cân nhắc |
| --- | --- | --- |
| FAISS | Nhanh, local, dễ benchmark thuật toán. | Prototype, single-machine, bài đánh giá học thuật. |
| Chroma | Dễ dùng, lưu metadata tiện, phù hợp local/dev. | Demo hoặc app nhỏ cần quản lý collection. |
| Qdrant | Có service/API, filter metadata tốt, production-friendly. | App nhiều tài liệu, cần filter theo metadata hoặc deploy server. |
| Milvus | Quy mô lớn, phân tán, nhiều index backend. | Dữ liệu rất lớn, cần vận hành cluster. |

