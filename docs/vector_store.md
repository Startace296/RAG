# Vector Store

Vector Store lưu embedding của các chunk và truy vấn lại những chunk gần nhất với câu hỏi. Trong project này Vector Store dùng FAISS, metadata của chunk lưu riêng trong `metadata.json`.

## Cách tổ chức hiện tại

Mỗi vector tương ứng đúng một chunk trong metadata:

- `index.faiss`: FAISS index chứa embedding đã normalize.
- `metadata.json`: thông tin chunk như `chunk_id`, `document_id`, `source`, `page`, `chunk_index`, `section_title`, `parent_text`.
- `index_config.json`: cấu hình index đã build như `index_type`, `hnsw_m`, `ivf_nlist`, `ivf_nprobe`, cùng khối `pipeline` ghi PDF nguồn (tên và SHA-256), model embedding, prefix và tham số chunking.

Khi chạy `python -m src.app.main`, nếu bất kỳ giá trị nào trong `index_config.json` khác với `.env` hiện tại thì index được build lại tự động. Index cũ không có khối `pipeline` cũng được build lại.

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
python -m src.app.main --rebuild
```

## Benchmark

Chạy:

```powershell
python -m src.benchmarks.vector_store_benchmark
```

Hoặc chỉ test một vài index:

```powershell
python -m src.benchmarks.vector_store_benchmark --indexes flat hnsw ivf
```

Bảng kết quả gồm:

| Cột | Ý nghĩa |
| --- | --- |
| `index_type` | Loại FAISS index đang test. |
| `chunks` | Số chunk được đưa vào index. |
| `dimension` | Số chiều embedding. |
| `queries` | Số câu hỏi đánh giá. |
| `hit@5` | Tỷ lệ câu hỏi có ít nhất một chunk đúng trong top 5. |
| `precision@5` | Tỷ lệ chunk liên quan trong 5 chunk trả về. |
| `recall@5` | Tỷ lệ trang nhãn được phủ bởi ít nhất một chunk liên quan trong top 5. |
| `mrr` | Chunk đúng xuất hiện càng cao thì điểm càng lớn (rank 1 = 1.0, rank 2 = 0.5, ...). |
| `ndcg@5` | Đánh giá chất lượng thứ hạng trong top 5. |
| `build_s` | Thời gian tạo và build FAISS index (giây). **Không gồm thời gian embed**: embedding được tính một lần trước khi đo, nên không so sánh được với `build_s` trong [chunking.md](chunking.md) hay [embedding.md](embedding.md). |
| `retrieval_ms` | Thời gian search FAISS trung bình mỗi câu hỏi (mili giây), không gồm thời gian encode câu hỏi. |

### Kết quả thực nghiệm

Cấu hình: `KinhteVN.pdf`, 29 câu hỏi trong `evaluation/kinhtevn_questions.jsonl`, chunking `recursive` (300/50), embedding `intfloat/multilingual-e5-small` trên CPU, dense retrieval, `top_k=5`, `FAISS_HNSW_M=32`, `FAISS_IVF_NLIST=64`, `FAISS_IVF_NPROBE=8`. Output gốc ở `evaluation/results/vector_store_benchmark.txt`.

```text
index_type  chunks  dimension  queries   hit@5  precision@5  recall@5     mrr  ndcg@5  build_s  retrieval_ms
----------  ------  ---------  -------  ------  -----------  --------  ------  ------  -------  ------------
flat           107        384       29  0.9655       0.3034    0.9655  0.8575  0.8660   0.0007        0.0422
hnsw           107        384       29  0.9655       0.3034    0.9655  0.8575  0.8660   0.0034        0.0510
ivf            107        384       29  1.0000       0.3103    0.9828  0.8661  0.8769   0.0048        0.0606
```

Nhận xét:

- **Dòng `flat` trùng với dòng `recursive` trong [chunking.md](chunking.md)** vì cùng cấu hình. Chỉ `build_s` khác (0.0007s so với ~9.8s), do chunking benchmark tính cả thời gian embed.
- **`hnsw` cho kết quả giống hệt `flat`**: với 107 vector, đồ thị HNSW tìm đúng các láng giềng gần nhất như tìm vét cạn.
- **`ivf` nhỉnh hơn `flat` là ngẫu nhiên, không phải IVF tốt hơn.** `flat` là tìm kiếm chính xác theo cosine; IVF chỉ tình cờ đưa vào top 5 một chunk đúng theo nhãn mà `flat` xếp dưới hạng 5. Chênh lệch tương ứng 1/29 câu. Ở lần chạy trước với `parent_child` (100 chunk), IVF lại kém `flat` (hit@5 0.8621 so với 0.8966).
- **IVF đang bị cấu hình quá tay cho dữ liệu nhỏ**: FAISS cảnh báo `clustering 107 points to 64 centroids: please provide at least 2496 training points`. Trung bình chưa tới 2 vector mỗi cụm, nên các centroid gần như vô nghĩa.
- **`flat` nhanh nhất cả lúc build lẫn truy vấn.** Mọi con số đều dưới 0.1 ms, chênh lệch giữa các index nằm trong nhiễu đo và thay đổi giữa các lần chạy. HNSW/IVF chỉ có lợi về tốc độ khi số vector lên hàng chục nghìn trở lên.

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


## Hybrid retrieval (dense + BM25)

Embedding làm "loãng" một câu ngắn chứa đáp án nằm trong chunk dài nhiều chủ đề. Vì vậy app mặc định dùng `RETRIEVAL_MODE=hybrid`:

1. FAISS lấy `HYBRID_CANDIDATES` chunk gần nhất và bỏ các chunk có điểm dense dưới `SIMILARITY_THRESHOLD`.
2. `BM25Index` (`src/retrieval/bm25.py`) chấm điểm từ khóa cho toàn bộ chunk, build trong RAM lúc truy vấn đầu tiên.
3. Các candidate được xếp lại bằng Reciprocal Rank Fusion: `1/(k + hạng dense) + 1/(k + hạng BM25)` với `k = HYBRID_RRF_K`, rồi lấy `TOP_K` chunk.

Ngưỡng vẫn áp lên điểm dense, nên câu hỏi ngoài tài liệu vẫn không có context. Đặt `RETRIEVAL_MODE=dense` để quay về FAISS thuần.

```makefile
RETRIEVAL_MODE=hybrid
HYBRID_CANDIDATES=10
HYBRID_RRF_K=60
```

Kết quả retrieval trên 29 câu `kinhtevn_questions.jsonl` (e5-small, recursive, TOP_K=5):

| Cấu hình | page_hit | Keyword đáp án có trong context |
| --- | --- | --- |
| Dense, text gốc | 0.931 | 0.911 |
| Dense, bỏ header | 0.931 | 0.905 |
| Hybrid, text gốc | 1.000 | 0.980 |
| Hybrid, bỏ header (mặc định) | 1.000 | 0.991 |

`HYBRID_CANDIDATES` 10 hay 20 cho cùng kết quả trên chỉ mục hiện tại (keyword 0.991, MRR 0.943), nên mặc định là 10 để reranker chỉ chấm 10 chunk; `HYBRID_RRF_K` 20 hay 60 không đổi kết quả.

## Cross-encoder reranker

Sau RRF, nếu `RERANKER_MODEL` có giá trị, `CrossEncoderReranker` (`src/retrieval/reranker.py`) chấm lại từng cặp (câu hỏi, chunk) và sắp xếp các candidate theo điểm đó trước khi lấy `TOP_K`. Mô hình mặc định là `BAAI/bge-reranker-v2-m3`.

Thứ tự trong `RagService.retrieve()`:

1. FAISS lấy `HYBRID_CANDIDATES` chunk, lọc theo `SIMILARITY_THRESHOLD`.
2. Hybrid: xếp lại bằng RRF với BM25.
3. Loại trùng lặp.
4. Cross-encoder chấm điểm (văn bản cha nếu có), sắp xếp giảm dần theo `rerank_score`.
5. Lấy `TOP_K` chunk.

```makefile
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_DEVICE=cpu
```

- Mô hình chỉ được nạp ở câu hỏi đầu tiên có candidate. Để trống `RERANKER_MODEL` để tắt bước này.
- `RERANKER_MODEL` nhận tên trên Hugging Face hoặc thư mục local (tương đối theo thư mục project), ví dụ `storage/models/bge-reranker-v2-m3` khi ổ chứa cache Hugging Face không đủ chỗ (mô hình ~2,3GB).
- Khi bật reranker ở chế độ `dense`, FAISS vẫn lấy `HYBRID_CANDIDATES` chunk để reranker có tập ứng viên.
- Ngưỡng vẫn áp lên điểm dense, nên reranker không làm thay đổi khả năng từ chối câu hỏi ngoài tài liệu.
- Trên GPU, reranker chạy fp16. Với GPU 4GB, Qwen 4-bit cộng reranker vượt VRAM nên giữ `RERANKER_DEVICE=cpu`.

Kết quả và chi phí: xem [retrieval_eval.md](retrieval_eval.md#so-sánh-có-và-không-có-reranker) và Chương 8 trong [BAO_CAO.md](BAO_CAO.md).
