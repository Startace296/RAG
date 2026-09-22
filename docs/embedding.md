# Embedding Models

Tài liệu này mô tả cách project chọn, cấu hình và đánh giá embedding model cho RAG tiếng Việt.

## Vai Trò Của Embedding

Embedding chuyển câu hỏi và chunk tài liệu thành vector số. Vector store dùng các vector này để tìm chunk gần nhất với câu hỏi.

Nếu embedding model yếu với tiếng Việt, retrieval có thể lấy sai chunk dù chunking tốt.

## Cấu Hình Hiện Tại

`.env`:

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DOCUMENT_PREFIX=
EMBEDDING_QUERY_PREFIX=
EMBEDDING_BENCHMARK_MODELS=miniLM-multilingual,mpnet-multilingual,e5-small,e5-base
```

Nếu đổi `EMBEDDING_MODEL`, cần rebuild index:

```powershell
python -m src.main --rebuild
```

Nếu không rebuild, FAISS index vẫn chứa vector được tạo từ model cũ.

## Model Đã Đưa Vào Benchmark

### miniLM-multilingual

Model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Ưu điểm:

- Nhẹ, nhanh.
- Dễ chạy local.
- Baseline multilingual ổn cho tiếng Việt.

Nhược điểm:

- Vector 384 chiều, chất lượng semantic retrieval có thể thấp hơn model lớn.
- Không chuyên retrieval bằng E5.

Nên dùng khi:

- Máy yếu.
- Cần demo nhanh.
- Tài liệu ít và câu hỏi tương đối gần từ khóa.

### mpnet-multilingual

Model:

```text
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Ưu điểm:

- Multilingual sentence embedding mạnh hơn MiniLM.
- Vector 768 chiều, thường biểu diễn ngữ nghĩa tốt hơn.

Nhược điểm:

- Nặng và chậm hơn MiniLM.
- Tốn RAM/VRAM hơn.

Nên dùng khi:

- Muốn baseline multilingual chất lượng cao hơn.
- Máy đủ tài nguyên.

### e5-small

Model:

```text
intfloat/multilingual-e5-small
```

Prefix:

```text
document_prefix = "passage: "
query_prefix = "query: "
```

Ưu điểm:

- Được thiết kế cho retrieval.
- Nhẹ hơn E5-base.
- Thường là ứng viên tốt cho RAG tiếng Việt khi cần cân bằng tốc độ và chất lượng.

Nhược điểm:

- Cần dùng đúng prefix `query:` và `passage:`.
- Có thể kém model lớn trong câu hỏi khó.

Nên dùng khi:

- Cần model tốt hơn MiniLM cho retrieval.
- Muốn ưu tiên tiếng Việt/multilingual nhưng vẫn giữ tốc độ.

### e5-base

Model:

```text
intfloat/multilingual-e5-base
```

Prefix:

```text
document_prefix = "passage: "
query_prefix = "query: "
```

Ưu điểm:

- Retrieval multilingual mạnh.
- Ứng viên tốt nhất trong danh sách này nếu máy đủ tài nguyên.

Nhược điểm:

- Nặng hơn, chậm hơn.
- Build index và encode query lâu hơn.

Nên dùng khi:

- Ưu tiên chất lượng retrieval.
- Máy có đủ RAM/VRAM.
- Tập tài liệu/câu hỏi có nhiều diễn đạt khác từ nhau.

## Model Nào Tốt Cho Tiếng Việt?

Không nên chọn chỉ bằng cảm tính. Với project này, chọn model dựa trên benchmark:

```powershell
python -m src.embedding_benchmark
```

Quy tắc chọn thực dụng:

- Nếu `e5-base` có `MRR`/`nDCG@5` cao nhất và latency chấp nhận được: chọn `e5-base`.
- Nếu `e5-base` quá chậm nhưng `e5-small` gần bằng điểm: chọn `e5-small`.
- Nếu máy yếu hoặc cần demo nhanh: chọn `miniLM-multilingual`.
- Nếu `mpnet-multilingual` cao hơn E5 trên tài liệu cụ thể này: có thể chọn `mpnet-multilingual`, nhưng vẫn cần kiểm tra latency.

Với tiếng Việt RAG, nên ưu tiên thử:

```text
e5-small -> e5-base -> mpnet-multilingual -> miniLM-multilingual
```

Kết luận chỉ nên chốt sau khi có bảng số trên chính tài liệu và câu hỏi của project.

## Benchmark Embedding

Benchmark dùng cùng file ground truth:

```text
evaluation/chunking_questions.jsonl
```

Chạy toàn bộ model trong `.env`:

```powershell
python -m src.embedding_benchmark
```

Chạy một vài model:

```powershell
python -m src.embedding_benchmark --models e5-small e5-base
```

Chạy bằng raw Hugging Face model name:

```powershell
python -m src.embedding_benchmark --models intfloat/multilingual-e5-small
```

Benchmark in các cột:

```text
model       dimension  chunks  recall@5  mrr  ndcg@5  load_s  build_s  query_ms  retrieval_ms
```

Ý nghĩa:

- `model`: alias/model đang test.
- `dimension`: số chiều vector.
- `chunks`: số chunk được embed.
- `recall@5`: top 5 có chứa trang/chunk đúng không.
- `mrr`: kết quả đúng nằm càng cao thì càng tốt.
- `ndcg@5`: chất lượng thứ hạng top 5.
- `load_s`: thời gian load model.
- `build_s`: thời gian embed toàn bộ chunk và build index.
- `query_ms`: thời gian encode trung bình một câu hỏi.
- `retrieval_ms`: thời gian search FAISS trung bình.

## Chọn Model Cho App Chính

Nếu benchmark cho thấy `e5-small` tốt, cấu hình:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DOCUMENT_PREFIX=passage: 
EMBEDDING_QUERY_PREFIX=query: 
```

Nếu dùng `e5-base`:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_DOCUMENT_PREFIX=passage: 
EMBEDDING_QUERY_PREFIX=query: 
```

Sau đó rebuild:

```powershell
python -m src.main --rebuild
```

## Trạng Thái Hiện Tại

Đã có:

- Nhiều embedding model candidate.
- Prefix support cho E5.
- Benchmark định lượng bằng `Recall@k`, `MRR`, `nDCG@k`, latency.
- Dùng chung ground truth với chunking benchmark.

Cần làm sau khi chạy benchmark:

- Ghi bảng kết quả thật vào báo cáo.
- Chọn model chính thức cho `.env`.
- Nếu cần chất lượng cao hơn, bổ sung thêm model tiếng Việt chuyên biệt và benchmark lại.

