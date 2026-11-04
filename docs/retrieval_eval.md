# Chạy đánh giá Dense, Hybrid retrieval và Reranker

Hướng dẫn tạo lại kết quả mục **7.5** trong [BAO_CAO.md](BAO_CAO.md): so sánh tìm kiếm thuần vector (`dense`) với tìm kiếm kết hợp (`hybrid`, dense + BM25), trên văn bản gốc và văn bản đã loại header.

Cách hoạt động của hybrid retrieval xem ở [vector_store.md](vector_store.md#hybrid-retrieval-dense--bm25), của reranker ở [vector_store.md](vector_store.md#cross-encoder-reranker).

Để đánh giá reranker (Chương 8), xem mục [So sánh có và không có reranker](#so-sánh-có-và-không-có-reranker).

Để so sánh document model theo trang với document model structured (hình, bảng, hộp, thời điểm, nguồn), chạy `python evaluation/run_document_model_eval.py`; thiết kế và kết quả ở [bao_cao_document_model.md](bao_cao_document_model.md).

## Script dùng

[`evaluation/run_kinhtevn_rag_eval.py`](../evaluation/run_kinhtevn_rag_eval.py) chạy 29 câu hỏi trong `evaluation/kinhtevn_questions.jsonl` qua `RagService.answer()` và in tổng kết ở cuối.

Hai biến môi trường quyết định cấu hình:

| Biến | Giá trị | Ý nghĩa |
| --- | --- | --- |
| `RETRIEVAL_MODE` | `dense` / `hybrid` | Chỉ FAISS, hoặc FAISS + xếp hạng lại bằng BM25 (RRF). |
| `DOCUMENT_STRIP_BOILERPLATE` | `false` / `true` | Giữ văn bản gốc, hoặc loại header/footer và số trang lặp lại. |

Biến đặt bằng `$env:` trong terminal được ưu tiên hơn `.env` (`load_dotenv` không ghi đè), nên không cần sửa `.env`. Index được build lại tự động khi `DOCUMENT_STRIP_BOILERPLATE` thay đổi.

Các tham số còn lại lấy từ `.env`: `e5-small`, chunking `recursive` 300/50, `TOP_K=5`, `HYBRID_CANDIDATES=10`, `HYBRID_RRF_K=60`, `SIMILARITY_THRESHOLD=0.795`.

Bảng 7.5 đo retrieval không có reranker. Khi chạy lại các lệnh dưới đây, thêm `$env:RERANKER_MODEL=""` để tắt reranker (nếu không, reranker trong `.env` cũng chạy và mỗi câu chậm thêm ~10 giây trên CPU).

## Lệnh chạy

Chạy trong PowerShell, tại thư mục `rag`:

```powershell
# 1. Dense, văn bản gốc
$env:RETRIEVAL_MODE="dense";  $env:DOCUMENT_STRIP_BOILERPLATE="false"; python evaluation/run_kinhtevn_rag_eval.py --report evaluation/report_dense_raw.jsonl

# 2. Dense, loại header
$env:RETRIEVAL_MODE="dense";  $env:DOCUMENT_STRIP_BOILERPLATE="true";  python evaluation/run_kinhtevn_rag_eval.py --report evaluation/report_dense_strip.jsonl

# 3. Hybrid, văn bản gốc
$env:RETRIEVAL_MODE="hybrid"; $env:DOCUMENT_STRIP_BOILERPLATE="false"; python evaluation/run_kinhtevn_rag_eval.py --report evaluation/report_hybrid_raw.jsonl

# 4. Hybrid, loại header (cấu hình mặc định của app)
$env:RETRIEVAL_MODE="hybrid"; $env:DOCUMENT_STRIP_BOILERPLATE="true";  python evaluation/run_kinhtevn_rag_eval.py --report evaluation/report_hybrid_strip.jsonl
```

`--report` tách file kết quả chi tiết cho từng cấu hình; nếu bỏ, cả bốn lần chạy cùng ghi đè `evaluation/kinhtevn_rag_console_report.jsonl`.

Chạy xong, xóa biến để terminal quay về cấu hình trong `.env`, rồi build lại index cho app:

```powershell
Remove-Item Env:RETRIEVAL_MODE, Env:DOCUMENT_STRIP_BOILERPLATE, Env:RERANKER_MODEL -ErrorAction SilentlyContinue
python -m src.app.main --rebuild
```

Tùy chọn khác của script:

- `--limit N`: chỉ chạy N câu đầu, dùng để thử nhanh.
- `--rebuild`: ép build lại index.
- `--include-answer-checks`: thêm các câu trong `evaluation/kinhtevn_answer_checks.jsonl`.

## Đọc kết quả

Mỗi câu hỏi in một dòng:

```text
[1/29] <câu hỏi>
  pages=[5, 6, 12, 3, 9] page_hit=True keyword_ratio=0.5
```

Cuối cùng là tổng kết (số liệu minh họa):

```json
{
  "questions": 29,
  "page_hit_rate": 1.0,
  "avg_answer_keyword_hit_ratio": 0.9,
  "no_context_pass_rate": null
}
```

| Trường | Ý nghĩa | Cột trong bảng 7.5 |
| --- | --- | --- |
| `page_hit_rate` | Tỷ lệ câu hỏi có ít nhất một nguồn thuộc trang đáp án trong top 5. | **Tỷ lệ trúng trang** |
| `avg_answer_keyword_hit_ratio` | Tỷ lệ trung bình `expected_keywords` xuất hiện trong **câu trả lời của LLM**. | Không tương ứng |
| `no_context_pass_rate` | Tỷ lệ câu hỏi loại `no_answer` được từ chối đúng. `null` nếu bộ câu hỏi không có loại này. | Không tương ứng |

## Lưu ý

- **Cột "Từ khóa đáp án có trong ngữ cảnh" không do script này đo** (dùng `run_retrieval_eval.py`, xem mục cuối). Script này đo từ khóa trong câu trả lời của LLM, không phải trong 5 chunk ngữ cảnh, nên `avg_answer_keyword_hit_ratio` sẽ khác 0,911 / 0,905 / 0,980 / 0,991.
- **Script gọi LLM** để sinh câu trả lời, nên mỗi cấu hình mất khá lâu dù bảng chỉ đánh giá phần retrieval. Dùng `--limit` khi chỉ muốn kiểm tra lệnh chạy được.
- `page_hit_rate` không phụ thuộc LLM: nguồn được lấy từ bước retrieval trước khi gọi mô hình, nên kết quả ổn định giữa các lần chạy.

## So sánh có và không có reranker

[`evaluation/run_retrieval_eval.py`](../evaluation/run_retrieval_eval.py) chạy các câu hỏi qua `RagService.retrieve()`, **không gọi LLM**, với tối đa bốn cấu hình trong cùng một lần chạy: `dense`, `hybrid`, `dense+rerank`, `hybrid+rerank`. Reranker lấy từ `RERANKER_MODEL` trong `.env`.

```powershell
# Cả bốn cấu hình (khoảng 10 phút trên CPU vì mỗi câu rerank mất ~10 giây)
python evaluation/run_retrieval_eval.py

# Chỉ một số cấu hình, ghi ra file khác
python evaluation/run_retrieval_eval.py --configs hybrid,hybrid+rerank --report evaluation/results/reranker_eval_hybrid.json

# Rerank 20 ứng viên thay vì 10 (mặc định)
$env:HYBRID_CANDIDATES="20"; python evaluation/run_retrieval_eval.py --report evaluation/results/reranker_eval.json
Remove-Item Env:HYBRID_CANDIDATES
```

Mỗi cấu hình in một dòng tổng kết. Kết quả chi tiết từng câu được ghi vào `evaluation/results/reranker_eval.json` (mặc định). Hai file có sẵn: `reranker_eval.json` (20 ứng viên, cả bốn cấu hình) và `reranker_eval_candidates10.json` (10 ứng viên, hybrid và hybrid+rerank).

| Trường | Ý nghĩa |
| --- | --- |
| `page_hit_rate` | Tỷ lệ câu có ít nhất một chunk thuộc trang đáp án trong top-k. |
| `mrr` | Trung bình nghịch đảo thứ hạng của chunk đầu tiên thuộc trang đáp án. |
| `context_keyword_ratio` | Tỷ lệ `expected_keywords` có trong văn bản của top-k chunk (ngữ cảnh đưa cho LLM). |
| `avg_retrieve_ms` | Thời gian `retrieve()` trung bình mỗi câu, không tính thời gian nạp mô hình. |

Kết quả trên 29 câu (e5-small, recursive 300/50, loại header, `TOP_K=5`, `HYBRID_CANDIDATES=20` trừ hai dòng cuối, reranker `bge-reranker-v2-m3` trên CPU, laptop RTX 3050 4GB / RAM 16GB):

| Cấu hình | page_hit | MRR | Keyword trong context | ms / câu |
| --- | --- | --- | --- | --- |
| dense | 0.966 | 0.880 | 0.957 | 18 |
| hybrid | 1.000 | 0.943 | 0.991 | 20 |
| dense+rerank | 1.000 | 0.977 | 0.991 | 19 683 |
| hybrid+rerank | 1.000 | 0.977 | 0.991 | 19 712 |
| hybrid, `HYBRID_CANDIDATES=10` | 1.000 | 0.943 | 0.991 | 19 |
| **hybrid+rerank, `HYBRID_CANDIDATES=10` (mặc định)** | 1.000 | 0.977 | 0.991 | 10 081 |

Script này cũng tái lập được cột "Từ khóa đáp án có trong ngữ cảnh" của bảng 7.5 (`context_keyword_ratio`), điều mà `run_kinhtevn_rag_eval.py` không làm được.

Không nên chạy reranker trên GPU cùng Qwen3-4B khi GPU chỉ có 4GB: đỉnh bộ nhớ lên ~4,26GB, vượt VRAM, và thời gian sinh câu trả lời tăng từ ~10 giây lên 120–176 giây.
