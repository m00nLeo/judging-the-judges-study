# Hướng dẫn làm việc trên repo này (cho người mới + AI)

Đọc `docs/MASTER_PLAN.md` trước — file đó nói dự án này là gì và đang ở bước nào. File này nói **cách làm cụ thể**: quy tắc bắt buộc, cách data hoạt động, và thứ tự chạy từng phase.

## 1. Quy tắc bắt buộc (áp dụng cho cả người và AI assistant)

1. **Không bịa số liệu hoặc cấu trúc file.** Chưa biết định dạng dữ liệu thì viết code khám phá trước (đọc vài dòng, in ra schema), không đoán rồi viết thẳng logic dựa trên đoán đó.
2. **Logic nằm trong `src/`, notebook chỉ gọi hàm.** Một notebook không được chứa quá vài dòng xử lý — nếu thấy mình viết loop/if phức tạp trong cell, dừng lại, chuyển vào `src/experiments/` hoặc `src/`, viết test, rồi mới gọi từ notebook.
3. **Mọi lời gọi API thật phải:**
   - Ghi cache từng mẫu ra file `.jsonl` (append, không ghi đè) — để phiên Colab bị ngắt vẫn resume được từ cache, không gọi lại.
   - Có trần ngân sách (`budget_usd` + `cost_model`) — chặn **trước** khi vượt, không phải cảnh báo sau.
   - Có retry + backoff cho lỗi tạm thời (503/429/timeout), fail ngay cho lỗi thật (sai key, 400).
   - Có timeout — không được để một request treo vô thời hạn.
   
   Hạ tầng này đã có sẵn trong `src/experiments/common.py` (`run_cached_calls`) — **dùng lại nó**, đừng viết loop gọi API mới từ đầu.
4. **Key chỉ lấy từ Colab Secrets hoặc biến môi trường** (`src/judges/clients.py::get_secret`). Không bao giờ dán key vào cell hay commit vào file.
5. **Mỗi hàm xử lý/tính toán phải có chế độ `--dry-run`/`dry_run=True`** không gọi API thật (dùng `build_judge("dry-run", dry_run=True)`), và phải chạy thử dry-run **trước** khi chạm vào key thật.
6. **Mỗi bước phải có test hoặc kiểm tra chấp nhận rõ ràng.** Viết test trong `tests/` dùng judge giả (xem các `ScriptedJudge`/`FakeTranslatorJudge` có sẵn làm mẫu) — không cần gọi API thật để test logic. Chạy `python -m pytest tests/ -q` phải pass trước khi coi bước đó là xong.
7. **Xong một bước thì dừng, không tự ý làm sang bước kế** trừ khi được yêu cầu rõ ràng. Báo cáo kết quả, chờ xác nhận.
8. **AI assistant không tự ý `git commit`** trừ khi chủ repo yêu cầu rõ ràng. Sửa file thoải mái, nhưng để người review diff rồi tự commit/push.

## 2. Cách bắt đầu (môi trường mới, máy mới, hoặc Colab)

```bash
git clone https://github.com/m00nLeo/judging-the-judges-study.git
cd judging-the-judges-study
pip install -q anthropic openai google-genai pyyaml
bash scripts/fetch_upstream.sh   # tải lại data công khai từ repo tác giả — xem mục 3
python -m pytest tests/ -q        # phải pass hết trước khi chạy bất kỳ notebook nào
```

Trên Colab, thêm key vào **Secrets** (icon 🔑 bên trái) trước khi chạy cell nào gọi API thật — tên biến chính xác: `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, (tuỳ chọn, free) `GROQ_API_KEY`, `OPENAI_API_KEY`.

## 3. Các loại file data và cách xử lý

| Thư mục/file | Là gì | Có commit không? | Lấy/tạo lại bằng cách nào |
| --- | --- | --- | --- |
| `data/upstream/` | Mirror các file **công khai** từ repo tác giả (aggregated metrics, controlled_pairs.jsonl, mt_bench.jsonl, llmbar.jsonl, configs, table CSV) | **Không** — `.gitignore` | `bash scripts/fetch_upstream.sh` (chỉ tải GitHub raw công khai, không cần key, miễn phí) |
| `config.yaml` | Seed, trần ngân sách, judge mặc định, cỡ mẫu từng phase | **Có** — không chứa key | Sửa trực tiếp, không có key thật trong này |
| `results/` (toàn bộ) | Output API thật: cache jsonl, bản dịch, annotation sheet + **key giải mã**, câu trả lời annotator | **Không** — `.gitignore` toàn bộ thư mục | Sinh ra khi chạy notebook; gửi trực tiếp (zip/download) cho người cần đọc, không push lên git |
| `results/e4_vietnamese_style/annotation_key.json` | Key giải mã thứ tự ngẫu nhiên của annotation sheet | **Tuyệt đối không chia sẻ cho annotator**, không commit | Sinh tự động bởi `generate_annotation_sheet()` |
| `tests/` | Test, dùng judge giả | Có | — |
| `src/`, `notebooks/`, `docs/`, `scripts/` | Code + tài liệu | Có | — |

**Vì sao `data/upstream/` không commit**: đó là bản sao của repo người khác, tải lại được bất cứ lúc nào, commit vào sẽ làm phình repo không cần thiết và không phải nội dung gốc của dự án này.

**Vì sao `results/` không commit**: chứa output API thật (có thể chứa dữ liệu dịch, câu trả lời người gán nhãn) gắn với một lần chạy cụ thể của một người cụ thể — không phải code, không nên nằm trong lịch sử git.

## 4. Thứ tự chạy các phase (xem `docs/MASTER_PLAN.md` mục 3 để biết trạng thái hiện tại)

Mọi notebook từ 03 trở đi đều theo đúng 1 khuôn: **dry-run (0 đồng) → pilot (mẫu nhỏ, kiểm tra cost/chất lượng) → full**. Không bỏ qua bước nào, kể cả khi đã quen.

1. `00_setup_and_data_audit.ipynb` — không API, chạy để xác nhận cache tác giả còn tải được.
2. `01_E0_recompute_from_cache.ipynb`, `02_E1_power_and_strata.ipynb` — không API.
3. `03_E2_position_bias_protocols.ipynb` — API, 2 judge mặc định (`config.yaml` → `judges.e2_position_bias`).
4. `04_E3_style_bias_new_judges.ipynb` và `06_E5_format_normalization.ipynb` — độc lập nhau, có thể làm song song.
5. `05_E4_vietnamese_style_study.ipynb` — Part A/C tự động, **Part B cần ≥3 người gán nhãn thật, không thể tự động hoá**. Nếu không có người, bỏ qua notebook này (giữ 00–03, 04, 06).
6. `07_analysis_and_approach.ipynb` (chưa viết) — tổng hợp, chốt `report/findings.md`, chỉ làm sau khi có đủ kết quả E2–E5 cần thiết.

Mỗi notebook ghi rõ trong markdown cell đầu: cần secret nào, cost dự kiến, và cần gửi file gì lại sau khi chạy xong.

## 5. Khi đổi judge mặc định

Sửa `config.yaml` → mục `judges.*` và `cost_per_1k_tokens.*`. Model id phải khớp đúng prefix mà `src/judges/clients.py::build_judge()` nhận diện (`claude-`, `gpt-`, `gemini-`, `llama-`) — thêm provider mới thì thêm cả class `*Judge` tương ứng (copy cấu trúc `GroqJudge` làm mẫu, nó chỉ là OpenAI-compatible endpoint khác base_url).
