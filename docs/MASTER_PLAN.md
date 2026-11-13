# Master Plan: Judging the Judges (tái hiện và phân tích tiếp)

Cập nhật: 2026-11-14

## 1. Dự án này là gì

Repo này tái hiện lại và phân tích tiếp bài **"Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines"** (Sadman Kabir Soumik, TMLR 2026, [arXiv 2604.23178](https://arxiv.org/abs/2604.23178)). Bài gốc đo 4 loại bias (position, verbosity, style, self-preference) của 5 LLM judge, so sánh 9 chiến lược giảm bias, trên 3 benchmark.

Repo mã nguồn của tác giả: [sksoumik/judging-the-judges](https://github.com/sksoumik/judging-the-judges) (public). Tài liệu gốc chi tiết hơn — phân tích literature, các nghi vấn W1–W10, lý do chọn bài này — nằm ở file "Master Plan Phân tích Soumik 2026 (Judging the Judges).md" trong thư mục cha (Literature/), ngoài phạm vi repo này. File này là bản tóm tắt dành riêng cho codebase, để một người (hoặc AI) mở repo lần đầu biết ngay đang ở đâu và làm gì tiếp.

**Mục tiêu**: (1) chạy lại một phần kết luận của bài để rút insight, (2) từ đó chọn một approach cụ thể cho judge tiếng Việt.

## 2. Giới hạn dữ liệu đã xác nhận (không phải giả định)

Đã kiểm tra trực tiếp qua GitHub API, không đoán:

- Repo tác giả **public, truy cập được**. Cấu trúc khớp với README của họ.
- `results/raw/` (cache verdict theo từng mẫu) và `data/human_study/` (câu trả lời + key giải mã của annotator) đều bị `.gitignore` trong repo gốc — **không có trong repo công khai**. Điều này chặn việc tái tạo chính xác McNemar, Holm-Bonferroni, và tách tie theo từng mẫu cụ thể.
- Chỉ có `results/aggregated/*_metrics.json` (số tổng hợp: verdict_distribution, agreement_rate, cost, token) và các bảng đã tính sẵn trong `analysis/figures/` (table1, table2, …).
- `data/custom/controlled_pairs.jsonl` (375 cặp), `data/raw/mt_bench/mt_bench.jsonl` (3355 dòng), `data/raw/llmbar/llmbar.jsonl` (419 dòng) đều công khai.
- Xác nhận `gemini-2.5-flash` (judge mặc định của bài) đã bị Google **ngừng cấp cho user mới** (lỗi 404 thật khi gọi, tháng 11/2026) → đổi sang `gemini-3.8-flash`, ghi nhận là một confound (đổi thế hệ model, không chỉ đổi endpoint).

→ Hệ quả: **E0/E1 phải thu hẹp phạm vi** so với thiết kế gốc, chỉ đối chiếu số đã công bố thay vì tính lại từ đầu. Xem `notebooks/01_E0_recompute_from_cache.ipynb` và `02_E1_power_and_strata.ipynb` để biết chính xác phần nào làm được, phần nào bị chặn (`BLOCKED_ANALYSES` trong `src/experiments/`).

## 3. Lộ trình tuần tự (các Cổng quyết định)

```
Cổng 0 ──► Giai đoạn 1 (E0+E1, không API) ──► Cổng 1 ──► Giai đoạn 2 (E2–E5, có API) ──► Cổng 2/3 ──► report/findings.md
```

| Bước | Nội dung | Cần API? | Trạng thái (2026-11-14) |
| --- | --- | --- | --- |
| **Cổng 0** | Repo & cache tác giả có mở được không | Không | ✅ Xong — public, truy cập được |
| **00** setup_and_data_audit | Mirror data công khai, đối chiếu Table 1 | Không | ✅ Xong, 25/25 ô khớp |
| **01** E0_recompute_from_cache | Tái tạo số liệu từ cache (thu hẹp) | Không | ✅ Xong (một phần — xem mục 2) |
| **02** E1_power_and_strata | Power/MDE, CI overlap (W2, W7) | Không | ✅ Xong (một phần) |
| **Cổng 1** | Có đáng tốn API không | — | ✅ Đã chốt: có |
| **03** E2_position_bias_protocols | Đo position bias 2 giao thức (W9) | Có | 🔄 Đang chạy (gặp 503 từ Google, đang retry) |
| **04** E3_style_bias_new_judges | Style bias có còn ở judge mới? (W3) | Có | ⬜ Sẵn sàng chạy, chưa chạy |
| **06** E5_format_normalization | N1/N2 (bỏ markdown) vs S5 (CoT) (W3) | Có | ⬜ Sẵn sàng chạy, chưa chạy |
| **05** E4_vietnamese_style_study | Style bias tiếng Việt + người gán nhãn (W1, W10) | Có + **người** | ⬜ Cần ≥3 người gán nhãn, chưa xác nhận có không |
| **Cổng 2** | Giai đoạn 2 có đổi hướng nghiên cứu không | — | Chưa tới |
| **Cổng 3** | Chốt approach (xem mục 4) | — | Chưa tới |
| 07 analysis_and_approach | Tổng hợp, `report/findings.md` | Không | Chưa bắt đầu |

**Thứ tự chạy khuyến nghị** (có thể làm song song 04/06 vì độc lập nhau): 03 → 04 → 06 → 05 (05 phụ thuộc kết quả 04 để so sánh judge vs người).

## 4. Ba hướng approach ở Cổng 3 (chưa quyết định, chờ kết quả E2–E5)

- **(a)** Style bias vẫn mạnh và chuyển sang tiếng Việt → đề xuất "chuẩn hóa định dạng + CoT" cho judge tiếng Việt.
- **(b)** Bias phụ thuộc từng model cụ thể → đề xuất kiểm tra bias theo từng judge trước khi dùng, không khuyến nghị chung.
- **(c)** E0/E1 làm sụp kết luận chính của bài → chuyển hướng sang kiểm định lại với cỡ mẫu đủ lớn.

## 5. Cấu trúc thư mục

```
judging-the-judges-study/
  config.yaml              # seed, budget cap, judge mặc định, sampling — đọc trước khi chạy bất kỳ notebook nào
  docs/MASTER_PLAN.md       # file này
  CONTRIBUTING.md           # quy tắc làm việc + cách xử lý data cho người/AI mới
  scripts/fetch_upstream.sh # tải lại data công khai từ repo tác giả (chạy lại mỗi phiên Colab mới)
  src/
    data_audit.py           # E0: nạp + đối chiếu cache aggregated
    stats_checks.py         # E1: power, CI overlap
    judges/clients.py        # client thống nhất Gemini/Claude/GPT/Groq, đọc key từ Colab Secrets/env
    experiments/
      common.py              # hạ tầng dùng chung: cache jsonl, budget cap, retry, chạy song song (max_workers)
      position_bias.py        # E2
      style_bias.py            # E3
      format_normalization.py # E5
      vietnamese_style.py      # E4 (translate + annotation sheet + ingest)
  notebooks/00..07          # mỗi notebook chỉ gọi hàm trong src/, không chứa logic
  data/upstream/            # mirror từ repo tác giả — KHÔNG commit, tự tạo lại bằng fetch_upstream.sh
  results/                  # output API thật (cache, chi phí, annotation) — KHÔNG commit, xem CONTRIBUTING.md
  tests/                    # test cho mọi hàm trong src/, dùng judge giả, không gọi API thật
```

Xem `CONTRIBUTING.md` để biết chi tiết quy tắc chạy và xử lý từng loại file data.
