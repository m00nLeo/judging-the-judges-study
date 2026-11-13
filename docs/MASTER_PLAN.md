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

## 3. Lộ trình tuần tự & Đóng góp cải tiến mới (MAC-GLM)

```
Cổng 0 ──► Giai đoạn 1 (E0+E1, không API) ──► Giai đoạn 2 (E2, E3, E5 qua OpenRouter) ──► Giai đoạn 3 (E6: Counterfactual GLM Debiasing, $0 API) ──► report/findings.md
```

| Bước | Nội dung | Cần API? | Trạng thái |
| --- | --- | --- | --- |
| **Cổng 0** | Repo & cache tác giả có mở được không | Không | ✅ Xong — public, truy cập được |
| **00** setup_and_data_audit | Mirror data công khai, đối chiếu Table 1 | Không | ✅ Xong, 25/25 ô khớp |
| **01** E0_recompute_from_cache | Tái tạo số liệu từ cache (thu hẹp) | Không | ✅ Xong (Table 1 khớp 100%, W8 tie rate S1=32.75% vs S8=12.50%) |
| **02** E1_power_and_strata | Power/MDE, CI overlap (W2, W7) | Không | ✅ Xong (W2 khớp Wald CI, W7 xác nhận Flash S8 vs Claude S8 không có ý nghĩa thống kê, $p=0.6426$) |
| **03** E2_position_bias_protocols | Đo position bias 2 giao thức A vs B (W9) | Có (`OPENROUTER_API_KEY`) | ⬜ Sẵn sàng chạy (`dry-run` đã pass) |
| **04** E3_style_bias_new_judges | Style bias có còn ở judge mới? (W3) | Có (`OPENROUTER_API_KEY`) | ⬜ Sẵn sàng chạy (`dry-run` đã pass) |
| **05** E5_format_normalization | N1/N2 (bỏ markdown) vs S5 (CoT) vs B0 (W3) | Có (`OPENROUTER_API_KEY`) | ⬜ Sẵn sàng chạy (`dry-run` đã pass) |
| **06** E6_counterfactual_glm_debiasing | **Đóng góp mới (MAC-GLM)**: Kiểm toán đa cộng tuyến Style–Length–Truncation (RQ1) + Phân rã dấu vân tay thiên kiến $\hat{\boldsymbol{\gamma}}$ (RQ2) + Hậu hiệu chỉnh đối thực $\text{do}(\mathbf{z}=\mathbf{0})$ chi phí \$0 (RQ3) | **Không (\$0)** | ✅ RQ1 đã chạy xong trên dữ liệu gốc; RQ2 & RQ3 tự động chạy ngay khi có cache E3/E5 |

**Phát hiện quan trọng từ RQ1 (trong `06_E6_counterfactual_glm_debiasing.ipynb`)**:
Kiểm toán trực tiếp 100 cặp `STYLE + STYLE_BA` trong `data/custom/controlled_pairs.jsonl` của tác giả cho thấy tập dữ liệu bị **nhiễu nghiêm trọng (confounded)**:
- **74%** số cặp có bên Markdown dài hơn bên văn xuôi, với tỷ lệ số từ trung bình gấp **2.31 lần**.
- Chỉ **48/100** cặp có cả hai câu trả lời hoàn chỉnh; **52/100** cặp có ít nhất một câu trả lời bị cắt cụt giữa chừng (mid-sentence truncation, chiếm **22%** bất đối xứng hoàn toàn).
- Mật độ Markdown (`bold`, `header`, `list`) tương quan thuận mạnh với chênh lệch độ dài ($r = 0.49 \rightarrow 0.57$).

## 4. Đóng góp Phương pháp Mới: Multi-Artifact Counterfactual GLM (MAC-GLM)

Thay vì can thiệp xóa văn bản làm hỏng cấu trúc bảng/code (`N1`) hoặc gọi Chain-of-Thought đắt gấp 60 lần output token (`S5`), `src/experiments/counterfactual_glm.py` xây dựng mô hình hồi quy đối thực đa biến:
1. Trích xuất vector đặc trưng bề mặt phản đối xứng $\mathbf{z}_{AB} = [\log(\text{len}_A/\text{len}_B), \Delta\text{bold}, \Delta\text{header}, \Delta\text{list}, \Delta\text{struct}, \Delta\text{completeness}]$.
2. Khớp mô hình GLM có điều chuẩn $L_2$ để tách biệt **dấu vân tay thiên kiến bề mặt** $\hat{\boldsymbol{\gamma}} \pm 95\%\text{ CI}$ của từng Judge.
3. Thực hiện suy luận đối thực $\text{do}(\mathbf{z}_{AB} = \mathbf{0})$ bằng ước lượng phần dư trực giao (out-of-fold Neyman-orthogonal residual) để khử đồng thời thiên kiến Vị trí, Độ dài, Độ hoàn chỉnh và Định dạng Markdown với **chi phí token = 0**.

## 5. Cấu trúc thư mục

```
judging-the-judges-study/
  config.yaml               # seed, budget cap, judge mặc định, bảng giá OpenRouter
  docs/MASTER_PLAN.md        # file này
  CONTRIBUTING.md            # quy tắc làm việc + cách xử lý data
  scripts/
    fetch_upstream.py        # tải dữ liệu công khai đa nền tảng (Windows/Linux/Colab)
    fetch_upstream.sh        # wrapper bash
    run_pipeline.py          # chạy toàn bộ pipeline (--mode dry-run | pilot | full | e6-only)
  src/
    data_audit.py            # E0: nạp + đối chiếu cache aggregated
    stats_checks.py          # E1: power, CI overlap
    judges/clients.py        # client thống nhất qua OpenRouter (OPENROUTER_API_KEY) + fallback
    experiments/
      common.py              # hạ tầng dùng chung: cache jsonl, budget cap, retry, max_workers
      position_bias.py       # E2
      style_bias.py          # E3
      format_normalization.py# E5
      counterfactual_glm.py  # E6 (MAC-GLM: Confounding Audit + Bias Fingerprint + Debiasing)
  notebooks/00..06           # mỗi notebook chỉ gọi hàm trong src/, không chứa logic
  data/upstream/             # mirror từ repo tác giả — KHÔNG commit
  results/                   # output API thật — KHÔNG commit
  tests/                     # 40 unit tests (pytest) kiểm chứng toàn bộ src/, không gọi API thật
```

Xem `CONTRIBUTING.md` để biết chi tiết quy tắc chạy và xử lý từng loại file data.
