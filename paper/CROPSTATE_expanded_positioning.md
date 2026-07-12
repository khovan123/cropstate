# CROPSTATE — Định vị lại bài báo (paper-ready)

> **Cách dùng:** các khối tiếng Anh dưới đây bê thẳng vào bài. Ghi chú tiếng Việt (in nghiêng) là hướng dẫn, KHÔNG đưa vào bài. Mọi con số lấy từ `CROPSTATE_RESULTS/` (2.900 ảnh RiceSEG, test 263 ảnh, seed 42 + 7 + 123).

---

## 1. Chuyển định vị

*Bài pilot cũ: "552 ảnh, chỉ phân loại 6 giai đoạn, retrieval để ngoài phạm vi." Bản mở rộng nâng lên hai trụ cột mới: (a) giao thức đánh giá đáng tin cậy, (b) tra cứu tri thức an-toàn-theo-giai-đoạn.*

| | Pilot (đã xuất bản) | Bản mở rộng (công việc này) |
|---|---|---|
| Dữ liệu | 552 ảnh | **2.900 ảnh (RiceSEG)** |
| Đóng góp chính | Phân loại 6 giai đoạn | Phân loại + **audit rò rỉ** + **retrieval an toàn** |
| Đánh giá | multi-seed lẫn split/init | **fixed-split multi-seed**, tách phương sai |
| Retrieval | ngoài phạm vi | **trong phạm vi, 420 kịch bản, có kiểm định** |

---

## 2. Abstract (bản viết lại — English, paste-ready)

> Timely knowledge of a rice crop's growth stage underpins stage-specific agronomic
> decisions. We present CROPSTATE, a system that classifies field images into the six
> canonical BBCH macro-stages and retrieves stage-appropriate agronomic guidance. Beyond
> classification accuracy, our focus is **evaluation reliability and retrieval safety**.
> First, we audit the 2,900-image benchmark for near-duplicate leakage across splits and
> confirm the splits are leak-free at every reasonable perceptual-hash threshold, so reported
> scores are not inflated by memorisation. Second, using a fixed-split, multi-seed protocol we
> show that an ordinal-aware loss does not change overall accuracy but **stabilises recall on
> the safety-critical reproductive stage** (σ=0.000 vs 0.071 across seeds). Third, we cast
> confidence handling as **selective prediction**: abstaining on the least-confident ~37% of
> cases halves the error rate (0.205 → 0.097). Finally, on 420 automatically-derived retrieval
> scenarios, our confidence-adaptive soft gating **matches the strongest baseline in ranking
> quality while reducing stage-incompatible retrievals ~3×** — the property a stage-aware
> advisory system most needs. CROPSTATE claims no new architecture or dataset; its contribution
> is a **reliable, safety-oriented evaluation of the full classification-to-retrieval pipeline.**

---

## 3. Contributions (English, paste-ready)

> 1. **Leak-free benchmark audit.** A perceptual-hash near-duplicate audit over 2,900 images
>    across a threshold sweep, showing **zero cross-split leaking pairs** at any reasonable
>    threshold — a prerequisite for trusting every downstream number.
> 2. **Reliable stage-classification evaluation.** A fixed-split multi-seed protocol
>    (accuracy 0.807±0.009 for ResNet-18) that separates initialisation variance from split
>    variance, plus per-image significance tests (McNemar, paired bootstrap).
> 3. **Stability of ordinal loss on the reproductive stage.** Ordinal-aware training yields
>    identical reproductive recall across all seeds (0.917) versus a highly variable baseline
>    (0.861±0.071), with a transparent statement that the *mean* difference is not significant
>    at this sample size.
> 4. **Selective prediction for deployment.** A risk–coverage analysis (AURC=0.125) showing the
>    error rate halves when the model abstains on its least-confident cases.
> 5. **Stage-safe retrieval.** A confidence-adaptive soft-gating retriever that, over 420
>    scenarios, matches the strongest baseline on nDCG/recall while cutting stage-incompatible
>    retrievals from 0.159 to 0.039 (sIRR).

---

## 4. Bảng kết quả tổng hợp (paper-ready)

**Bảng 1 — Vision (fixed-split, 3 seed, mean±std):**

| Loss | Accuracy | Macro-F1 | MASD ↓ | ECE ↓ | Reproductive recall |
|---|---|---|---|---|---|
| Cross-entropy | **0.807 ± 0.009** | 0.798 ± 0.019 | **0.223 ± 0.020** | **0.057 ± 0.015** | 0.861 ± 0.071 |
| Ordinal | 0.802 ± 0.008 | **0.801 ± 0.006** | 0.234 ± 0.010 | 0.067 ± 0.014 | **0.917 ± 0.000** |
| Focal | 0.773 ± 0.006 | 0.764 ± 0.008 | 0.265 ± 0.005 | 0.173 ± 0.019 | 0.889 ± 0.039 |

*Ordinal thắng ở tính ổn định reproductive (σ=0) và macro-F1; không thắng accuracy/MASD. Pooled n=72: p=0.289 (không ý nghĩa) → chỉ claim "stable".*

**Bảng 2 — Retrieval (420 kịch bản, k=5):**

| Method | nDCG ↑ | Recall ↑ | sIRR ↓ (an toàn) |
|---|---|---|---|
| B0 no-gating | 0.165 | 0.118 | 0.136 |
| B1 query-expansion | **0.336** | 0.193 | 0.159 (kém an toàn nhất) |
| B2 hard-filter | 0.259 | 0.181 | **0.002** (cứng nhắc) |
| B3 fixed-soft | 0.267 | 0.164 | 0.072 |
| **P adaptive (ours)** | 0.314 | **0.199** | 0.039 |
| oracle | 0.349 | 0.222 | 0.022 |

*P_adaptive thắng B0/B3 mọi metric (CI không chứa 0); ngang B1 về nDCG/recall nhưng sIRR thấp hơn 0.12 (~3×). Xem `retrieval_tradeoff.png`.*

---

## 5. Limitations (English, paste-ready — bản chốt)

> Reproductive stage has n=24 test images; per-image significance is underpowered (McNemar
> p=0.50, and p=0.29 even when the three seeds are pooled to n=72), so we frame ordinal's
> benefit as recall **stability across seeds** (σ=0 vs 0.071) rather than a significant
> single-run gain. Retrieval relevance is auto-derived from stage-compatibility, not
> expert-annotated. Temporal fusion uses simulated trajectories (no real time-series) and is
> reported as future work. Focal loss degraded calibration and is included only as a negative
> ablation.

---

## 6. Hình dùng cho bài

- `CROPSTATE_RESULTS/novelty/selective_prediction.png` — đường risk–coverage (đóng góp #4).
- `CROPSTATE_RESULTS/retrieval/retrieval_tradeoff.png` — đánh đổi chất lượng vs an toàn (đóng góp #5).
- (khuyến nghị bổ sung) biểu đồ cột reproductive recall per-seed để làm nổi bật σ=0 vs σ=0.071.
