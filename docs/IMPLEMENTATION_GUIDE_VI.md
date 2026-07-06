# HƯỚNG DẪN TRIỂN KHAI TOÀN BỘ NGHIÊN CỨU CROPSTATE

## 1. Định vị đề tài sau khi cập nhật

CROPSTATE không còn là chatbot hỏi - đáp. Đầu vào bắt buộc là ảnh ruộng hoặc cây lúa. Hệ thống thực hiện hai tầng:

1. Mô hình thị giác nhận dạng một trong sáu giai đoạn sinh trưởng và xuất phân phối xác suất.
2. Bộ truy xuất dùng toàn bộ phân phối đó để ưu tiên tài liệu nông nghiệp phù hợp, thay vì chỉ hard-filter theo nhãn top-1.

Câu hỏi khoa học chính:

- RQ1: Có thể nhận dạng sáu macro-stage của lúa từ ảnh thực địa với độ chính xác và độ tổng quát nào trên ruộng/mùa chưa thấy?
- RQ2: Khi mô hình ảnh sai hoặc không chắc chắn, soft gating có làm retrieval suy giảm ít hơn hard filtering hay không?

## 2. Phân loại, không phải object detection

Với dữ liệu mẫu dạng patch 512 x 512, nhiệm vụ chính là image classification:

```text
ảnh -> một phân phối trên 6 stage
```

Object detection chỉ cần khi muốn khoanh vùng bông lúa, gốc hoặc thân như một auxiliary task. Không cần bounding box để trả lời RQ1 và RQ2.

## 3. Đánh giá bộ ảnh mẫu hiện tại

Sáu ảnh đi kèm package chỉ đủ để kiểm tra đọc file, manifest, augmentation và leakage audit. Chúng không đủ để train mô hình khoa học vì:

- số lượng quá ít;
- chủ yếu thể hiện giai đoạn sớm;
- chưa có panicle/grain/ripening rõ;
- tên file có `subset_overlap`, cho thấy nguy cơ patch gần trùng;
- chưa có field ID, ngày chụp, BBCH và nhãn chuyên gia.

Không gán nhãn chỉ dựa vào cảm giác nhìn ảnh. Hãy thu metadata và xác nhận hình thái.

## 4. Số lượng và độ đa dạng cần đạt

### Pilot

- 50-100 ảnh mỗi lớp;
- tổng 300-600 ảnh;
- ít nhất 3 field hoặc capture group khác nhau mỗi lớp;
- dùng để kiểm tra feasibility, không phải kết luận cuối.

### Main experiment

- mục tiêu thực tế: 300-500 ảnh mỗi lớp;
- tổng 1.800-3.000 ảnh;
- nhiều field, ngày, mùa, giống, camera và điều kiện sáng;
- test set phải có field hoặc season không xuất hiện trong train.

Một nghìn patch từ cùng vài ảnh gốc không có giá trị tương đương một nghìn sample độc lập.

## 5. Metadata bắt buộc

Mỗi ảnh/patch phải có:

```text
image_id
image_path
parent_image_id
field_id
capture_session
capture_date
season
variety
days_after_sowing
bbch_code
macro_stage
source
license
annotator_1
annotator_2
adjudicated_label
split
```

`parent_image_id` đặc biệt quan trọng với dữ liệu crop/overlap.

## 6. Quy trình thu ảnh

Mỗi thời điểm nên chụp:

- một ảnh top-down để thấy canopy;
- một ảnh ngang để thấy chiều cao và thân;
- một ảnh cận gốc ở giai đoạn sớm;
- một ảnh cận panicle từ heading trở đi.

Nếu hệ thống cuối chỉ nhận một ảnh, hãy chuẩn hóa protocol chụp và chọn góc có đủ dấu hiệu hình thái cho cả sáu lớp. Một lựa chọn thực tế là yêu cầu ảnh nghiêng 30-45 độ, thấy cả canopy và phần thân/panicle.

## 7. Chia train/validation/test

Không random split theo patch. Chia theo nhóm:

Ưu tiên:

1. field_id;
2. parent_image_id;
3. capture_session;
4. ngày hoặc season.

Ví dụ:

```text
Train: Field A, B, C
Validation: Field D
Test: Field E
```

Tất cả patch từ cùng ảnh gốc phải ở cùng split. Chạy `scripts/audit_dataset.py` trước mỗi experiment.

Nếu dùng `Image_Manifest_Template` từ Google Sheet, export sheet đó thành CSV rồi convert sang manifest chuẩn:

```bash
python scripts/convert_image_manifest.py \
  --input KNOWLEDGE_BASE_SAMPLE/Image_Manifest_Template.csv \
  --data-root data \
  --output data/image_manifest.csv
```

Converter chỉ giữ ảnh `usable` với nhãn S01-S06. S07 Uncertain và S08 Unusable được ghi vào `data/image_manifest_excluded.csv`, không đưa vào supervised training. Nếu bật checksum mặc định, exact duplicates được ghi vào `data/image_manifest_duplicates.csv`.

Audit manifest trước khi train:

```bash
python scripts/audit_dataset.py \
  --manifest data/image_manifest.csv \
  --data-root data \
  --checksum
```

Nếu chưa điền xong sheet manifest, có thể tạo manifest pilot trực tiếp từ sáu folder stage:

```bash
python scripts/build_stage_manifest.py \
  --data-root data \
  --output data/stage_folder_manifest.csv

python scripts/audit_dataset.py \
  --manifest data/stage_folder_manifest.csv \
  --data-root data \
  --checksum
```

Manifest pilot này dùng label theo folder và để `split=unassigned`; `train_vision.py` sẽ tự chia split theo `parent_image_id` khi chạy training. Không dùng manifest này để báo cáo khoa học nếu chưa review annotation.

## 8. Experiment 0 - Vision feasibility

### Mục tiêu

Kiểm tra ảnh và nhãn có đủ tín hiệu để phân biệt stage hay không.

### Baseline

- Majority class.
- HOG + SVM hoặc shallow-feature baseline.
- ResNet18 pretrained.
- EfficientNet-B0 pretrained.

### Metrics

- Accuracy;
- Macro-F1;
- per-class recall;
- confusion matrix;
- MASD: mean absolute distance giữa stage dự đoán và ground truth;
- ECE và Brier score sau calibration.

### Tiêu chí quyết định

Không có một threshold tuyệt đối, nhưng phải quan sát:

- model vượt majority rõ rệt;
- Macro-F1 không chỉ đến từ một vài lớp dễ;
- lỗi chủ yếu giữa stage kề nhau thay vì nhảy xa;
- field-based test không sụp đổ hoàn toàn so với random split;
- ảnh/nhãn các lớp cuối có đủ dấu hiệu panicle và grain.

Nếu không đạt, sửa dataset trước khi xây retrieval.

## 9. Vision training

Trên Google Colab, cài package ở chế độ editable rồi fine-tune pretrained model từ các folder stage đã chia sẵn:

```bash
pip install -r requirements.txt
pip install -e .
export PYTHONPATH=src
python scripts/train_vision.py \
  --data-root data \
  --config configs/vision.yaml \
  --output results/vision_resnet18
```

Config mặc định dùng fine-tuning hai pha: 3 epoch đầu freeze backbone và chỉ train classifier head, sau đó unfreeze toàn bộ model với learning rate thấp hơn cho backbone. Có thể override trực tiếp:

```bash
python scripts/train_vision.py \
  --data-root data \
  --config configs/vision.yaml \
  --freeze-backbone-epochs 3 \
  --learning-rate 0.0003 \
  --backbone-learning-rate 0.00003 \
  --output results/vision_resnet18_finetune
```

Để fine-tune tiếp từ checkpoint đã train trước đó, dùng `--resume-checkpoint` và ghi output sang thư mục mới:

```bash
python scripts/train_vision.py \
  --manifest results/vision_resnet18_finetune/manifest_from_folders.csv \
  --data-root data \
  --config configs/vision.yaml \
  --resume-checkpoint results/vision_resnet18_finetune/best_checkpoint.pt \
  --freeze-backbone-epochs 0 \
  --learning-rate 0.0001 \
  --backbone-learning-rate 0.00001 \
  --output results/vision_resnet18_finetune_round2
```

Nếu đã có manifest metadata đầy đủ, truyền thêm `--manifest data/image_manifest.csv`. Nếu không truyền manifest, script tự tạo manifest từ các thư mục `01_establishment`, `02_tillering`, `03_stem_booting`, `04_reproductive`, `05_grain_filling`, `06_ripening`, kể cả khi tên thư mục có khoảng trắng đầu. Split mặc định theo `parent_image_id` để các patch overlap từ cùng ảnh gốc không rơi vào nhiều split. Các folder `07_uncertain` và `08_unusable` không thuộc six-class training.

Augmentation an toàn:

- resize và random crop nhẹ;
- horizontal flip;
- brightness/contrast vừa phải;
- rotation nhỏ.

Không dùng augmentation làm biến dạng hình thái hoặc xóa panicle.

### Test một ảnh bằng file chọn từ máy

Sau khi training xong trên Colab, chạy cell này để mở hộp thoại chọn ảnh từ máy tính:

```python
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded))
print("Uploaded:", image_path)
```

Sau đó chạy inference bằng checkpoint tốt nhất:

```bash
!PYTHONPATH=src python scripts/predict_image.py \
  --checkpoint results/vision_resnet18/best_checkpoint.pt \
  --image "{image_path}"
```

Kết quả trả về `predicted_stage`, `confidence`, và `stage_belief` là phân phối xác suất trên sáu stage.

## 10. Calibration và stage belief

Mô hình phải xuất vector sáu xác suất, không chỉ nhãn top-1:

```text
[establishment, tillering, stem_booting,
 reproductive, grain_filling, ripening]
```

Dùng temperature scaling trên validation set. Báo cáo ECE và Brier score trước/sau calibration. Xuất belief của test set bằng `scripts/export_predictions.py`.

## 11. Confidence

Image-only confidence kết hợp:

- normalized entropy concentration;
- top-two margin.

Nếu có temporal prior từ ngày gieo, thêm agreement bằng Jensen-Shannon divergence. Confidence chỉ là routing signal, không chứng minh stage đúng. Case concentrated-but-wrong phải được phân tích riêng.

## 12. Knowledge base

Mỗi chunk có:

```json
{
  "chunk_id": "SRC001_C001",
  "text": "...",
  "topic": "water_management",
  "stage_compatibility": [0, 1, 0.6, 0, 0, 0],
  "authority_score": 0.9,
  "source_id": "SRC001",
  "review_status": "reviewed"
}
```

Các topic cố định:

- water management;
- nutrient management;
- disease risk;
- pest risk;
- weed management.

Không có câu hỏi tự do. Hệ thống tự chạy retrieval cho từng topic sau khi nhận ảnh.

Nếu dùng `Knowledge_Chunks` từ Google Sheet, export sheet đó thành CSV rồi convert sang JSONL:

```bash
python scripts/convert_knowledge_base.py \
  --input KNOWLEDGE_BASE_SAMPLE/Knowledge_Chunks.csv \
  --output data/knowledge_chunks.jsonl
```

Các chunk có `review_status=sample_only_not_agronomic_ground_truth` chỉ dùng để test pipeline, không dùng làm khuyến nghị canh tác thật.

## 13. Retrieval baselines

- B0 Ungated hybrid: BM25 + dense + RRF, không dùng stage.
- B1 Hard top-1: dùng stage dự đoán để filter.
- B2 Fixed soft: dùng toàn bộ belief nhưng beta cố định.
- P Adaptive soft: beta thay đổi theo confidence.
- Oracle: dùng ground-truth stage để đo upper bound của stage-aware retrieval.

Mọi phương pháp dùng cùng corpus, chunking, embedding model, candidate depth và top-k.

## 14. RRF và chuẩn hóa

BM25 và dense tạo hai ranking. RRF hợp nhất ranking. Sau đó min-max normalize RRF trong candidate pool của từng query trước khi cộng stage score. Không cộng raw RRF với compatibility [0,1].

## 15. Experiment 1 - Oracle retrieval

Dùng ground-truth stage của ảnh để tạo belief one-hot hoặc narrow belief. So sánh B0, hard filter, fixed soft và adaptive soft.

Mục đích: xác minh stage annotation thực sự giúp retrieval khi loại bỏ lỗi vision.

Metrics:

- P@5;
- R@5;
- nDCG@5;
- SIRR@5.

Nếu oracle-stage không giảm SIRR hoặc không tạo lợi ích rõ, cần sửa knowledge metadata, scenario labels hoặc corpus trước.

## 16. Experiment 2 - End-to-end image-driven retrieval

Pipeline:

```text
Test image -> calibrated vision probabilities -> confidence
-> fixed topic retrieval -> adaptive re-ranking -> top-k evidence
```

Chia kết quả theo nhóm:

- correct and high confidence;
- correct and low confidence;
- adjacent-stage error;
- distant-stage error;
- concentrated but incorrect;
- optional vision-time conflict.

So sánh degradation:

```text
Delta nDCG = nDCG_correct - nDCG_condition
Delta SIRR = SIRR_condition - SIRR_correct
```

RQ2 được hỗ trợ khi adaptive soft có degradation nhỏ hơn hard filtering trên phần lớn nhóm lỗi và paired confidence interval cho thấy effect có ý nghĩa thực tế.

## 17. SIRR

SIRR@k là tỷ lệ tài liệu top-k có compatibility bằng 0 đối với ground-truth stage. Nếu hard filter trả ít hơn k tài liệu, vẫn chia cho k để hệ thống bị phạt vì mất recall.

## 18. Ablation

Chạy ít nhất:

- Full adaptive method;
- fixed beta;
- entropy only;
- không top-two margin;
- không JSD agreement;
- không adjacent-stage compatibility;
- không source score;
- không RRF normalization.

## 19. Statistical analysis

Vì các method chạy trên cùng image-topic cases, dùng paired analysis:

- 95% paired bootstrap CI;
- Wilcoxon signed-rank hoặc paired permutation;
- Holm correction khi so nhiều baseline;
- effect size.

Không chỉ báo cáo p-value.

## 20. Error analysis

Gán category cho failure:

- ambiguous image;
- wrong/weak ground truth;
- patch removed decisive structure;
- background shortcut;
- adjacent-stage confusion;
- confident distant error;
- incorrect chunk-stage metadata;
- retriever miss;
- generic chunk ranked too high;
- missing source knowledge.

## 21. Lộ trình 10 tuần

### Tuần 1

- khóa 6 stage và protocol chụp;
- hoàn thiện annotation guideline;
- audit nguồn ảnh và license.

### Tuần 2-3

- thu ảnh pilot;
- ghi parent/field/session metadata;
- double-annotate subset;
- split theo field.

### Tuần 4

- train ResNet18 và EfficientNet-B0;
- chạy Experiment 0;
- sửa dataset nếu leakage hoặc class confusion bất thường.

### Tuần 5

- calibration;
- export probability vectors;
- phân tích confusion và MASD.

### Tuần 6

- xây knowledge chunks và compatibility vectors;
- implement B0, hard, fixed soft, adaptive soft.

### Tuần 7

- tạo image-topic relevance labels;
- chạy oracle retrieval.

### Tuần 8

- chạy end-to-end retrieval;
- robustness grouping và degradation.

### Tuần 9

- ablation, sensitivity, bootstrap và tests.

### Tuần 10

- error analysis;
- điền bảng paper từ kết quả thật;
- cập nhật Results, Discussion và Conclusion.

## 22. Điều kiện hoàn thành tối thiểu

- image dataset đủ cả sáu lớp;
- split không leakage;
- ít nhất hai pretrained vision baseline;
- calibration metrics;
- knowledge corpus có provenance và stage metadata;
- oracle retrieval;
- end-to-end image retrieval;
- nDCG, SIRR và degradation;
- ablation;
- paired statistics;
- không có số liệu giả.

## 23. Việc cần làm ngay với sáu ảnh mẫu

1. Xác định mỗi ảnh có phải patch từ ảnh lớn không.
2. Ghi parent_image_id và tọa độ crop nếu có.
3. Tìm field, ngày chụp, days after sowing và BBCH.
4. Nhờ hai người gán nhãn độc lập.
5. Không đưa vào test nếu cùng field/parent với train.
6. Chỉ dùng làm early-stage samples nếu dấu hiệu morphology đủ rõ.
7. Tiếp tục thu ảnh reproductive, grain filling và ripening có panicle rõ.
