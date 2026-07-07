# Retrieval trong CROPSTATE

## Retrieval là gì?

Trong CROPSTATE, retrieval là bước tìm kiếm tài liệu phù hợp trong knowledge base sau khi vision model đã xác định giai đoạn sinh trưởng của lúa.

Retrieval không dùng để phân loại ảnh.

Luồng đúng là:

```text
Ảnh cây lúa
-> vision model dự đoán giai đoạn
-> sinh xác suất của 6 giai đoạn
-> retrieval tìm kiến thức phù hợp
-> reranking ưu tiên tài liệu đúng giai đoạn
-> trả về bằng chứng và khuyến nghị
```

## Vision model và retrieval khác nhau thế nào?

| Thành phần | Công dụng |
| --- | --- |
| Vision model | Nhìn ảnh và dự đoán lúa đang ở giai đoạn nào |
| Retrieval | Tìm thông tin chăm sóc phù hợp với giai đoạn đó |
| Knowledge base | Nơi chứa các đoạn tài liệu về nước, phân bón, sâu bệnh, thu hoạch |
| Reranking | Sắp xếp lại kết quả để đoạn đúng giai đoạn đứng cao hơn |

Ví dụ vision model trả về:

```text
Establishment: 5%
Tillering: 75%
Stem/booting: 15%
Reproductive: 3%
Grain filling: 1%
Ripening: 1%
```

Khi đó retrieval sẽ ưu tiên kiến thức liên quan tới:

```text
Tillering - giai đoạn đẻ nhánh
```

Tuy nhiên hệ thống vẫn có thể xét một phần nội dung của giai đoạn liền kề như `Stem/booting`, vì model chưa chắc chắn tuyệt đối.

## Retrieval tìm những nội dung gì?

Knowledge base hiện chia theo các topic:

```text
water_management
nutrient_management
pest_risk
disease_risk
weed_management
harvest_readiness
residue_management
climate_adaptation
general_crop_care
```

Ví dụ cây được xác định là giai đoạn đẻ nhánh, hệ thống có thể chạy lần lượt:

```text
Tìm hướng dẫn quản lý nước cho lúa giai đoạn đẻ nhánh
Tìm hướng dẫn dinh dưỡng cho lúa giai đoạn đẻ nhánh
Tìm nguy cơ sâu hại ở giai đoạn đẻ nhánh
Tìm nguy cơ bệnh hại ở giai đoạn đẻ nhánh
```

Đây là các truy vấn cố định do hệ thống tạo, không cần người dùng nhập câu hỏi tự do.

## Retrieval hoạt động như thế nào?

### 1. Lọc theo topic

Khi cần `water_management`, hệ thống ưu tiên các chunk có topic:

```text
water_management
```

Nó không nên tìm toàn bộ knowledge base nếu topic đã được xác định.

### 2. Tìm theo từ khóa bằng BM25

BM25 tìm những đoạn có từ gần với truy vấn, ví dụ:

```text
quản lý nước
đẻ nhánh
rút nước
mực nước
tưới
```

### 3. Tìm theo ngữ nghĩa bằng dense retrieval

Dense retrieval dùng embedding để tìm những đoạn có ý nghĩa tương tự, ngay cả khi không dùng chính xác cùng từ.

Ví dụ truy vấn nói:

```text
duy trì độ ẩm ruộng khi lúa đẻ nhánh
```

vẫn có thể tìm được chunk viết:

```text
chỉ đưa nước vào ruộng khi mực nước xuống dưới mặt đất...
```

### 4. Gộp BM25 và dense bằng RRF

Hai danh sách kết quả được gộp lại bằng Reciprocal Rank Fusion.

### 5. Rerank theo giai đoạn

Mỗi chunk có vector:

```text
[
  establishment,
  tillering,
  stem_booting,
  reproductive,
  grain_filling,
  ripening
]
```

Ví dụ:

```json
[0.3, 1.0, 0.6, 0.0, 0.0, 0.0]
```

Nghĩa là chunk này:

- rất phù hợp với Tillering;
- có thể liên quan một phần đến Stem/booting;
- ít phù hợp với Establishment;
- không phù hợp với các giai đoạn sau.

Code hiện tính điểm dựa trên:

```text
độ liên quan nội dung
+ độ phù hợp với giai đoạn
+ độ uy tín của nguồn
```

Retriever trong repo đã có BM25, dense retrieval, topic filter, RRF và stage-aware reranking.

## Retrieval được sử dụng ở đâu trong dự án?

### Trong ứng dụng thực tế

Sau khi người dùng tải ảnh:

```text
1. Backend nhận ảnh.
2. Vision model dự đoán stage.
3. Backend lấy probability của 6 stage.
4. Backend chọn các topic cần tìm.
5. Retrieval đọc knowledge base.
6. Hệ thống trả về các đoạn kiến thức phù hợp.
7. Giao diện hiển thị hướng dẫn và nguồn tham khảo.
```

Ví dụ kết quả hiển thị:

```text
Giai đoạn dự đoán: Đẻ nhánh
Độ tin cậy: 75%

Quản lý nước:
- Giữ ruộng đủ ẩm.
- Thực hiện rút nước theo lịch phù hợp.
- Tránh duy trì ngập sâu liên tục.

Nguồn:
Sổ tay hướng dẫn quy trình kỹ thuật sản xuất lúa...
Trang: 35
```

### Trong source code

Chạy retrieval bằng:

```bash
PYTHONPATH=src python scripts/run_retrieval.py \
  --corpus rice_knowledge_nonrestricted.jsonl \
  --topic water_management \
  --stage tillering \
  --mode research \
  --top-k 5
```

Script này:

```text
đọc knowledge base
-> tạo query
-> chạy BM25 và dense retrieval
-> gộp kết quả
-> rerank theo stage
-> trả top-k chunks
```

### Trong thí nghiệm của paper

Retrieval được dùng chủ yếu trong hai thí nghiệm.

Experiment 1 - Oracle-stage retrieval:

```text
Dùng giai đoạn đúng đã được gán nhãn để kiểm tra:
Biết đúng stage có giúp retrieval tốt hơn không?
```

So sánh:

```text
Ungated retrieval
Hard stage filter
Fixed soft gating
Adaptive soft gating
Oracle stage
```

Experiment 2 - End-to-end:

```text
Ảnh
-> stage prediction
-> retrieval
-> đánh giá kết quả cuối
```

Các metric gồm:

```text
P@k
R@k
nDCG@k
SIRR@k
```

Script evaluation hiện hỗ trợ ungated, hard top-1, fixed-soft, adaptive-soft và oracle.

## Ví dụ đầy đủ

Người dùng gửi ảnh lúa.

Vision model trả về:

```json
{
  "tillering": 0.72,
  "stem_booting": 0.20,
  "establishment": 0.05,
  "reproductive": 0.02,
  "grain_filling": 0.01,
  "ripening": 0.00
}
```

Hệ thống chạy retrieval cho topic:

```text
water_management
```

Query tự động:

```text
Bằng chứng quản lý nước phù hợp với lúa
ở giai đoạn đẻ nhánh, BBCH 20-29
```

Retriever tìm được 20 chunk, sau đó rerank.

Kết quả cuối:

```text
Top 1: Quản lý nước ở giai đoạn đẻ nhánh
Top 2: Tưới ướt khô xen kẽ trong thời kỳ sinh trưởng
Top 3: Rút nước giữa vụ
Top 4: Quản lý nước trước làm đòng
Top 5: Hướng dẫn quản lý nước chung
```

Chunk về thu hoạch hoặc chín vàng sẽ bị đẩy xuống vì không phù hợp với stage hiện tại.

## Tóm lại

```text
Vision model trả lời:
"Cây lúa đang ở giai đoạn nào?"

Retrieval trả lời:
"Ở giai đoạn này, tài liệu nào phù hợp nhất để chăm sóc cây?"
```

Không có retrieval thì hệ thống chỉ nhận dạng được stage, nhưng không thể tìm được kiến thức canh tác phù hợp từ knowledge base.
