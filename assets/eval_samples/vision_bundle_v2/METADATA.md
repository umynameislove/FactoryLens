# Vision Bundle v2 — Metadata

## Nguồn gốc

- Ngày chạy: 2026-07-20T19:57:57+07:00
- Commit hash của repo lúc dựng bank: dc58546c660b2b3dc97584ed9dd4637392fb2006
- Máy / OS / Python version: macOS-15.3.2-arm64-arm-64bit / 3.12.6

## Dữ liệu train

- Số ảnh train/good đã dùng: 391
- Nguồn: MVTec AD hazelnut train/good
- Có lẫn ảnh test không: KHÔNG (đã kiểm bằng MD5 toàn bộ train/test)

## Tham số dựng bank

- Backbone: ResNet18
- Layer: layer2 + layer3
- Embedding dim: 384
- Coreset size: 1024
- Resize ảnh đầu vào: 512 × 512
- Normalize: ImageNet mean/std; L2-normalized patch embeddings

## Kết quả bank

- memory_bank shape: (1024, 384)
- memory_bank SHA-256: 66aeba456f2dc90929bf0da507055501530807bd09003c455ab13685f1450735
- distance_scale: 0.51876783

## Ngưỡng

- anomaly_threshold ĐỀ XUẤT: 0.31051757
- Hiệu chỉnh trên bank nào: chính bank trong bundle này (SHA-256 ở trên)
- Tiêu chí chọn: youden (đánh đổi cân bằng sensitivity/specificity cho demo)
- Ngưỡng này nằm trong dải điểm: min 0.29522151 ≤ threshold ≤ max 0.33707439

## Số đo trên 110 ảnh test

- AUROC: 0.96178571
- Dải điểm (min – max): 0.29522151 – 0.33707439
- FP trên good: 4/40
- FN trên defect: 6/70
- Precision / Recall / F1: 0.94117647 / 0.91428571 / 0.92753623

## So với baseline v1

- AUROC v1 = 0.8304 → v2 = 0.96178571
- FP v1 (tốt nhất ép được) = 9/40 → v2 = 4/40
