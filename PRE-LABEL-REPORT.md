# Báo cáo thực hành PointPillars — Day 13

Giữ bản đã điền ngoài Git, trong thư mục nhóm private do LC thu. Đây là kiểm tra formative; không ghi điểm của người khác.

## Nhóm và provenance

- Mã nhóm/phòng: Nhóm Ngựa Hí Hí — Phòng C402 Lab Day 13
- Thành viên: xem `TEAMMATES.md` (họ tên/MSSV, vai trò từng lượt).
- Trạng thái: `executed-by-group` (nhóm tự chạy thành công qua `student-bundle.py`).
- Người thực sự chạy; ngày/giờ; hệ máy/architecture: Nguyễn Vũ Quang Minh; 2026-10-01 08:14 UTC (15:14:10 – 15:14:50 giờ Việt Nam); Windows 11 x64, Docker Desktop (Linux container amd64, runtime limits: 4 CPUs, 4GB RAM).
- Image tag và image ID; phiên bản repo:
  - Image tag: `day13-pointpillars:lc-20261001-amd64`
  - Image ID: `sha256:e03983bd922ec29890bf547db8de408402efd82583680b62e671c20da2fd2c82`
  - Phiên bản repo: `0831856d921609312d42c7582c366e5a311bb7b1`
- PCD được cấp / frame_id; nơi được phép chạy; fingerprint nếu LC cấp:
  - File PCD: `demo.pcd` (17,238 điểm; `source_bytes`: 275,808 bytes).
  - `frame_id`: `demo` (trích từ KITTI Vision Benchmark Suite / MMDetection3D demo 000008).
  - SHA256 PCD: `3b5ea3da13e2b19149cab6a8d521c2ca55f2df93f026b5a3f8c273ce70645d60`.
  - Nơi được phép chạy: Thư mục máy local của nhóm, không phát tán ra ngoài.
- Checkpoint: PointPillars KITTI có sẵn trong image:
  - Đường dẫn checkpoint: `/opt/PointPillars/pretrained/epoch_160.pth`
  - SHA256 checkpoint: `482dfcf63b932cc5ccf012b4bbdad52aa51aa33becf87d0a39d61c39b377b5b1`
- Phạm vi: front-window (vùng quan sát phía trước xe); score threshold: `0.3`
- Giả định kênh thứ tư/intensity và nguồn z_ground:
  - Kênh thứ tư (reflectance): Đã bị loại bỏ trong bản PCD (dùng hằng số giả lập adapter, RGB = uint32 0 placeholder).
  - Nguồn z_ground: Ước lượng tự động từ thuật toán tách mặt đường trên đám mây điểm, $z_{ground} \approx 0.075\text{ m}$.

---

## Ba lượt inference thật

Runner chạy đủ ba lượt từ một lệnh `student-bundle.py run --bundle . --out ..\output-nhom`. Thông số lấy trực tiếp từ `summary.csv` của từng lượt:

| Lượt | delta | Pillar XY | Số hộp | mean_z | File JSON/Side/CSV | Quan sát có bằng chứng |
| --- | --- | --- | --- | --- | --- | --- |
| **A** | 0 | 0.16 | 1 | 0.330 | `run-A/boxes-demo-delta-0-voxel-0.16.json`<br>`run-A/side-demo-delta-0-voxel-0.16.png`<br>`run-A/summary.csv` | Chỉ phát hiện đúng **1 hộp duy nhất** (class `vehicles`, score = 0.32). Tâm $z = 0.33\text{ m}$, cao $1.46\text{ m}$, đáy nằm ở $-0.40\text{ m}$ (chìm dưới mặt đất $z_{ground} \approx 0.075\text{ m}$). Do không hạ point cloud ($\delta = 0$), mạng không tìm thấy cụm điểm tại cao độ mặt đường quen thuộc của KITTI (~ -1.73m) nên bỏ sót hầu như toàn bộ vật thể. |
| **B** | 1.73 | 0.16 | 13 | 1.034 | `run-B/boxes-demo-delta-1.73-voxel-0.16.json`<br>`run-B/side-demo-delta-1.73-voxel-0.16.png`<br>`run-B/summary.csv` | **Mốc chuẩn (Baseline)**: Phát hiện **13 hộp** gồm 10 `vehicles`, 2 `pedestrian`, 1 `two-wheels` (score từ 0.32 đến 0.93). Các hộp bám đều mặt đường và phân bố hợp lý theo chiều sâu dọc trục xe trên ảnh Side. |
| **C** | 1.73 | 0.32 | 6 | 1.091 | `run-C/boxes-demo-delta-1.73-voxel-0.32.json`<br>`run-C/side-demo-delta-1.73-voxel-0.32.png`<br>`run-C/summary.csv` | Phát hiện **6 hộp**, nhưng **100% hộp đều bị phân loại là `pedestrian`** (scores: 0.30 đến 0.81), hoàn toàn biến mất các hộp ô tô (`vehicles`). Ô lưới thô gấp đôi làm gộp đặc trưng sai lệch nghiêm trọng so với cấu hình huấn luyện. |

### Phân tích chuyên sâu:

- **A/B: Thay input trước model có khác dịch cùng một hằng số cho output không? Vì sao?**
  - **Khác hoàn toàn.** Thay đổi `delta` là can thiệp trực tiếp vào tọa độ z của đám mây điểm đầu vào trước khi đưa vào mạng:
    $$z_{model} = z_{source} - z_{ground} - \delta$$
  - Khi z thay đổi, quá trình voxelize (chia pillar), trích xuất đặc trưng hình học của PointPillars và so khớp anchor 3D bị biến đổi hoàn toàn. Kết quả là mạng phát hiện ra số lượng đối tượng khác hẳn (Lượt A chỉ ra 1 hộp, Lượt B ra 13 hộp), chứ không đơn thuần là lấy 13 hộp ở lượt B rồi cộng/trừ 1.73m.

- **B/C: Thấy gì khi đổi pillar? Có đủ bằng chứng để nói cấu hình nào tốt hơn không?**
  - Khi tăng kích thước pillar từ 0.16m lên 0.32m (gấp đôi diện tích đáy cột điểm): Số hộp giảm từ 13 xuống còn 6 hộp, và đặc biệt toàn bộ 10 xe ô tô biến mất, thay vào đó model dự đoán 6 hộp người đi bộ (`pedestrian`). Nguyên nhân do mạng pretrained được tối ưu hóa ở độ phân giải lưới 0.16m; khi lưới thô hơn, phân bố đặc trưng điểm trong pillar không còn tương thích.
  - **Chưa đủ cơ sở** để kết luận cấu hình nào tốt hơn trên phương diện tổng quát vì thí nghiệm này chỉ thực hiện trên duy nhất 1 frame tĩnh (`demo.pcd`). Tuy nhiên, kết quả này chứng minh không thể tự ý thay đổi kích thước voxel mà không huấn luyện lại mạng.

- **Giới hạn ROI và góc Side ảnh hưởng cách đọc miss/yaw thế nào?**
  - *Giới hạn ROI (Front Window)*: Mô hình chỉ thực hiện suy luận trên vùng không gian phía trước xe. Các đối tượng thực tế nằm ở hai bên sườn hoặc phía sau xe sẽ hoàn toàn không có hộp; đây là giới hạn phạm vi quét có chủ đích, không phải do model bỏ sót (miss).
  - *Góc nhìn Side (Chiếu cạnh)*: Ảnh Side là hình chiếu trực giao toàn cảnh 2D dọc thân xe, khiến các đối tượng ở các làn đường và khoảng cách ngang (y) khác nhau bị chiếu đè chồng lên nhau. Góc Side chỉ giúp quan sát chiều dài, chiều cao và độ bám mặt đường cục bộ, nhưng **không thể xác định được chiều rộng (y) và góc quay hướng đầu xe (yaw)**; bắt buộc phải kết hợp góc nhìn từ trên xuống (Top/BEV) và góc nhìn thẳng (Front view).

- **JSON nào còn chưa đủ cơ sở để import? Cần kiểm gì tiếp?**
  - **Toàn bộ các file JSON (run-A, run-B, run-C và qc-cases) đều chưa đủ cơ sở và TUYỆT ĐỐI KHÔNG ĐƯỢC IMPORT vào CVAT.**
  - Lý do: Đây chỉ là kết quả của model chạy thử trên 1 frame demo KITTI để học cách phân biệt lỗi pipeline, không phải nhãn thật của bộ dữ liệu Robotaxi VinFast.
  - Trong bài làm thật trên CVAT, cần mở từng frame bài nguồn (`-source`), đối chiếu point cloud với ảnh camera thật để kiểm tra 5 nhãn, xác định chính xác footprint, đáy hộp bám mặt đường cục bộ và hướng đầu xe (heading).

---

## Ca QC có kiểm soát — không import CVAT

| Ca | Số hộp lệch z / tổng hộp | Lượng lệch | Class/x/y/yaw có đổi? | Dừng batch, kiểm từng hộp hay chưa rõ? | Bằng chứng |
| --- | --- | --- | --- | --- | --- |
| **case-correct** | 0 / 13 | 0 m | Không đổi | **Kiểm từng hộp** (đối chiếu chuẩn) | File JSON và ảnh `side-correct.png` trùng khớp 100% với baseline run-B. Dùng làm mốc tham chiếu hình học. |
| **case-batch-z** | 13 / 13 (100%) | -1.805 m | Không đổi (chỉ lệch trục z) | **DỪNG BATCH, BÁO NGAY CHO LC** | Toàn bộ 13/13 hộp đồng loạt chìm sâu xuống dưới mặt đất đúng 1.805m ($= \delta + z_{ground} = 1.73 + 0.075$). Đây là lỗi hệ thống do pipeline quên thực hiện bước nghịch đảo cao độ ($z_{source} = z_{model} + z_{ground} + \delta$). **Không được sửa tay từng hộp**. |
| **case-one-box-z** | 1 / 13 | -1.805 m (chỉ đúng hộp xe đầu tiên; 12 hộp khác lệch 0m) | Không đổi | **Kiểm từng hộp**, sửa riêng hộp bị lệch | Chỉ duy nhất 1 hộp bị chìm xuống dưới mặt đất, 12 hộp còn lại nằm đúng vị trí. Đây là lỗi cục bộ của một đối tượng cụ thể; kiểm tra đa góc nhìn và sửa riêng hộp đó. |

*Ghi chú:* Helper script tạo biến đổi có chủ đích từ prediction run-B, không phải kết quả inference riêng biệt hoặc nhãn chân thực (ground truth).

---

## Nhận xét cá nhân

### Nguyễn Vũ Quang Minh — Người vận hành hệ thống (Chạy lệnh Terminal/Docker):
- **Vai trò đã làm:** Phụ trách nạp image Docker, kiểm tra tương thích kiến trúc CPU/RAM, điều phối lệnh chạy `student-bundle.py` tuần tự 3 lượt A/B/C và sinh bộ ca lỗi QC.
- **Quan sát A/B/C:** Nhận thấy lượt B chạy ổn định với 13 hộp bám mặt đường đều đặn, trong khi lượt A gần như không nhận diện được gì do mặt sàn bị đẩy lên cao, và lượt C chạy rất nhanh (3.29s so với 5.76s của B) nhưng mất toàn bộ xe con.
- **Diễn giải phép biến đổi z:** Trước khi vào model, cần hạ điểm theo $z_{model} = z_{source} - z_{ground} - \delta$ để đưa mặt đường về $-1.73\text{ m}$ cho khớp KITTI; sau khi model xuất hộp, phải bù ngược lại $z_{source} = z_{model} + z_{ground} + \delta$ để đưa hộp về tọa độ thực tế của xe.
- **Quyết định lỗi batch:** Khi gặp trường hợp như `case-batch-z` (toàn bộ 13 hộp cùng tụt -1.805m), quyết định kiên quyết là **dừng toàn bộ việc gán nhãn thủ công và báo ngay cho kỹ sư pipeline/LC**. Việc cố tình dùng tay kéo từng hộp sẽ phá vỡ tính nhất quán của dữ liệu.
- **Điều còn chưa chắc:** Chưa rõ cách thuật toán tự động tính $z_{ground} = 0.075\text{ m}$ xử lý ra sao trên các đoạn đường có độ dốc lớn hoặc gồ ghề.

---

### Phạm Nguyễn Tuân — Người phân tích số liệu (Đọc tham số & Bảng CSV):
- **Vai trò đã làm:** Trích xuất và đối chiếu các thông số từ file `smoke.json`, `summary.csv`, kiểm tra độ tự tin (score) và giá trị cao độ trung bình `mean_z` qua các lượt chạy.
- **Quan sát A/B/C:** Nhận thấy `mean_z` của lượt A chỉ là 0.330m với 1 hộp duy nhất (score 0.32), trong khi lượt B đạt `mean_z = 1.034m` với 13 hộp phân bố score từ 0.32 đến 0.93. Đáng chú ý ở lượt C, dù `mean_z` xấp xỉ B (1.091m) nhưng bản chất phân bố nhãn bị lật ngược hoàn toàn (chỉ còn lại `pedestrian`).
- **Diễn giải phép biến đổi z:** Hiểu rõ $\delta = 1.73\text{ m}$ là độ cao đặt cảm biến chuẩn của KITTI. Nếu thiếu bước chuyển đổi này thì đặc trưng hình học mà mô hình học được từ tập KITTI sẽ hoàn toàn không thể kích hoạt chính xác trên dữ liệu mới.
- **Quyết định lỗi batch:** Nhận định rằng lỗi lệch đồng loạt cả batch thể hiện sự sai khác ở cấp độ ma trận biến đổi tọa độ hoặc cấu hình tham số tiền xử lý. Hành động đúng đắn duy nhất là dừng sửa và yêu cầu fix từ pipeline.
- **Điều còn chưa chắc:** Ngưỡng lọc score cố định ở mức 0.3 có thể làm lọt một số đối tượng ở xa bị điểm thưa hay không.

---

### Đinh Công Minh — Người phân tích hình học (Soi ảnh Side & Không gian 3D):
- **Vai trò đã làm:** Mở và so sánh trực quan các ảnh chiếu cạnh `side-*.png` của các lượt chạy A, B, C và 3 ca lỗi mẫu; phân tích tương quan không gian giữa các hộp và cụm điểm point cloud.
- **Quan sát A/B/C:** Trên ảnh `side-demo-delta-0-voxel-0.16.png` và file `run-A/boxes-demo-delta-0-voxel-0.16.json`, hộp duy nhất có tâm $z = 0.33\text{ m}$ và chiều cao $h = 1.46\text{ m}$, do đó đáy hộp nằm ở cao độ $z_{bottom} = 0.33 - 1.46/2 = -0.40\text{ m}$. So với mặt đường ước lượng $z_{ground} \approx 0.075\text{ m}$, hộp này thực tế nằm thấp/chìm nhẹ dưới mặt đất do chưa được bù delta đúng, chứ không phải lơ lửng trên không. Ở lượt B, 13 hộp bám đều mặt đường và phân bố hợp lý theo chiều sâu dọc trục xe. Ở lượt C, ô pillar lớn làm mất hết xe con, chỉ còn các hộp hẹp của người đi bộ.
- **Diễn giải phép biến đổi z:** Phép biến đổi thuận đưa cụm điểm về hệ quy chiếu chuẩn của model, phép nghịch đảo trả hộp về đúng tọa độ cảm biến của xe. Nếu phép nghịch đảo bị bỏ sót, đáy hộp sẽ cắm sâu dưới lòng đất như trong `case-batch-z`.
- **Quyết định lỗi batch:** Khi quan sát ảnh `side-batch-z.png`, thấy toàn bộ các hộp đều có khoảng cách đáy hộp so với mặt đường lệch một khoảng bằng nhau $\rightarrow$ Đây là signature của lỗi hệ thống, tuyệt đối không chỉnh thủ công.
- **Điều còn chưa chắc:** Trên ảnh Side 2D, các xe ở làn bên cạnh bị chiếu đè lên xe ở làn giữa nên rất khó khẳng định kích thước chiều dài hộp đã chuẩn xác hay chưa nếu không có góc nhìn Top-down.

---

### Nguyễn Đình Độ — Người ghi chép biên bản (Lập báo cáo & Tổng hợp):
- **Vai trò đã làm:** Ghi chép nhật ký thực hành, tổng hợp các số liệu đo lường, biên soạn báo cáo `PRE-LABEL-REPORT.md` và phân bổ vai trò trong `TEAMMATES.md`.
- **Quan sát A/B/C:** Đối chiếu trực tiếp giữa các file tổng kết: `run-A/summary.csv` chỉ ghi nhận `n_boxes = 1`, `mean_z = 0.330 m`; trong khi sang `run-B/summary.csv` ($\delta = 1.73\text{ m}$), số hộp tăng vọt lên `n_boxes = 13`, `mean_z = 1.034 m` với 10 `vehicles`, 2 `pedestrian`, 1 `two-wheels` (theo `run-B/boxes-demo-delta-1.73-voxel-0.16.json`). Ở `run-C/summary.csv` (`voxel_size = 0.32 m`), số hộp giảm còn `n_boxes = 6`, `mean_z = 1.091 m` và file `run-C/boxes-demo-delta-1.73-voxel-0.32.json` cho thấy 100% hộp bị biến thành `pedestrian`. Điều này minh chứng việc thay đổi biểu diễn đầu vào làm thay đổi căn bản kết quả trích xuất đặc trưng của mạng nơ-ron.
- **Diễn giải phép biến đổi z:** Cảm biến LiDAR gắn trên nóc xe có cao độ cách mặt đất khoảng 1.73m. Mô hình PointPillars huấn luyện trên KITTI mặc định mặt đường nằm ở khoảng $-1.73\text{ m}$. Vì vậy, trước khi đưa điểm vào mô hình, ta phải trừ cao độ mặt đất và trừ tiếp $\delta = 1.73\text{ m}$ ($z_{model} = z_{source} - z_{ground} - \delta$) để "hạ" toàn bộ đám mây điểm xuống hệ quy chiếu mà mô hình đã học. Sau khi mô hình suy luận ra tọa độ hộp 3D, ta bắt buộc phải cộng bù ngược lại ($z_{source} = z_{model} + z_{ground} + \delta$) để đưa hộp trở lại hệ tọa độ thực tế của xe tự hành ban đầu.
- **Quyết định lỗi batch:** Đồng thuận với cả nhóm rằng lỗi batch phải được xử lý ở tầng kỹ thuật hệ thống (pipeline), việc người gán nhãn cố tình can thiệp bằng tay sẽ làm sai lệch cơ sở dữ liệu huấn luyện sau này.
- **Điều còn chưa chắc:** Cần được hướng dẫn thêm cách nhận biết góc xoay đầu xe (yaw/heading) khi xe bị che khuất phần đầu trên dữ liệu thực tế ở Phần 2 (CVAT).

---

## LC ghi nhận riêng

> LC ghi nhận ngày 01/10/2026. **Kết luận: ĐẠT.**

- **Quyền dùng PCD/image và đúng ca:** Gói Student KITTI 000008 (giấy phép CC BY-NC-SA 3.0), không dùng dữ liệu Robotaxi. Input SHA-256 `3b5ea3da…` và image `sha256:e03983bd…` (amd64) khớp `smoke.json`.
- **Có chạy thật / chỉ phân tích; còn cần lượt thực hành bổ sung:** Chạy thật (Windows 11, Docker Desktop). `smoke.json` passed: nạp image 22,1 s, A/B/C 6,1 / 5,8 / 3,3 s, kết quả 1/13/6. Báo cáo ghi "08:14 UTC = 15:14 giờ Việt Nam".
- **Output đủ, giữ bản gốc, không đưa ca lỗi vào CVAT:** Đủ `run-A/B/C` và `qc-cases`. Không sửa JSON. Không đưa ca lỗi vào CVAT.
- **Nhận xét từng thành viên và quyết định dừng pipeline:** Đủ 4 người. Câu import đúng và rõ nhất lớp (KITTI demo, tuyệt đối không import). Nhận xét giải thích đúng phép thuận đưa mặt đường về −1,73 m. Đã đính chính nhận xét hộp A ở −0,40 m (nằm thấp); người ghi chép đã dẫn file/số liệu và diễn giải phép z chi tiết.
- **Đồng ý chuyển sang chỉnh/QC / cần bổ sung; lý do:** **Đồng ý chuyển sang chỉnh/QC.**
