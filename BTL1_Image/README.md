=============================================================================================
                    BÁO CÁO CHI TIẾT CÔNG VIỆC VÀ CHỨC NĂNG CÁC MODULE
=============================================================================================

1. FILE `src/dataset.py` (CHỊU TRÁCH NHIỆM CHUẨN BỊ VÀ TĂNG CƯỜNG DỮ LIỆU)
- Class BloodCellDataset: Xây dựng một cấu trúc dữ liệu tùy chỉnh kế thừa từ PyTorch. Nó tự động vào ổ cứng, đọc từng bức ảnh, chuyển đổi đồng loạt sang hệ màu RGB và encode 0-7 tương ứng với tên thư mục chứa ảnh đó.
- Kỹ thuật Stratified Split (Chia những giữ phân phối): Vì dữ liệu tế bào máu bị mất cân bằng nghiêm trọng, việc chia tập Train/Val ngẫu nhiên sẽ làm thiếu hụt các lớp hiếm ở tập Val. Nhóm dùng `stratify` để ép tỷ lệ các loại tế bào ở cả hai tập phải giống hệt nhau.

- Pipeline Tăng cường dữ liệu (Data Augmentation): 
  + Cấu hình "Không Augment": Chỉ ép kích thước về 224x224 và chuẩn hóa màu theo ImageNet. Dùng làm baseline đối chứng.
  + Cấu hình "Có Augment": Tích hợp `RandAugment` - một thuật toán tự động áp dụng các phép biến dạng hình học và làm nhiễu màu sắc phức tạp. Mục đích là tạo ra các mẫu tế bào "khó nhìn" để rèn luyện khả năng chống chịu nhiễu (robustness) của mô hình.
- Tính toán Class Weights: Tự động đếm số lượng ảnh của từng loại và tính ra một bộ "trọng số phạt". Loại tế bào nào càng ít ảnh, trọng số phạt càng cao, giúp hàm mất mát chú ý hơn vào các lớp thiểu số.

2. FILE `src/models.py` 
Nơi chứa các kiến trúc mạng học sâu, áp dụng kỹ thuật Transfer Learning (Học chuyển giao) từ bộ dữ liệu ImageNet để hội tụ nhanh hơn.
- Tích hợp 3 họ kiến trúc đối chứng:
  + ResNet18: Đại diện cho mạng tích chập (CNN) truyền thống, hoạt động cực kỳ ổn định trên ảnh vi thể.
  + ViT-B/16 (Vision Transformer): Đại diện cho cơ chế Attention hiện đại, bẻ ảnh thành các patch nhỏ để phân tích toàn cục. Đây là yêu cầu so sánh bắt buộc của đề tài.
  + MobileNetV3: Đại diện cho mạng siêu nhẹ, có số lượng tham số ít, chuyên dùng để chứng minh sự hiệu quả khi chạy trên thiết bị cấu hình yếu.
- Cơ chế Freeze Backbone: Thêm công tắc `freeze_backbone` để nhóm có thể linh hoạt chọn: (1) Chỉ huấn luyện lớp phân loại cuối cùng (nhanh nhưng điểm thấp) HOẶC (2) Huấn luyện lại toàn bộ mạng (lâu nhưng điểm cao).

3. FILE `src/losses.py` 
Nơi định nghĩa các thước đo hình phạt khi mô hình đoán sai.
- Tích hợp Focal Loss: Khắc phục điểm yếu của CrossEntropy. Focal Loss có một tham số (gamma) giúp mô hình tự động "lờ đi" các bức ảnh bạch cầu dễ nhận diện, và dồn toàn bộ sự chú ý vào các bức ảnh bị đoán sai nhiều lần hoặc thuộc nhóm tế bào hiếm.

4. FILE `src/train.py` 
Nơi quản lý vòng lặp huấn luyện, tối ưu hóa và ghi nhận kết quả.
- Tốc độ học theo tầng (Layer-wise Learning Rate): Thay vì dùng 1 tốc độ học chung chung, nhóm thiết lập để phần móng (Backbone) học rất chậm nhằm giữ lại kiến thức pre-train, còn phần ngọn (Classifier) học nhanh gấp 10 lần để kịp thời làm quen với tế bào máu.
- Scheduler & Early Stopping: 
  + Scheduler: Tự động giảm tốc độ học xuống 10 lần sau mỗi vài vòng để mô hình hội tụ mượt mà, không bị "nhảy" khỏi điểm tối ưu.
  + Early Stopping: Thuật toán "Dừng sớm". Cứ sau mỗi vòng, mô hình sẽ thi thử trên tập Validation. Nếu sau 5 vòng liên tiếp điểm không tăng, hệ thống tự động ngắt huấn luyện để tiết kiệm thời gian và chống học vẹt (Overfitting), đồng thời tự động sao lưu file trọng số (`.pth`) tốt nhất.
- Tự động hóa Thực nghiệm: Code được thiết kế thành một vòng lặp tự động chạy 10 kịch bản đối chứng khác nhau (từ Baseline, thay đổi Model, thêm Augment đến Focal Loss). Nhờ đó nhóm có được một bộ dữ liệu so sánh cực kỳ đồ sộ và khách quan.

5. FILE `src/evaluate.py` (CHỊU TRÁCH NHIỆM CHẤM ĐIỂM VÀ PHÂN TÍCH LỖI)
Nơi vắt kiệt giá trị từ các file trọng số để đưa ra các báo cáo định lượng và định tính.
- Đánh giá 5 Metrics toàn diện: Tính toán Accuracy, Precision, Recall và F1-Score. Đặc biệt, dùng tham số `average='weighted'` để điểm số phản ánh đúng hiệu năng trên tập dữ liệu lệch lớp.
- Đo lường độ tin cậy ECE (Expected Calibration Error): Tính toán xem mức độ "tự tin" của mô hình có đi đôi với năng lực thực tế không. ECE thấp nghĩa là mô hình dự đoán rất chắc chắn, không bị ảo tưởng sức mạnh.
- Vẽ Ma trận nhầm lẫn (Confusion Matrix): Tự động xuất biểu đồ nhiệt (Heatmap) để nhóm nhìn thấy ngay mô hình đang hay bị nhầm lẫn loại tế bào A với loại tế bào B nào.
- Trích xuất ảnh phân tích lỗi (Error Analysis): Viết thuật toán gom các bức ảnh bị mô hình đoán sai, in ra một lưới ảnh kèm theo nhãn gốc (True) và nhãn sai (Pred). Cực kỳ hữu ích để đưa vào báo cáo chứng minh sự hiểu biết về dữ liệu.

6. FILE `src/efficiency.py` (CHỊU TRÁCH NHIỆM ĐÁNH GIÁ TÀI NGUYÊN PHẦN CỨNG)
Nơi thực hiện các bài test về tính khả thi khi triển khai thực tế.
- Đo lường kích thước: Tự động đếm tổng số lượng tham số (Parameters) và dung lượng file tĩnh (MB) của 3 kiến trúc CNN, ViT và MobileNet.
- Bấm giờ tốc độ suy luận (Inference Time): Chạy thử nghiệm giả lập (kèm Warm-up GPU) để đo chính xác thời gian phản hồi cho một bức ảnh (tính bằng mili-giây).
- Trực quan hóa Trade-off: Vẽ biểu đồ bong bóng (Bubble Chart) kết hợp 3 thông số: Tốc độ suy luận (Trục X), Độ chính xác (Trục Y) và Kích thước mạng (Độ to của bong bóng). Giúp minh họa sinh động việc MobileNet nhỏ gọn, nhanh chóng mà vẫn giữ được Accuracy cao, so với sự cồng kềnh của ViT.

=============================================================================================
                                          CHIẾN LƯỚC TEST
=============================================================================================

Nhóm 1: Đi tìm Chiến lược Fine-tune tối ưu (Test 01, 02, 03)
Mục tiêu: Trả lời câu hỏi "Nên cập nhật trọng số như thế nào khi dùng Transfer Learning?"
Nhóm sử dụng mạng ResNet18 cơ bản (Không Augment, Không Focal Loss) để làm bài test:

Test 01 (Freeze): Đóng băng toàn bộ mạng, chỉ huấn luyện lớp phân loại (Classifier) cuối cùng. Kỳ vọng train rất nhanh nhưng điểm thấp do đặc trưng của ImageNet (vật thể thường) không khớp với ảnh y tế.

Test 02 (Full Fine-tune): Mở khóa và huấn luyện lại toàn bộ mạng với cùng một tốc độ học (Learning Rate). Khả năng cao sẽ phá hỏng các đặc trưng tốt đã được pre-train.

Test 03 (Layer-wise LR): Mở khóa toàn bộ, nhưng các tầng trích xuất đặc trưng bên dưới học rất chậm, tầng phân loại bên trên học nhanh.
👉 Kết luận mong đợi: Test 03 sẽ là chiến lược tốt nhất. Từ Test 04 trở đi, nhóm sẽ dùng mặc định chiến lược Layer-wise.

Nhóm 2: Đánh giá Kiến trúc mạng & Hiệu quả phần cứng (Test 03, 04, 05, 06)
Mục tiêu: Trả lời yêu cầu "So sánh CNN vs ViT" và "Đánh giá sự đánh đổi giữa Kích thước/Tốc độ/Độ chính xác".

Test 04 & 05 (ViT-B/16): Đưa mạng Transformer vào chạy thử nghiệm với cấu hình Freeze và Layer-wise. Nhóm sẽ so sánh Test 05 với Test 03 (ResNet18) để xem CNN hay ViT phân loại tế bào máu tốt hơn.

Test 06 (MobileNetV3): Đưa mạng siêu nhẹ vào huấn luyện. Điểm số của Test 06 sẽ được đem lên bàn cân với Test 03 và Test 05 để chứng minh: MobileNet là lựa chọn tối ưu nhất nếu phải triển khai ứng dụng trên thiết bị y tế cấu hình yếu (dung lượng nhỏ, suy luận nhanh).

Nhóm 3: Đánh giá sức mạnh của Tăng cường dữ liệu (Test 03 vs Test 07)
Mục tiêu: Trả lời yêu cầu "So sánh Có / Không Augmentation".

Nhóm lấy Test 03 (hoàn toàn dùng ảnh gốc) làm hệ quy chiếu.

Test 07: Giữ nguyên cấu hình của Test 03 nhưng bật RandAugment (thêm nhiễu màu, bóp méo hình học mạnh).
👉 Kết luận mong đợi: So sánh đồ thị huấn luyện của 2 test này sẽ cho thấy RandAugment giúp mô hình bớt hiện tượng học vẹt (Overfitting) trên tập Train và có khả năng khái quát hóa tốt hơn trên tập Validation.

Nhóm 4: Đánh giá Hàm mất mát trị mất cân bằng (Test 03 vs Test 08)
Mục tiêu: Trả lời yêu cầu "Xử lý dữ liệu lệch lớp".

Tập dữ liệu có sự chênh lệch lớn giữa các loại bạch cầu. Test 03 đang sử dụng kỹ thuật cân bằng cơ bản là CrossEntropy + Đánh trọng số (Reweighting).

Test 08: Chuyển sang sử dụng Focal Loss.
👉 Kết luận mong đợi: Bằng cách so sánh chỉ số F1-Score và Recall của các lớp thiểu số (như Basophil) giữa Test 03 và Test 08, nhóm sẽ chứng minh được Focal Loss ép mô hình chú ý vào các mẫu khó tốt hơn hẳn phương pháp truyền thống.

Nhóm 5: Cấu hình "Quái vật" & Chuẩn bị cho Ensemble (Test 09, 10)
Mục tiêu: Lắp ráp tất cả các kỹ thuật tốt nhất lại với nhau để tạo ra phiên bản mô hình mạnh nhất.

Test 09 (ResNet_Ultimate): CNN + Layer-wise + RandAugment + Focal Loss.

Test 10 (ViT_Ultimate): ViT + Layer-wise + RandAugment + Focal Loss.
👉 Kết luận mong đợi: Hai mô hình này sẽ đại diện cho sức mạnh đỉnh cao của hai trường phái. Kết quả dự đoán của chúng sẽ được kết hợp lại trong kỹ thuật Ensemble (Soft Voting) để tạo ra một siêu mô hình đánh bại mọi giới hạn Accuracy trước đó. Đồng thời, Test 09 sẽ được dùng để chạy bản đồ nhiệt Grad-CAM giải thích mô hình.

=============================================================================================
                                          ĐÁNH GIÁ KẾT QUẢ
=============================================================================================
1. Phân tích Chiến lược Fine-tune (Nhóm Test 01, 02, 03)
Chiến thuật tách lớp để đánh giá (Freeze vs Full vs Layer-wise) đã cho thấy sự chênh lệch rõ ràng nhất trong toàn bộ bảng:

Sự thất bại của Freeze (Test 01): Việc đóng băng trọng số (chỉ đạt 85.32%) chứng tỏ các đặc trưng hình ảnh được học từ ImageNet (chó, mèo, ô tô) không thể áp dụng trực tiếp lên hình ảnh vi thể của tế bào máu. Hơn nữa, chỉ số ECE rất cao (0.1639) cho thấy mô hình này đang "ảo tưởng sức mạnh" – đoán sai nhưng lại cực kỳ tự tin.

Chiến thắng tuyệt đối của Layer-wise (Test 03): Đạt điểm cao nhất toàn bảng (98.92%). Việc mở khóa toàn bộ mạng nhưng cho các tầng móng học chậm, tầng ngọn học nhanh giúp mô hình vừa giữ được khả năng trích xuất biên độ/màu sắc cơ bản, vừa thích nghi hoàn hảo với tế bào máu. Đồng thời, chỉ số ECE giảm xuống đáy (0.0061), cho thấy dự đoán cực kỳ đáng tin cậy.

2. So sánh Kiến trúc: CNN vs ViT vs MobileNet (Nhóm Test 03, 05, 06)
CNN đánh bại ViT trên dữ liệu y tế: ResNet18 (Test 03 - 98.92%) nhỉnh hơn ViT-B/16 (Test 05 - 98.74%). Điều này hoàn toàn khớp với lý thuyết: Vision Transformer thường cần dữ liệu khổng lồ để phát huy sức mạnh (do thiếu inductive bias), trong khi CNN hoạt động cực kỳ ổn định và chắt lọc đặc trưng tốt trên các tập dữ liệu vừa và nhỏ.

Sự ưu việt của MobileNetV3 (Test 06): Dù là mô hình siêu nhẹ, nó vẫn đạt Accuracy 98.45%. Đặc biệt, MobileNet có chỉ số Calibration ECE thấp nhất toàn bảng (0.0058). Đây là luận điểm cực mạnh để báo cáo: MobileNet là lựa chọn số 1 để triển khai thực tế vì nó nhẹ, nhanh, độ chính xác cao và rất "biết mình biết ta" (tự tin đúng lúc).

3. Nghịch lý của Tăng cường dữ liệu (Test 03 vs Test 07)
Đồ án thường mặc định "thêm Augmentation thì điểm sẽ cao hơn", nhưng bảng kết quả của bạn lại đi ngược lại: Cấu hình RandAugment (Test 07) làm điểm Accuracy giảm từ 98.92% xuống 98.57%.

Phân tích ghi điểm: Giảng viên sẽ cực kỳ thích nếu bạn đưa ra nhận định này: "Đối với dữ liệu y tế vi thể, hình thái học (kích thước nhân, độ nhám của màng tế bào) là yếu tố quyết định. Việc lạm dụng RandAugment (bóp méo hình học, thay đổi độ tương phản quá mạnh) đã phá hủy các cấu trúc sinh học nguyên bản này, khiến mô hình bị bối rối."

4. Hiệu ứng của Focal Loss (Test 03 vs Test 08)
Khi thay Reweighting bằng Focal Loss, điểm tổng thể giảm nhẹ (từ 98.92% xuống 98.74%), và ECE tăng nhẹ (0.0095).

Phân tích: Focal Loss hoạt động bằng cách "ép" mô hình bớt quan tâm đến các lớp dễ đoán (đa số) để tập trung cứu vãn các lớp khó đoán (thiểu số). Do đó, việc điểm tổng thể (chịu ảnh hưởng lớn bởi lớp đa số) bị kéo xuống một chút là sự đánh đổi hoàn toàn bình thường và có thể chấp nhận được để có một mô hình công bằng hơn.

5. Sự sụp đổ của cấu hình "Ultimate" (Test 09, 10)
Dù nhồi nhét tất cả các kỹ thuật xịn nhất (Layer-wise + RandAugment + Focal Loss), Test 09 và 10 lại không phải là những mô hình đứng Top 1.

Bài học rút ra: Việc lạm dụng quá nhiều cơ chế Regularization (chống học vẹt) cùng lúc đã khiến mô hình bị Underfitting (học không tới).