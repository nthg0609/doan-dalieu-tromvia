import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import uuid
import datetime
import requests
import urllib.request
import pandas as pd
import warnings
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from streamlit_gsheets import GSheetsConnection
import cloudinary
import cloudinary.uploader

warnings.filterwarnings("ignore")

# =================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐÁM MÂY
# =================================================================

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name = "dq7whcy51",
    api_key = "677482925994952",
    api_secret = "1WYJ_fYnUu_nNhgDqLfRCVSAr1Q"
)

# Kết nối Google Sheets
conn = st.connection("gsheets", type=GSheetsConnection)

# =================================================================
# 2. ĐỊNH NGHĨA CÁC LỚP MÔ HÌNH (GỘP CBAM & EFFICIENTNET)
# =================================================================

class HybridSegmentation(nn.Module):
    def __init__(self, unet, deeplab):
        super().__init__()
        self.unet, self.deeplab = unet, deeplab
    def forward(self, x):
        with torch.no_grad():
            p1 = torch.sigmoid(self.unet(x))
            p2 = torch.sigmoid(self.deeplab(x))
            return torch.max(p1, p2)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sa(torch.cat([avg_out, max_out], dim=1))

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = CBAM(self.feature_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone.forward_features(x)
        return self.classifier(self.attention(x))

# =================================================================
# 3. HÀM TẢI & NẠP MÔ HÌNH (GỘP CHUNG CHO AN TOÀN)
# =================================================================

# =================================================================
# 4. TẢI VÀ NẠP MÔ HÌNH (SỬ DỤNG FILES_TO_DOWNLOAD TỪ HUGGING FACE)
# =================================================================

@st.cache_resource
def load_all_models():
    import segmentation_models_pytorch as smp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- BƯỚC 1: DANH SÁCH FILES TẢI TỪ HUGGING FACE ---
    FILES_TO_DOWNLOAD = {
        "unet_best.pth": "https://huggingface.co/nthg0609/doan-dalieu/resolve/main/unet_best.pth",
        "deeplabv3plus_best.pth": "https://huggingface.co/nthg0609/doan-dalieu/resolve/main/deeplabv3plus_best.pth",
        "efficientnet_attention_best.pth": "https://huggingface.co/nthg0609/doan-dalieu/resolve/main/efficientnet_attention_best.pth",
        "hybrid_best.pth": "https://huggingface.co/nthg0609/doan-dalieu/resolve/main/hybrid_best.pth"
    }

    # --- BƯỚC 2: KIỂM TRA VÀ TẢI FILE ---
    for filename, url in FILES_TO_DOWNLOAD.items():
        # Nếu file chưa có HOẶC file bị hỏng (dung lượng quá nhỏ < 1MB)
        if not os.path.exists(filename) or os.path.getsize(filename) < 1000000:
            with st.spinner(f"Đang tải {filename} từ Hugging Face..."):
                try:
                    urllib.request.urlretrieve(url, filename)
                    st.success(f"✅ Đã tải xong {filename}")
                except Exception as e:
                    st.error(f"❌ Lỗi tải {filename}: {e}")
                    st.stop()

    # --- BƯỚC 3: NẠP MÔ HÌNH VÀO RAM ---
    try:
        # 1. Load Segmentation
        u_net = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)
        u_net.load_state_dict(torch.load("unet_best.pth", map_location=device, weights_only=False)["model_state_dict"])
        
        d_lab = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=3, classes=1).to(device)
        d_lab.load_state_dict(torch.load("deeplabv3plus_best.pth", map_location=device, weights_only=False)["model_state_dict"])
        
        # Tạo mô hình Hybrid từ 2 core trên
        hybrid = HybridSegmentation(u_net, d_lab).to(device).eval()

        # 2. Load Classification (EfficientNet + Attention)
        with open("06_classification_complete.json", "r") as f: 
            cls_ckpt = json.load(f)
        num_classes = cls_ckpt["config"]["num_classes"]
        
        cls_model = EfficientNetWithAttention(num_classes).to(device)
        state = torch.load("efficientnet_attention_best.pth", map_location=device, weights_only=False)
        
        # Lấy state dict (xử lý DataParallel nếu có)
        weights = state['model_state_dict'] if 'model_state_dict' in state else state
        new_weights = {k.replace('module.', ''): v for k, v in weights.items()}
        cls_model.load_state_dict(new_weights, strict=False)
        cls_model.eval()
        
        idx_to_class = {v: k for k, v in (state.get("class_to_idx") or cls_ckpt.get("class_to_idx")).items()}
        
        # 3. Tải Font cho PDF (Nếu chưa có)
        os.makedirs("fonts", exist_ok=True)
        f_reg = "fonts/NotoSans-Regular.ttf"
        if not os.path.exists(f_reg):
            urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", f_reg)
            
        return hybrid, cls_model, idx_to_class, device, f_reg
        
    except Exception as e:
        st.error(f"❌ Lỗi nạp mô hình vào RAM: {e}")
        st.write("File trong folder hiện tại:", os.listdir("."))
        st.stop()

# Khởi động nạp toàn bộ hệ thống
hybrid, cls_model, idx_to_class, device, FONT_PATH = load_all_models()
# =================================================================
# 4. LOGIC CHẨN ĐOÁN AI (ĐỒNG BỘ ACCURACY)
# =================================================================

def run_inference(image, patient_name, age, gender, note):
    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h_orig, w_orig = image.shape[:2]
    
    # Step 1: Segmentation
    img_input = cv2.resize(image, (256, 256)).astype(np.float32)/255.0
    img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        mask = hybrid(tensor).squeeze().cpu().numpy()
    
    mask_resized = cv2.resize(mask, (w_orig, h_orig))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    
    # Step 2: ROI Extraction (Smart)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(w_orig, x + w + pad_w), min(h_orig, y + h + pad_h)
        roi = image[y1:y2, x1:x2]
    else: roi = image
    roi = cv2.resize(roi, (224, 224))

    # Step 3: Classification
    roi_input = roi.astype(np.float32)/255.0
    roi_input = (roi_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    roi_t = torch.from_numpy(roi_input).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        probs = torch.softmax(cls_model(roi_t), 1).cpu().numpy()[0]
    label, conf = idx_to_class[np.argmax(probs)], probs[np.argmax(probs)]

    # Step 4: Visualization
    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(cv2.applyColorMap(np.uint8(255*mask_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.3, 0)
    mask_vis = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)

    # Step 5: Cloud Sync
    def up_cv2(img_np, tag):
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        return cloudinary.uploader.upload(buf.tobytes(), folder="skin_app", public_id=f"{record_id}_{tag}")['secure_url']

    with st.spinner("Đang đồng bộ đám mây..."):
        urls = [up_cv2(image, "orig"), up_cv2(overlay, "ov"), up_cv2(mask_vis, "mask")]

    # Step 6: Save GSheets
    data_row = pd.DataFrame([{
        "record_id": record_id, "timestamp": timestamp, "name": patient_name, "age": int(age), 
        "gender": gender, "note": note, "diagnosis": label, "confidence": float(conf),
        "url_orig": urls[0], "url_ov": urls[1], "url_mask": urls[2]
    }])
    try:
        existing = conn.read(worksheet="Sheet1")
        conn.update(worksheet="Sheet1", data=pd.concat([existing, data_row], ignore_index=True))
    except:
        conn.update(worksheet="Sheet1", data=data_row)

    return overlay, mask_vis, label, conf, record_id

# =================================================================
# 5. XUẤT BÁO CÁO PDF (CHUYÊN NGHIỆP 3 ẢNH)
# =================================================================

def export_patient_pdf(record_id):
    try:
        df = conn.read(worksheet="Sheet1")
        r = df[df['record_id'] == record_id].iloc[0]
        W, H = 1240, 1754 # A4
        page = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        f_title = ImageFont.truetype(FONT_PATH, 55)
        f_text = ImageFont.truetype(FONT_PATH, 38)

        draw.text((W//2-380, 100), "BÁO CÁO CHẨN ĐOÁN DA LIỄU AI", fill=(20,60,120), font=f_title)
        draw.line((80, 180, 1160, 180), fill=(200,200,200), width=3)

        y = 250
        lines = [f"ID: {r['record_id']}", f"Bệnh nhân: {r['name']}", f"Tuổi/Giới tính: {r['age']} / {r['gender']}", 
                 f"Chẩn đoán: {r['diagnosis']}", f"Độ tin cậy: {r['confidence']*100:.2f}%", f"Ghi chú: {r['note']}"]
        for line in lines:
            draw.text((100, y), line, fill=(0,0,0), font=f_text)
            y += 75

        # Dàn hàng 3 ảnh
        img_size = 340
        def paste_url(url, x_pos, cap):
            img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
            img.thumbnail((img_size, img_size))
            page.paste(img, (x_pos, 850))
            draw.text((x_pos, 1200), cap, fill=(100,100,100), font=ImageFont.truetype(FONT_PATH, 28))

        paste_url(r['url_orig'], 80, "Ảnh gốc")
        paste_url(r['url_ov'], 450, "Ảnh phân vùng")
        paste_url(r['url_mask'], 820, "Mặt nạ AI")

        buf = BytesIO()
        page.save(buf, format="PDF")
        return buf.getvalue()
    except: return None

# =================================================================
# 6. GIAO DIỆN CHÍNH
# =================================================================

st.set_page_config(page_title="AI Dermatology", layout="wide")
st.title("🩺 Hệ thống Chẩn đoán bệnh da liễu AI")

t1, t2 = st.tabs(["Chẩn đoán mới", "Tra cứu dữ liệu"])

with t1:
    up = st.file_uploader("Tải ảnh", type=["jpg", "png", "jpeg"])
    c1, c2 = st.columns(2)
    name = c1.text_input("Tên bệnh nhân")
    age = c2.number_input("Tuổi", 0, 120, 25)
    gen = c1.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
    note = st.text_area("Ghi chú lâm sàng")
    
    if st.button("Tiến hành AI Analysis"):
        if up and name:
            img = np.array(Image.open(up).convert("RGB"))
            with st.spinner("AI đang phân tích..."):
                ov, mk, lbl, cf, rid = run_inference(img, name, age, gen, note)
                st.image(ov, use_container_width=True)
                st.info(f"**Kết quả:** {lbl} ({cf*100:.2f}%)")
                st.download_button("📥 Tải báo cáo PDF", export_patient_pdf(rid), f"BaoCao_{rid}.pdf")
        else: st.warning("Vui lòng điền đủ thông tin!")

with t2:
    sid = st.text_input("Nhập ID bệnh án để tra cứu")
    if st.button("Tìm kiếm"):
        try:
            df = conn.read(worksheet="Sheet1")
            r = df[df['record_id'] == sid].iloc[0]
            st.image(r['url_ov'], caption="Kết quả chẩn đoán cũ", use_container_width=True)
            st.info(f"Bệnh nhân: {r['name']} | Chẩn đoán: {r['diagnosis']}")
            st.download_button("📥 Tải lại PDF", export_patient_pdf(sid), f"BaoCao_{sid}.pdf")
        except: st.error("Không tìm thấy ID này trên hệ thống.")