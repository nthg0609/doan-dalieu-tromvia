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
import textwrap
import re
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
# 2. ĐỊNH NGHĨA CÁC LỚP MÔ HÌNH
# =================================================================

class HybridSegmentation(nn.Module):
    def __init__(self, unet, deeplab):
        super().__init__()
        self.unet, self.deeplab = unet, deeplab
    def forward(self, x):
        with torch.no_grad():
            return torch.max(torch.sigmoid(self.unet(x)), torch.sigmoid(self.deeplab(x)))

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
        return x * self.sa(torch.cat([torch.mean(x,1,True), torch.max(x,1,True)[0]], 1))

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
# 3. HÀM TẢI MÔ HÌNH VÀ TÀI NGUYÊN
# =================================================================

@st.cache_resource
def load_all_models():
    import segmentation_models_pytorch as smp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    u_net = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    u_net.load_state_dict(torch.load("unet_best.pth", map_location=device, weights_only=False)["model_state_dict"])
    
    d_lab = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=3, classes=1).to(device)
    d_lab.load_state_dict(torch.load("deeplabv3plus_best.pth", map_location=device, weights_only=False)["model_state_dict"])
    
    hybrid = HybridSegmentation(u_net, d_lab).to(device).eval()

    with open("06_classification_complete.json", "r") as f: cls_ckpt = json.load(f)
    num_classes = cls_ckpt["config"]["num_classes"]
    cls_model = EfficientNetWithAttention(num_classes).to(device)
    state = torch.load("efficientnet_attention_best.pth", map_location=device, weights_only=False)
    
    # Xử lý tên lớp và load weights
    weights = state['model_state_dict'] if 'model_state_dict' in state else state
    new_weights = {k.replace('module.', ''): v for k, v in weights.items()}
    cls_model.load_state_dict(new_weights, strict=False)
    cls_model.eval()
    
    idx_to_class = {v: k for k, v in (state.get("class_to_idx") or cls_ckpt.get("class_to_idx")).items()}
    
    # Load Fonts cho PDF
    os.makedirs("fonts", exist_ok=True)
    f_reg = "fonts/NotoSans-Regular.ttf"
    if not os.path.exists(f_reg):
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", f_reg)
    
    return hybrid, cls_model, idx_to_class, device, f_reg

hybrid, cls_model, idx_to_class, device, FONT_PATH = load_all_models()

# =================================================================
# 4. LOGIC XỬ LÝ AI
# =================================================================

def run_inference(image, patient_name, age, gender, note):
    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h_orig, w_orig = image.shape[:2]
    
    # 1. AI Process (Segmentation)
    img_input = cv2.resize(image, (256, 256)).astype(np.float32)/255.0
    img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device).float()
    
    with torch.no_grad():
        mask = hybrid(tensor).squeeze().cpu().numpy()
    
    mask_resized = cv2.resize(mask, (w_orig, h_orig))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    
    # --- CẢI TIẾN ROI THÔNG MINH ---
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad_w, pad_h = int(w * 0.15), int(h * 0.15)
        x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
        x2, y2 = min(w_orig, x + w + pad_w), min(h_orig, y + h + pad_h)
        roi = image[y1:y2, x1:x2]
    else:
        roi = image
    roi_resized = cv2.resize(roi, (224, 224))

    # 2. Classification
    roi_t = torch.from_numpy((roi_resized.astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).permute(2,0,1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        probs = torch.softmax(cls_model(roi_t), 1).cpu().numpy()[0]
    label, conf = idx_to_class[np.argmax(probs)], probs[np.argmax(probs)]

    # 3. Cloud Sync
    def upload_cv2(img_np, tag):
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        return cloudinary.uploader.upload(buf.tobytes(), folder="skin_app", public_id=f"{record_id}_{tag}")['secure_url']

    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(cv2.applyColorMap(np.uint8(255*mask_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.3, 0)
    mask_vis = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)
    
    with st.spinner("Đang lưu trữ dữ liệu..."):
        urls = [upload_cv2(image, "orig"), upload_cv2(overlay, "ov"), upload_cv2(mask_vis, "mask")]

    # 4. Save GSheets
    data_row = pd.DataFrame([{
        "record_id": record_id, "timestamp": timestamp, "name": patient_name, "age": age, 
        "gender": gender, "note": note, "diagnosis": label, "confidence": float(conf),
        "url_orig": urls[0], "url_ov": urls[1], "url_mask": urls[2]
    }])
    try:
        existing = conn.read(worksheet="Sheet1")
        conn.update(worksheet="Sheet1", data=pd.concat([existing, data_row], ignore_index=True))
    except:
        conn.update(worksheet="Sheet1", data=data_row)

    info = f"**ID:** {record_id}  \n**Chẩn đoán:** {label} ({conf*100:.2f}%)  \n**Bệnh nhân:** {patient_name}"
    return overlay, mask_vis, info, record_id

# =================================================================
# 5. XUẤT BÁO CÁO PDF (3 ẢNH)
# =================================================================

def export_patient_pdf(record_id):
    try:
        df = conn.read(worksheet="Sheet1")
        r = df[df['record_id'] == record_id].iloc[0]
        
        # A4 DPI 150
        W, H = 1240, 1754
        margin = 80
        page = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(page)
        font = ImageFont.truetype(FONT_PATH, 40)
        font_bold = ImageFont.truetype(FONT_PATH, 55)

        # Header
        draw.text((W//2 - 350, 100), "BÁO CÁO CHẨN ĐOÁN DA LIỄU AI", fill=(20, 60, 120), font=font_bold)
        draw.line((margin, 180, W-margin, 180), fill=(200, 200, 200), width=3)

        # Info
        info_y = 220
        fields = [f"ID Bệnh án: {r['record_id']}", f"Bệnh nhân: {r['name']}", f"Tuổi/Giới tính: {r['age']} / {r['gender']}", 
                  f"Thời gian: {r['timestamp']}", f"Chẩn đoán: {r['diagnosis']}", f"Độ tin cậy: {r['confidence']*100:.2f}%"]
        for f in fields:
            draw.text((margin, info_y), f, fill=(0, 0, 0), font=font)
            info_y += 70

        # Images (3 ảnh: Gốc - Overlay - Mask)
        img_w, img_h = 350, 350
        def paste_url(url, x, y, cap):
            img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
            img.thumbnail((img_w, img_h))
            page.paste(img, (x + (img_w - img.size[0])//2, y))
            draw.text((x + 20, y + img_h + 20), cap, fill=(100, 100, 100), font=ImageFont.truetype(FONT_PATH, 30))

        paste_url(r['url_orig'], margin, 850, "Ảnh tổn thương gốc")
        paste_url(r['url_ov'], margin + 380, 850, "Ảnh phân vùng AI")
        paste_url(r['url_mask'], margin + 760, 850, "Mặt nạ phân đoạn")

        # Footer
        draw.text((W//2 - 200, H - 100), "Hệ thống AI Dermatology", fill=(150, 150, 150), font=font)
        
        pdf_buf = BytesIO()
        page.save(pdf_buf, format="PDF")
        return pdf_buf.getvalue()
    except: return None

# =================================================================
# 6. GIAO DIỆN STREAMLIT
# =================================================================

st.set_page_config(page_title="Skin AI", layout="wide")
st.title("🩺 Chẩn đoán bệnh da liễu AI")

t1, t2 = st.tabs(["Chẩn đoán", "Tra cứu"])

with t1:
    up = st.file_uploader("Tải ảnh", type=["jpg", "png", "jpeg"])
    name = st.text_input("Tên bệnh nhân")
    age = st.number_input("Tuổi", 0, 120, 25)
    gen = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
    note = st.text_area("Ghi chú")
    if st.button("Bắt đầu AI"):
        if up and name:
            img = np.array(Image.open(up).convert("RGB"))
            with st.spinner("AI đang tính toán..."):
                ov, mk, info, rid = run_inference(img, name, age, gen, note)
                st.image(ov, use_container_width=True)
                st.success(info)
                st.download_button("📥 Tải báo cáo PDF", export_patient_pdf(rid), f"BA_{rid}.pdf")

with t2:
    sid = st.text_input("Nhập ID bệnh án")
    if st.button("Xem chi tiết"):
        try:
            df = conn.read(worksheet="Sheet1")
            r = df[df['record_id'] == sid].iloc[0]
            col1, col2, col3 = st.columns(3)
            col1.image(r['url_orig'], caption="Ảnh gốc")
            col2.image(r['url_ov'], caption="AI Overlay")
            col3.image(r['url_mask'], caption="AI Mask")
            st.info(f"Bệnh nhân: {r['name']} | Chẩn đoán: {r['diagnosis']}")
            st.download_button("📥 Tải PDF lại", export_patient_pdf(sid), f"BA_{sid}.pdf")
        except: st.error("Không tìm thấy ID")