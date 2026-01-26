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
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from streamlit_gsheets import GSheetsConnection
import cloudinary
import cloudinary.uploader

# =================================================================
# CẤU HÌNH & LOAD MODEL (KHÔNG ĐỔI KIẾN TRÚC)
# =================================================================
cloudinary.config(cloud_name="dq7whcy51", api_key="677482925994952", api_secret="1WYJ_fYnUu_nNhgDqLfRCVSAr1Q")
conn = st.connection("gsheets", type=GSheetsConnection)

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
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False), nn.ReLU(), nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
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
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(self.feature_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    def forward(self, x):
        return self.classifier(self.attention(self.backbone.forward_features(x)))

@st.cache_resource
def load_all_models():
    import segmentation_models_pytorch as smp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HF_BASE = "https://huggingface.co/nthg0609/doan-dalieu/resolve/main"
    FILES = {"unet_best.pth": f"{HF_BASE}/unet_best.pth", "deeplabv3plus_best.pth": f"{HF_BASE}/deeplabv3plus_best.pth", "efficientnet_attention_best.pth": f"{HF_BASE}/efficientnet_attention_best.pth"}
    
    for name, url in FILES.items():
        if not os.path.exists(name) or os.path.getsize(name) < 1000:
            urllib.request.urlretrieve(url, name)

    u = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    u.load_state_dict(torch.load("unet_best.pth", map_location=device)["model_state_dict"])
    d = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=3, classes=1).to(device)
    d.load_state_dict(torch.load("deeplabv3plus_best.pth", map_location=device)["model_state_dict"])
    hy = HybridSegmentation(u, d).to(device).eval()

    with open("06_classification_complete.json", "r") as f: ck = json.load(f)
    cls = EfficientNetWithAttention(ck["config"]["num_classes"]).to(device)
    st_d = torch.load("efficientnet_attention_best.pth", map_location=device)
    w = st_d['model_state_dict'] if 'model_state_dict' in st_d else st_d
    cls.load_state_dict({k.replace('module.', ''): v for k, v in w.items()}, strict=False)
    cls.eval()
    
    idx_to_cls = {v: k for k, v in (st_d.get("class_to_idx") or ck.get("class_to_idx")).items()}
    os.makedirs("fonts", exist_ok=True)
    f_reg = "fonts/NotoSans-Regular.ttf"
    if not os.path.exists(f_reg): urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", f_reg)
    return hy, cls, idx_to_cls, device, f_reg

hy, cls_m, idx_cls, dev, F_PATH = load_all_models()

def run_inference(image, name, age, gen, note):
    rid = str(uuid.uuid4())[:8]
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Segmentation
    inp = (cv2.resize(image, (256, 256)).astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(dev).float()
    with torch.no_grad(): m = hy(t).squeeze().cpu().numpy()
    m_r = cv2.resize(m, (image.shape[1], image.shape[0]))
    
    # ROI CŨ (Dùng lại logic gốc của mày để tăng Accuracy)
    ys, xs = np.where(m_r > 0.5)
    if len(xs) > 0:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        p = 30
        roi = image[max(0,y1-p):min(image.shape[0],y2+p), max(0,x1-p):min(image.shape[1],x2+p)]
    else: roi = image
    roi_t = torch.from_numpy((cv2.resize(roi, (224, 224)).astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).permute(2,0,1).unsqueeze(0).to(dev).float()
    
    with torch.no_grad(): pr = torch.softmax(cls_m(roi_t), 1).cpu().numpy()[0]
    lbl, conf = idx_cls[np.argmax(pr)], pr[np.argmax(pr)]

    # Upload & Save
    def up(img, t): return cloudinary.uploader.upload(cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes(), folder="skin_app")['secure_url']
    ov = cv2.addWeighted(image, 0.7, cv2.cvtColor(cv2.applyColorMap(np.uint8(255*m_r), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.3, 0)
    urls = [up(image, "o"), up(ov, "v"), up(cv2.cvtColor((m_r>0.5).astype(np.uint8)*255, cv2.COLOR_GRAY2RGB), "m")]
    
    # FIX LỖI PUBLIC SPREADSHEET 
    # 4. Save Sheets (SỬA LẠI ĐOẠN NÀY)
    new_row = pd.DataFrame([{"record_id": rid, "timestamp": ts, "name": name, "age": int(age), "gender": gen, "note": note, "diagnosis": lbl, "confidence": float(conf), "url_orig": urls[0], "url_ov": urls[1], "url_mask": urls[2]}])
    
    # --- BẮT ĐẦU ĐOẠN SỬA ---
    try:
        # Đọc dữ liệu cũ về (ttl=0 để không bị cache)
        existing_data = conn.read(worksheet="Sheet1", ttl=0)
        # Nối dòng mới vào
        updated_df = pd.concat([existing_data, new_row], ignore_index=True)
        # Ghi đè lại toàn bộ (Luôn thành công nếu có quyền Editor)
        conn.update(worksheet="Sheet1", data=updated_df)
    except Exception as e:
        # Trường hợp Sheet đang trắng tinh chưa có gì thì ghi dòng đầu tiên vào
        conn.update(worksheet="Sheet1", data=new_row)
    # --- KẾT THÚC ĐOẠN SỬA ---

    # conn.create(data=new_row)  <-- XÓA DÒNG NÀY ĐI NHÉ
    
    return ov, lbl, float(conf), rid

# PDF & UI giữ nguyên logic nhưng bọc Try-Except chặt hơn
def export_pdf(rid):
    try:
        df = conn.read()
        r = df[df['record_id'].astype(str) == str(rid)].iloc[0]
        page = Image.new("RGB", (1240, 1754), (255,255,255))
        draw = ImageDraw.Draw(page)
        f = ImageFont.truetype(F_PATH, 40)
        draw.text((400, 100), "BAO CAO CHAN DOAN", fill=(0,0,0), font=f)
        draw.text((100, 300), f"ID: {r['record_id']}\nTen: {r['name']}\nChan doan: {r['diagnosis']}\nConf: {r['confidence']}", fill=(0,0,0), font=f)
        buf = BytesIO()
        page.save(buf, format="PDF")
        return buf.getvalue()
    except Exception as e:
        st.error(f"Lỗi PDF: {e}")
        return None

st.set_page_config(page_title="Skin AI")
st.title("🩺 AI Dermatology")
up_file = st.file_uploader("Ảnh", type=["jpg","png"])
name = st.text_input("Tên")
age = st.number_input("Tuổi", 0, 100, 25)
gen = st.radio("Giới tính", ["Nam", "Nữ"])
note = st.text_area("Ghi chú")

if st.button("Chẩn đoán"):
    if up_file and name:
        img = np.array(Image.open(up_file).convert("RGB"))
        ov, lbl, conf, rid = run_inference(img, name, age, gen, note)
        st.image(ov)
        st.success(f"Kết quả: {lbl} ({conf*100:.2f}%)")
        pdf = export_pdf(rid)
        if pdf: st.download_button("Tải PDF", pdf, f"BA_{rid}.pdf")