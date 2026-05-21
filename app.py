# simpal future extrect

# import streamlit as st
# import cv2
# import numpy as np
# import joblib
#
# # -----------------------------
# # LOAD MODELS
# # -----------------------------
# knn_model = joblib.load("sift_knn_model.pkl")
# kmeans = joblib.load("sift_kmeans.pkl")
# class_names = joblib.load("class_names.pkl")
# scaler = joblib.load("sift_scaler.pkl")
#
# sift = cv2.SIFT_create(nfeatures=500)
# IMG_SIZE = (224,224)
#
# # -----------------------------
# # FEATURE FUNCTION
# # -----------------------------
# def extract_feature(image):
#
#     image = cv2.resize(image, IMG_SIZE)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     kp, des = sift.detectAndCompute(gray, None)
#
#     if des is None:
#         return None, None, image
#
#     # Draw keypoints for visualization
#     key_img = cv2.drawKeypoints(image, kp, None,
#                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     hist = np.zeros(len(kmeans.cluster_centers_))
#     clusters = kmeans.predict(des)
#
#     for c in clusters:
#         hist[c] += 1
#
#     return hist.reshape(1,-1), kp, key_img
#
# # -----------------------------
# # UI
# # -----------------------------
# st.set_page_config(layout="wide")
# st.title("🧠 Indian Sign Language Research Demo (SIFT + KNN)")
#
# mode = st.sidebar.radio(
#     "Select Input Mode",
#     ["Upload Image", "Take Picture", "Live Webcam"]
# )
#
# # -----------------------------
# # Prediction Display Function
# # -----------------------------
# def show_prediction(feature):
#
#     probs = knn_model.predict_proba(feature)[0]
#
#     # Top 3 predictions
#     top3_idx = np.argsort(probs)[-3:][::-1]
#
#     st.subheader("🔎 Top Predictions")
#
#     for i in top3_idx:
#         label = class_names[i]
#         confidence = probs[i]*100
#         st.write(f"{label} — {confidence:.2f}%")
#         st.progress(int(confidence))
#
# # -----------------------------
# # IMAGE UPLOAD
# # -----------------------------
# if mode == "Upload Image":
#
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
#
#     if uploaded_file is not None:
#
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#
#         feature, kp, key_img = extract_feature(img)
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.image(img, channels="BGR", caption="Original Image")
#
#         if feature is None:
#             st.warning("⚠️ No SIFT features detected. Try clearer hand image.")
#         else:
#             feature = scaler.transform(feature)
#             with col2:
#                 st.image(key_img, channels="BGR", caption="SIFT Keypoints")
#
#             pred = knn_model.predict(feature)[0]
#             confidence = knn_model.predict_proba(feature)[0].max()*100
#
#             st.success(f"Prediction: {class_names[pred]}")
#             st.info(f"Confidence: {confidence:.2f}%")
#
#             show_prediction(feature)
#
# # -----------------------------
# # TAKE PICTURE
# # -----------------------------
# if mode == "Take Picture":
#
#     camera = st.camera_input("Capture image")
#
#     if camera is not None:
#
#         file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#
#         feature, kp, key_img = extract_feature(img)
#
#         st.image(img, channels="BGR")
#
#         if feature is None:
#             st.warning("⚠️ No SIFT features detected.")
#         else:
#             feature = scaler.transform(feature)
#             st.image(key_img, channels="BGR", caption="SIFT Keypoints")
#
#             pred = knn_model.predict(feature)[0]
#             confidence = knn_model.predict_proba(feature)[0].max()*100
#
#             st.success(f"Prediction: {class_names[pred]}")
#             st.info(f"Confidence: {confidence:.2f}%")
#
#             show_prediction(feature)
#
# # -----------------------------
# # 🔥 LIVE WEBCAM DETECTION
# # -----------------------------
# if mode == "Live Webcam":
#
#     run = st.checkbox("Start Live Detection")
#     FRAME_WINDOW = st.image([])
#
#     cap = cv2.VideoCapture(0)
#
#     while run:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         feature, kp, key_img = extract_feature(frame)
#
#         if feature is not None:
#
#             feature = scaler.transform(feature)
#             pred = knn_model.predict(feature)[0]
#             confidence = knn_model.predict_proba(feature)[0].max()*100
#
#             cv2.putText(
#                 frame,
#                 f"{class_names[pred]} ({confidence:.1f}%)",
#                 (10,30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0,255,0),
#                 2
#             )
#
#         FRAME_WINDOW.image(frame, channels="BGR")
#
#     cap.release()


# SHIFT+handmask
# import streamlit as st
# import cv2
# import numpy as np
# import joblib
#
# # -----------------------------
# # LOAD MODELS
# # -----------------------------
# knn_model = joblib.load("hand/sift_knn_model_hand.pkl")
# kmeans = joblib.load("hand/sift_kmeans_hand.pkl")
# class_names = joblib.load("hand/class_names_hand.pkl")
# scaler = joblib.load("hand/sift_scaler_hand.pkl")
#
# # knn_model = joblib.load("sift_knn_model.pkl")
# # kmeans = joblib.load("sift_kmeans.pkl")
# # class_names = joblib.load("class_names.pkl")
# # scaler = joblib.load("sift_scaler.pkl")
#
# sift = cv2.SIFT_create(nfeatures=500)
# IMG_SIZE = (224,224)
#
# def segment_hand_pro(image):
#
#     img = cv2.resize(image, (224,224))
#     blur = cv2.GaussianBlur(img,(5,5),0)
#
#     # --- Convert color spaces ---
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
#     ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
#
#     # --- Adaptive Skin Mask (More Robust) ---
#     lower_hsv = np.array([0, 25, 50], dtype=np.uint8)
#     upper_hsv = np.array([35, 200, 255], dtype=np.uint8)
#     mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
#
#     lower_ycrcb = np.array([0,135,85], dtype=np.uint8)
#     upper_ycrcb = np.array([255,180,135], dtype=np.uint8)
#     mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
#
#     mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
#
#     # --- Morphology (Noise Remove) ---
#     kernel = np.ones((5,5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     mask = cv2.medianBlur(mask,5)
#
#     # --- Largest Contour Selection ---
#     contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     if len(contours) > 0:
#
#         c = max(contours, key=cv2.contourArea)
#
#         # ignore very small detections
#         if cv2.contourArea(c) < 1500:
#             return img
#
#         x,y,w,h = cv2.boundingRect(c)
#
#         # padding
#         pad = int(0.1 * max(w,h))
#         x = max(0, x-pad)
#         y = max(0, y-pad)
#         w = min(224-x, w+2*pad)
#         h = min(224-y, h+2*pad)
#
#         hand = img[y:y+h, x:x+w]
#
#         # --- Background Suppression ---
#         hand_mask = np.zeros((224,224), dtype=np.uint8)
#         cv2.drawContours(hand_mask, [c], -1, 255, -1)
#
#         hand_only = cv2.bitwise_and(img, img, mask=hand_mask)
#         hand_only = hand_only[y:y+h, x:x+w]
#
#         hand_only = cv2.resize(hand_only,(224,224))
#
#         return hand_only
#
#     # fallback
#     return img
#
# def get_hand_mask(image):
#
#     img = cv2.resize(image,(224,224))
#
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#
#     lower_hsv = np.array([0,25,50],dtype=np.uint8)
#     upper_hsv = np.array([35,200,255],dtype=np.uint8)
#     mask_hsv = cv2.inRange(hsv,lower_hsv,upper_hsv)
#
#     lower_ycrcb = np.array([0,135,85],dtype=np.uint8)
#     upper_ycrcb = np.array([255,180,135],dtype=np.uint8)
#     mask_ycrcb = cv2.inRange(ycrcb,lower_ycrcb,upper_ycrcb)
#
#     mask = cv2.bitwise_and(mask_hsv,mask_ycrcb)
#
#     kernel = np.ones((5,5),np.uint8)
#     mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
#
#     return mask
#
#
#
# # -----------------------------
# # FEATURE FUNCTION
# # -----------------------------
# def extract_feature(image):
#
#     image = cv2.resize(image, IMG_SIZE)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # ⭐ get mask
#     mask = get_hand_mask(image)
#
#     # detect all keypoints
#     kp, des = sift.detectAndCompute(gray, None)
#
#     if des is None or len(kp) == 0:
#         return None, None, image
#
#     # ⭐ FILTER KEYPOINTS INSIDE MASK ONLY
#     filtered_des = []
#     filtered_kp = []
#
#     for i, point in enumerate(kp):
#
#         x = int(point.pt[0])
#         y = int(point.pt[1])
#
#         if mask[y,x] > 0:
#             filtered_des.append(des[i])
#             filtered_kp.append(point)
#
#     if len(filtered_des) == 0:
#         return None, None, image
#
#     filtered_des = np.array(filtered_des, dtype=np.float32)
#
#     # draw filtered keypoints
#     key_img = cv2.drawKeypoints(image, filtered_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#
#     # build histogram
#     hist = np.zeros(len(kmeans.cluster_centers_))
#     clusters = kmeans.predict(filtered_des)
#
#     for c in clusters:
#         hist[c] += 1
#
#     return hist.reshape(1,-1), filtered_kp, key_img
#
# # -----------------------------
# # UI
# # -----------------------------
# st.set_page_config(layout="wide")
# st.title("🧠 Indian Sign Language Research Demo (SIFT + KNN)")
#
# mode = st.sidebar.radio(
#     "Select Input Mode",
#     ["Upload Image", "Take Picture", "Folder Test"]
# )
#
#
# # -----------------------------
# # Prediction Display Function
# # -----------------------------
# def show_prediction(feature):
#
#     probs = knn_model.predict_proba(feature)[0]
#
#     # Top 3 predictions
#     top3_idx = np.argsort(probs)[-3:][::-1]
#
#     st.subheader("🔎 Top Predictions")
#
#     for i in top3_idx:
#         label = class_names[i]
#         confidence = probs[i]*100
#         st.write(f"{label} — {confidence:.2f}%")
#         st.progress(int(confidence))
#
# # -----------------------------
# # IMAGE UPLOAD
# # -----------------------------
# if mode == "Upload Image":
#
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
#
#     if uploaded_file is not None:
#
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#
#         feature, kp, key_img = extract_feature(img)
#
#         col1, col2 = st.columns(2)
#
#         with col1:
#             st.image(img, channels="BGR", caption="Original Image")
#
#         if feature is None:
#             st.warning("⚠️ No SIFT features detected. Try clearer hand image.")
#         else:
#             feature = scaler.transform(feature)
#             with col2:
#                 st.image(key_img, channels="BGR", caption="SIFT Keypoints")
#
#             pred = knn_model.predict(feature)[0]
#             confidence = knn_model.predict_proba(feature)[0].max()*100
#
#             st.success(f"Prediction: {class_names[pred]}")
#             st.info(f"Confidence: {confidence:.2f}%")
#
#             show_prediction(feature)
#
# # -----------------------------
# # TAKE PICTURE
# # -----------------------------
# if mode == "Take Picture":
#
#     camera = st.camera_input("Capture image")
#
#     if camera is not None:
#
#         file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#
#         feature, kp, key_img = extract_feature(img)
#
#         st.image(img, channels="BGR")
#
#         if feature is None:
#             st.warning("⚠️ No SIFT features detected.")
#         else:
#             feature = scaler.transform(feature)
#             st.image(key_img, channels="BGR", caption="SIFT Keypoints")
#
#             pred = knn_model.predict(feature)[0]
#             confidence = knn_model.predict_proba(feature)[0].max()*100
#
#             st.success(f"Prediction: {class_names[pred]}")
#             st.info(f"Confidence: {confidence:.2f}%")
#
#             show_prediction(feature)
#
# # -----------------------------
# # 📂 FULL RESEARCH FOLDER TEST PANEL
# # -----------------------------
# if mode == "Folder Test":
#
#     st.subheader("📂 Research Batch Evaluation Panel")
#
#     import os
#     from sklearn.metrics import confusion_matrix
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd
#
#     folder_path = st.text_input("Enter Folder Path")
#
#     if st.button("Run Evaluation"):
#
#         if folder_path == "":
#             st.warning("Please enter folder path.")
#         else:
#
#             y_true = []
#             y_pred = []
#
#             wrong_samples = []
#
#             progress = st.progress(0)
#             status = st.empty()
#
#             # class_list = sorted(os.listdir(folder_path))
#             #
#             all_images = []
#             #
#             # for cname in class_list:
#             #     cpath = os.path.join(folder_path, cname)
#             #
#             #     if not os.path.isdir(cpath):
#             #         continue
#             #
#             #     for img_name in os.listdir(cpath):
#             #         all_images.append((cname, os.path.join(cpath,img_name)))
#
#             # ⭐ detect if folder contains images directly
#             files = os.listdir(folder_path)
#
#             has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
#
#             if has_images:
#                 # single class folder
#                 true_class = os.path.basename(folder_path)
#
#                 for img_name in files:
#                     if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#                         all_images.append((true_class, os.path.join(folder_path, img_name)))
#
#             else:
#                 # multi-class folder
#                 class_list = sorted(os.listdir(folder_path))
#
#                 for cname in class_list:
#                     cpath = os.path.join(folder_path, cname)
#
#                     if not os.path.isdir(cpath):
#                         continue
#
#                     for img_name in os.listdir(cpath):
#                         if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#                             all_images.append((cname, os.path.join(cpath, img_name)))
#
#             total_imgs = len(all_images)
#
#             for idx,(true_class, img_path) in enumerate(all_images):
#
#                 img = cv2.imread(img_path)
#
#                 feature, kp, key_img = extract_feature(img)
#
#                 if feature is not None:
#
#                     feature = scaler.transform(feature)
#
#                     pred = knn_model.predict(feature)[0]
#                     pred_label = class_names[pred]
#
#                     y_true.append(true_class)
#                     y_pred.append(pred_label)
#
#                     # save wrong samples
#                     if pred_label != true_class:
#                         wrong_samples.append((img, true_class, pred_label))
#
#                 progress.progress((idx+1)/total_imgs)
#                 status.text(f"Processing {idx+1}/{total_imgs}")
#
#             # -----------------------------
#             # 📊 Accuracy
#             # -----------------------------
#             correct = sum([1 for t,p in zip(y_true,y_pred) if t==p])
#             total = len(y_true)
#
#             acc = (correct/total)*100 if total>0 else 0
#
#             st.success(f"Total Images: {total}")
#             st.success(f"Correct Predictions: {correct}")
#             st.success(f"Accuracy: {acc:.2f}%")
#
#             # -----------------------------
#             # 📈 Per Class Accuracy Table
#             # -----------------------------
#             st.subheader("📊 Per Class Accuracy")
#
#             df = pd.DataFrame({"true":y_true,"pred":y_pred})
#
#             class_acc = []
#
#             for cname in class_names:
#                 subset = df[df["true"]==cname]
#                 if len(subset)==0:
#                     continue
#
#                 correct_c = (subset["true"]==subset["pred"]).sum()
#                 acc_c = (correct_c/len(subset))*100
#
#                 class_acc.append([cname,len(subset),acc_c])
#
#             acc_df = pd.DataFrame(class_acc,columns=["Class","Total","Accuracy"])
#             st.dataframe(acc_df)
#
#             # -----------------------------
#             # 📊 Confusion Matrix
#             # -----------------------------
#             st.subheader("📊 Confusion Matrix")
#
#             cm = confusion_matrix(y_true,y_pred,labels=class_names)
#
#             fig, ax = plt.subplots(figsize=(10,8))
#             sns.heatmap(cm, cmap="Blues", ax=ax)
#             ax.set_xlabel("Predicted")
#             ax.set_ylabel("Actual")
#
#             st.pyplot(fig)
#
#             # -----------------------------
#             # ❌ Wrong Prediction Preview
#             # -----------------------------
#             st.subheader("❌ Misclassified Samples")
#
#             if len(wrong_samples)==0:
#                 st.success("No wrong predictions 🎉")
#             else:
#
#                 cols = st.columns(3)
#
#                 for i,(img,t,p) in enumerate(wrong_samples[:12]):
#
#                     with cols[i%3]:
#                         st.image(img,channels="BGR")
#                         st.write(f"Actual: {t}")
#                         st.write(f"Predicted: {p}")


# SHIFT+improv
import streamlit as st
import cv2
import numpy as np
import joblib

# -----------------------------
# LOAD MODELS
# -----------------------------
#soft Mask
knn_model = joblib.load("hand/SIFTSF/sift_SF_knn_model_hand.pkl")
kmeans = joblib.load("hand/SIFTSF/sift_SF_kmeans_hand.pkl")
class_names = joblib.load("hand/SIFTSF/class_SF_names_hand.pkl")
scaler = joblib.load("hand/SIFTSF/sift_SF_scaler_hand.pkl")

#Improv Mask
# knn_model = joblib.load("hand/SIFTIMP/sift_IMP_knn_model_hand.pkl")
# kmeans = joblib.load("hand/SIFTIMP/sift_IMP_kmeans_hand.pkl")
# class_names = joblib.load("hand/SIFTIMP/class_IMP_names_hand.pkl")
# scaler = joblib.load("hand/SIFTIMP/sift_IMP_scaler_hand.pkl")

#Simple Mask
# knn_model = joblib.load("sift_knn_model.pkl")
# kmeans = joblib.load("sift_kmeans.pkl")
# class_names = joblib.load("class_names.pkl")
# scaler = joblib.load("sift_scaler.pkl")

sift = cv2.SIFT_create(nfeatures=500)
IMG_SIZE = (224,224)

def segment_hand_pro(image):

    img = cv2.resize(image, (224,224))
    blur = cv2.GaussianBlur(img,(5,5),0)

    # --- Convert color spaces ---
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)

    # --- Adaptive Skin Mask (More Robust) ---
    lower_hsv = np.array([0, 25, 50], dtype=np.uint8)
    upper_hsv = np.array([35, 200, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    lower_ycrcb = np.array([0,135,85], dtype=np.uint8)
    upper_ycrcb = np.array([255,180,135], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # --- Morphology (Noise Remove) ---
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask,5)

    # --- Largest Contour Selection ---
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        c = max(contours, key=cv2.contourArea)

        # ignore very small detections
        if cv2.contourArea(c) < 1500:
            return img

        x,y,w,h = cv2.boundingRect(c)

        # padding
        pad = int(0.1 * max(w,h))
        x = max(0, x-pad)
        y = max(0, y-pad)
        w = min(224-x, w+2*pad)
        h = min(224-y, h+2*pad)

        hand = img[y:y+h, x:x+w]

        # --- Background Suppression ---
        hand_mask = np.zeros((224,224), dtype=np.uint8)
        cv2.drawContours(hand_mask, [c], -1, 255, -1)

        hand_only = cv2.bitwise_and(img, img, mask=hand_mask)
        hand_only = hand_only[y:y+h, x:x+w]

        hand_only = cv2.resize(hand_only,(224,224))

        return hand_only

    # fallback
    return img

def get_hand_mask(image):

    img = cv2.resize(image,(224,224))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0,25,50],dtype=np.uint8)
    upper_hsv = np.array([35,200,255],dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv,lower_hsv,upper_hsv)

    lower_ycrcb = np.array([0,135,85],dtype=np.uint8)
    upper_ycrcb = np.array([255,180,135],dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb,lower_ycrcb,upper_ycrcb)

    mask = cv2.bitwise_and(mask_hsv,mask_ycrcb)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)

    return mask

def sift_with_mask_filter(image):

    img = cv2.resize(image,(224,224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]

    mask = get_hand_mask(img)

    kp, des = sift.detectAndCompute(gray, None)

    if des is None or len(kp) == 0:
        return None, None

    weighted_des = []
    weighted_kp = []

    for i, point in enumerate(kp):

        x = int(point.pt[0])
        y = int(point.pt[1])

        # ⭐ face reject (top 20%)
        if y < int(0.20 * h):
            continue

        # ⭐ hand boost
        if mask[y,x] > 0:
            weighted_des.append(des[i])
            weighted_des.append(des[i])
            weighted_kp.append(point)

        # ⭐ center boost
        elif int(0.20*w) < x < int(0.80*w):
            weighted_des.append(des[i])
            weighted_des.append(des[i])
            weighted_kp.append(point)

        else:
            weighted_des.append(des[i])
            weighted_kp.append(point)

    if len(weighted_des) == 0:
        weighted_des = des
        weighted_kp = kp

    return np.array(weighted_des, dtype=np.float32), weighted_kp


# -----------------------------
# FEATURE FUNCTION
# -----------------------------
# def extract_feature(image):
#
#     image = cv2.resize(image, IMG_SIZE)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # ⭐ get mask
#     mask = get_hand_mask(image)
#
#     # detect all keypoints
#     kp, des = sift.detectAndCompute(gray, None)
#
#     if des is None or len(kp) == 0:
#         return None, None, image
#
#     # ⭐ FILTER KEYPOINTS INSIDE MASK ONLY
#     filtered_des = []
#     filtered_kp = []
#
#     for i, point in enumerate(kp):
#
#         x = int(point.pt[0])
#         y = int(point.pt[1])
#
#         if mask[y,x] > 0:
#             filtered_des.append(des[i])
#             filtered_kp.append(point)
#
#     if len(filtered_des) == 0:
#         return None, None, image
#
#     filtered_des = np.array(filtered_des, dtype=np.float32)
#
#     # draw filtered keypoints
#     key_img = cv2.drawKeypoints(image, filtered_kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#
#     # build histogram
#     hist = np.zeros(len(kmeans.cluster_centers_))
#     clusters = kmeans.predict(filtered_des)
#
#     for c in clusters:
#         hist[c] += 1
#
#     return hist.reshape(1,-1), filtered_kp, key_img
def extract_feature(image):

    image = cv2.resize(image, IMG_SIZE)

    des, kp = sift_with_mask_filter(image)

    if des is None:
        return None, None, image

    # draw keypoints
    key_img = cv2.drawKeypoints(
        image,
        kp,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # build histogram
    hist = np.zeros(len(kmeans.cluster_centers_))
    clusters = kmeans.predict(des)

    for c in clusters:
        hist[c] += 1

    return hist.reshape(1,-1), kp, key_img

# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("🧠 Indian Sign Language Research Demo (SIFT + KNN)")

mode = st.sidebar.radio(
    "Select Input Mode",
    ["Upload Image", "Take Picture", "Folder Test"]
)


# -----------------------------
# Prediction Display Function
# -----------------------------
def show_prediction(feature):

    probs = knn_model.predict_proba(feature)[0]

    # Top 3 predictions
    top3_idx = np.argsort(probs)[-3:][::-1]

    st.subheader("🔎 Top Predictions")

    for i in top3_idx:
        label = class_names[i]
        confidence = probs[i]*100
        st.write(f"{label} — {confidence:.2f}%")
        st.progress(int(confidence))

# -----------------------------
# IMAGE UPLOAD
# -----------------------------
if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        feature, kp, key_img = extract_feature(img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, channels="BGR", caption="Original Image")

        if feature is None:
            st.warning("⚠️ No SIFT features detected. Try clearer hand image.")
        else:
            feature = scaler.transform(feature)
            with col2:
                st.image(key_img, channels="BGR", caption="SIFT Keypoints")

            pred = knn_model.predict(feature)[0]
            confidence = knn_model.predict_proba(feature)[0].max()*100

            st.success(f"Prediction: {class_names[pred]}")
            st.info(f"Confidence: {confidence:.2f}%")

            show_prediction(feature)

# -----------------------------
# TAKE PICTURE
# -----------------------------
if mode == "Take Picture":

    camera = st.camera_input("Capture image")

    if camera is not None:

        file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        feature, kp, key_img = extract_feature(img)

        st.image(img, channels="BGR")

        if feature is None:
            st.warning("⚠️ No SIFT features detected.")
        else:
            feature = scaler.transform(feature)
            st.image(key_img, channels="BGR", caption="SIFT Keypoints")

            pred = knn_model.predict(feature)[0]
            confidence = knn_model.predict_proba(feature)[0].max()*100

            st.success(f"Prediction: {class_names[pred]}")
            st.info(f"Confidence: {confidence:.2f}%")

            show_prediction(feature)

# -----------------------------
# 📂 FULL RESEARCH FOLDER TEST PANEL
# -----------------------------
if mode == "Folder Test":

    st.subheader("📂 Research Batch Evaluation Panel")

    import os
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    folder_path = st.text_input("Enter Folder Path")

    if st.button("Run Evaluation"):

        if folder_path == "":
            st.warning("Please enter folder path.")
        else:

            y_true = []
            y_pred = []

            wrong_samples = []

            progress = st.progress(0)
            status = st.empty()

            # class_list = sorted(os.listdir(folder_path))
            #
            all_images = []
            #
            # for cname in class_list:
            #     cpath = os.path.join(folder_path, cname)
            #
            #     if not os.path.isdir(cpath):
            #         continue
            #
            #     for img_name in os.listdir(cpath):
            #         all_images.append((cname, os.path.join(cpath,img_name)))

            # ⭐ detect if folder contains images directly
            files = os.listdir(folder_path)

            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)

            if has_images:
                # single class folder
                true_class = os.path.basename(folder_path)

                for img_name in files:
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_images.append((true_class, os.path.join(folder_path, img_name)))

            else:
                # multi-class folder
                class_list = sorted(os.listdir(folder_path))

                for cname in class_list:
                    cpath = os.path.join(folder_path, cname)

                    if not os.path.isdir(cpath):
                        continue

                    for img_name in os.listdir(cpath):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            all_images.append((cname, os.path.join(cpath, img_name)))

            total_imgs = len(all_images)

            for idx,(true_class, img_path) in enumerate(all_images):

                img = cv2.imread(img_path)

                feature, kp, key_img = extract_feature(img)

                if feature is not None:

                    feature = scaler.transform(feature)

                    pred = knn_model.predict(feature)[0]
                    pred_label = class_names[pred]

                    y_true.append(true_class)
                    y_pred.append(pred_label)

                    # save wrong samples
                    if pred_label != true_class:
                        wrong_samples.append((img, true_class, pred_label))

                progress.progress((idx+1)/total_imgs)
                status.text(f"Processing {idx+1}/{total_imgs}")

            # -----------------------------
            # 📊 Accuracy
            # -----------------------------
            correct = sum([1 for t,p in zip(y_true,y_pred) if t==p])
            total = len(y_true)

            acc = (correct/total)*100 if total>0 else 0

            st.success(f"Total Images: {total}")
            st.success(f"Correct Predictions: {correct}")
            st.success(f"Accuracy: {acc:.2f}%")

            # -----------------------------
            # 📈 Per Class Accuracy Table
            # -----------------------------
            st.subheader("📊 Per Class Accuracy")

            df = pd.DataFrame({"true":y_true,"pred":y_pred})

            class_acc = []

            for cname in class_names:
                subset = df[df["true"]==cname]
                if len(subset)==0:
                    continue

                correct_c = (subset["true"]==subset["pred"]).sum()
                acc_c = (correct_c/len(subset))*100

                class_acc.append([cname,len(subset),acc_c])

            acc_df = pd.DataFrame(class_acc,columns=["Class","Total","Accuracy"])
            st.dataframe(acc_df)

            # -----------------------------
            # 📊 Confusion Matrix
            # -----------------------------
            st.subheader("📊 Confusion Matrix")

            cm = confusion_matrix(y_true,y_pred,labels=class_names)

            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(cm, cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)

            # -----------------------------
            # ❌ Wrong Prediction Preview
            # -----------------------------
            st.subheader("❌ Misclassified Samples")

            if len(wrong_samples)==0:
                st.success("No wrong predictions 🎉")
            else:

                cols = st.columns(3)

                for i,(img,t,p) in enumerate(wrong_samples[:12]):

                    with cols[i%3]:
                        st.image(img,channels="BGR")
                        st.write(f"Actual: {t}")
                        st.write(f"Predicted: {p}")
