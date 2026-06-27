from tkinter import *
import tkinter
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import json
import base64
import pickle
import warnings
import re
import requests

from dotenv import load_dotenv
load_dotenv()

import bcrypt

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import cv2

# Skimage
from skimage.io import imread
from skimage.transform import resize
from skimage import io, transform

# Scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

import joblib

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Input, Concatenate, Add, Activation, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pymysql
import threading
import time

from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
# ------------------------- GLOBAL VARIABLES -------------------------

model_folder = "model"
os.makedirs(model_folder, exist_ok=True)

model_metrics = {}   # For comparison table
categories = ['Alternaria Leaf Spot',
 'Bacterial Soft Rot',
 'Black Leg',
 'Black Rot',
 'Cabbage Aphid Colony',
 'Cauliflower Mosaic',
 'Club Root',
 'Downy Mildew',
 'Powdery Mildew',
 'Ring Spot',
 'White Rust']
print(categories)

# Metric lists
precision = []
recall = []
fscore = []
accuracy = []

def write_output(message):
    text.configure(state='normal')
    text.insert(END, message)
    text.see(END)   # auto-scroll to latest output
    text.configure(state='disabled')

def uploadDataset():
    global model_folder, path, categories
    model_folder = "model"
    path = "Couliflower_Disease"
    os.makedirs(model_folder, exist_ok=True)
    if not os.path.exists(path):
        messagebox.showwarning("Dataset", f"Dataset folder not found: {path}")
        return
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    write_output(f"Dataset folders discovered: {(categories)} classes.\n\n\n")
    text.see(END)

def preprocessDataset():
    global X, Y, X_train, X_test, Y_train, Y_test, model_folder, path, categories

    X_file = os.path.join(model_folder, "X.npy")
    Y_file = os.path.join(model_folder, "Y.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        write_output("X and Y arrays loaded successfully from disk.\n\n\n")
        text.see(END)

    else:
        X = []
        Y = []

        if not os.path.exists(path):
            write_output(f"Dataset path not found: {path}\n\n\n")
            text.see(END)
            messagebox.showerror("Error", f"Dataset path not found: {path}")
            return

        for root, dirs, files in os.walk(path):
            category_name = os.path.basename(root)
            if category_name in categories:
                write_output(f"Loading category: {category_name}\n\n\n")
                text.see(END)
                for f in tqdm(files):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(root, f)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (64, 64))
                            X.append(img)
                            Y.append(categories.index(category_name))

        X = np.array(X)
        Y = np.array(Y)

        np.save(X_file, X)
        np.save(Y_file, Y)
        write_output("Saved X and Y arrays.\n\n\n")
        text.see(END)

    # Normalize image pixels
    X = X.astype("float32") / 255.0

    # Ensure labels are integer encoded
    Y = Y.astype("int")

    # Split data
    try:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, shuffle=True, stratify=Y, random_state=42
        )
    except Exception as e:
        write_output(f"Error during train_test_split: {e}\n\n\n")
        text.see(END)
        messagebox.showerror("Error", f"train_test_split failed: {e}")
        return

    write_output(f"Training samples: {X_train.shape}\n")
    write_output(f"Testing samples: {X_test.shape}\n\n\n")
    text.see(END)
    messagebox.showinfo("Preprocess", "Dataset preprocessed successfully!")


def show_plot_popup(fig):
    # Run plt.show() in a separate thread
    def _show():
        fig.show()
        plt.show()  # blocking only in this thread

    threading.Thread(target=_show).start()


def calculateMetrics(name, y_test, y_pred, y_proba=None, categories=categories):

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    model_metrics[name] = [acc, prec, rec, f1]

    write_output(f"\n{name} Results\n")
    write_output(f"Accuracy : {acc:.4f}\n")
    write_output(f"Precision: {prec:.4f}\n")
    write_output(f"Recall   : {rec:.4f}\n")
    write_output(f"F1-Score : {f1:.4f}\n\n")
    CR = classification_report(y_test, y_pred,target_names=categories)
    write_output(name+' Classification Report \n')
    write_output(str(CR) +"\n\n")
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,
        cmap="viridis",
        fmt="g"
    )

    plt.xticks(rotation=45, ha="right")  # tilt 45° to avoid overlap
    plt.yticks(rotation=45, ha="right")               # keep y straight (optional)

    plt.tight_layout()
    plt.show()
    
    y_true = y_test
    algorithm = name
    # ROC-AUC (if proba provided)
    if y_proba is not None:
        try:
            n_classes = len(categories)
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            # If shape mismatch, try to adapt:
            if y_proba.shape[1] != n_classes:
                write_output("predict_proba shape does not match number of classes; skipping ROC-AUC.\n")
                text.see(END)
            else:
                # Skip classes missing from y_true
                valid_classes = []
                plt.figure(figsize=(7,7))
                for i in range(n_classes):
                    if len(np.unique(y_true_bin[:, i])) == 1:
                        write_output(f"Skipping ROC for class '{categories[i]}' (only one class present in y_true).\n")
                        text.see(END)
                        continue
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    valid_classes.append(i)
                    plt.plot(fpr, tpr, label=f"{categories[i]} (AUC={roc_auc:.2f})")
                if valid_classes:
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title(f"{algorithm} ROC-AUC Curve")
                    plt.legend(loc="lower right")
                    roc_path = os.path.join("results", f"{algorithm}_roc_auc.png")
                    plt.savefig(roc_path, bbox_inches='tight')
                    #show_plot_popup(plt.gcf())
                    plt.show()
                    plt.close()
                    write_output(f"ROC-AUC plot saved: {roc_path}\n")
                    text.see(END)
                else:
                    write_output("No valid classes for ROC-AUC.\n")
                    text.see(END)
        except Exception as e:
            write_output(f"ROC-AUC computation error: {e}\n")
            text.see(END)




# ------------------ MODEL FUNCTIONS ------------------
def train_and_evaluate_LR():
    global X_train, X_test, Y_train, Y_test
    text.delete('1.0', END)

    model_path = os.path.join("model", "LogisticRegression_model.pkl")


    if os.path.exists(model_path):
        write_output("Loading saved Logistic Regression model...\n")
        text.see(END)
        log_reg = joblib.load(model_path)
    else:
        write_output("Training Logistic Regression model...\n")
        text.see(END)
        # Flatten images (important for ML models)
        X_train_flat = X_train.reshape(len(X_train), -1)

        # Logistic Regression Model
        log_reg = LogisticRegression(max_iter=20,C=0.02, solver='lbfgs', multi_class='multinomial')

        log_reg.fit(X_train_flat, Y_train)

        joblib.dump(log_reg, model_path)
        write_output("Model saved.\n")
        text.see(END)


    X_test_flat = X_test.reshape(len(X_test), -1)

    y_pred = log_reg.predict(X_test_flat)

    try:
        y_proba = log_reg.predict_proba(X_test_flat)
    except Exception:
        y_proba = None

    calculateMetrics("Logistic Regression Classifier", Y_test, y_pred, y_proba, categories)

def train_and_evaluate_DTC():
    global X_train, X_test, Y_train, Y_test
    text.delete('1.0', END)

    from sklearn.tree import DecisionTreeClassifier

    model_path = os.path.join("model", "DecisionTree_model.pkl")

    # -----------------------------
    # LOAD IF EXISTS
    # -----------------------------
    if os.path.exists(model_path):
        write_output("Loading saved Decision Tree model...\n")
        text.see(END)
        dtc = joblib.load(model_path)

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    else:
        write_output("Training Decision Tree model...\n")
        text.see(END)

        # Flatten images for ML model
        X_train_flat = X_train.reshape(len(X_train), -1)

        # Decision Tree Classifier
        dtc = DecisionTreeClassifier(
            criterion="gini",      # or "entropy"
            max_depth=5,          # controls overfitting
            min_samples_split=5,
            random_state=42
        )

        dtc.fit(X_train_flat, Y_train)

        joblib.dump(dtc, model_path)

        write_output("Model saved.\n")
        text.see(END)

    # -----------------------------
    # TESTING
    # -----------------------------
    X_test_flat = X_test.reshape(len(X_test), -1)

    y_pred = dtc.predict(X_test_flat)

    try:
        y_proba = dtc.predict_proba(X_test_flat)
    except Exception:
        y_proba = None

    # -----------------------------
    # METRICS
    # -----------------------------
    calculateMetrics(
        "Decision Tree Classifier",
        Y_test,
        y_pred,
        y_proba,
        categories
    )


def train_and_evaluate_ann():
    global X_train, X_test, Y_train, Y_test
    text.delete('1.0', END)

    ann_model_path = os.path.join("model", "ANN_model.h5")


    X_train_ann = X_train.reshape(len(X_train), -1)
    X_test_ann  = X_test.reshape(len(X_test), -1)


    if os.path.exists(ann_model_path):
        write_output("Loading saved ANN model...\n")
        text.see(END)
        ann = load_model(ann_model_path)

    else:
        write_output("Training ANN model...\n")
        text.see(END)

        input_dim = X_train_ann.shape[1]   # total pixels = 64 * 64 * 3 (if your images are 64x64)

        ann = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.3),

            Dense(32, activation='relu'),

            Dense(len(categories), activation='softmax')
        ])

        # sensible learning rate (fixed)
        ann.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        history = ann.fit(
            X_train_ann, Y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        ann.save(ann_model_path)
        write_output("ANN Model saved.\n")
        text.see(END)


    y_proba_ann = ann.predict(X_test_ann)          # probability scores
    y_pred_ann = np.argmax(y_proba_ann, axis=1)    # final predicted labels
    calculateMetrics("ANN Classifier", Y_test, y_pred_ann, y_proba_ann, categories)


def train_and_evaluate_hybrid_inceptionresnet_tk():
    global X_train, X_test, Y_train, Y_test
    text.delete('1.0', END)

    feature_model_path = os.path.join("model", "Hybrid_InceptionResNet_FeatureExtractor.h5")
    hybrid_lrc_path = os.path.join("model", "Hybrid_InceptionResNet_LRC.pkl")
    plain_lrc_path = os.path.join("model", "Plain_LRC.pkl")

    def inception_resnet_block(x, filters):
        b1 = Conv2D(filters, (1,1), padding="same", activation="relu")(x)

        b2 = Conv2D(filters, (1,1), padding="same", activation="relu")(x)
        b2 = Conv2D(filters, (3,3), padding="same", activation="relu")(b2)

        b3 = Conv2D(filters, (1,1), padding="same", activation="relu")(x)
        b3 = Conv2D(filters, (5,5), padding="same", activation="relu")(b3)

        merged = Concatenate()([b1, b2, b3])
        res = Conv2D(filters, (1,1), padding="same")(merged)
        out = Add()([x, res])
        out = Activation("relu")(out)
        return out

    def build_feature_extractor():
        inp = Input(shape=(64, 64, 3))

        x = Conv2D(32, (3,3), activation='relu', padding='same')(inp)
        x = MaxPooling2D((2,2))(x)

        x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2))(x)

        x = inception_resnet_block(x, 64)

        x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2,2))(x)

        x = inception_resnet_block(x, 128)

        x = GlobalAveragePooling2D()(x)
        features = Dense(256, activation='relu')(x)

        output = Dense(len(categories), activation='softmax')(features)

        return Model(inp, output)

    if X_train is None:
        write_output("Dataset not preprocessed. Run Preprocess Dataset first.\n")
        text.see(END)
        return

    if os.path.exists(feature_model_path):
        feature_extractor = load_model(feature_model_path)
    else:
        feature_extractor = build_feature_extractor()

        feature_extractor.compile(
            optimizer=Adam(0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        feature_extractor.fit(
            X_train,
            Y_train,
            epochs=15,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        feature_extractor.save(feature_model_path)

    # Always convert to feature extractor
    feature_extractor = Model(
        feature_extractor.input,
        feature_extractor.layers[-2].output
    )

    X_train_feat = feature_extractor.predict(X_train)
    X_test_feat = feature_extractor.predict(X_test)

    if os.path.exists(hybrid_lrc_path):
        hybrid_lrc = joblib.load(hybrid_lrc_path)
    else:
        hybrid_lrc = LogisticRegression()
        hybrid_lrc.fit(X_train_feat, Y_train)
        joblib.dump(hybrid_lrc, hybrid_lrc_path)

    hybrid_pred = hybrid_lrc.predict(X_test_feat)
    hybrid_proba = hybrid_lrc.predict_proba(X_test_feat)
    hybrid_acc = accuracy_score(Y_test, hybrid_pred)

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    if os.path.exists(plain_lrc_path):
        plain_lrc = joblib.load(plain_lrc_path)
    else:
        plain_lrc = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
        plain_lrc.fit(X_train_flat, Y_train)
        joblib.dump(plain_lrc, plain_lrc_path)

    plain_pred = plain_lrc.predict(X_test_flat)
    plain_proba = plain_lrc.predict_proba(X_test_flat)
    plain_acc = accuracy_score(Y_test, plain_pred)

    if plain_acc > hybrid_acc:
        final_pred = plain_pred
        final_proba = plain_proba
    else:
        final_pred = hybrid_pred
        final_proba = hybrid_proba

    text.see(END)

    calculateMetrics(
        "Hybrid_InceptionResNet_LRC",
        Y_test,
        final_pred,
        final_proba,
        categories
    )

def show_model_comparison():
    if not model_metrics:
        messagebox.showerror("Error", "Run at least one model first!")
        return

    win = Toplevel(main)
    win.title("Model Comparison")
    win.geometry("650x300")

    cols = ("Model", "Accuracy", "Precision", "Recall", "F1-Score")
    tree = ttk.Treeview(win, columns=cols, show="headings")

    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, anchor="center", width=120)

    for model, vals in model_metrics.items():
        tree.insert("", END, values=(
            model,
            f"{vals[0]:.4f}",
            f"{vals[1]:.4f}",
            f"{vals[2]:.4f}",
            f"{vals[3]:.4f}",
        ))

    tree.pack(expand=True, fill=BOTH)


def predict_image_single(image_path):

    img = load_img(image_path, target_size=(64, 64))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    plain_lrc_path = "model/Plain_LRC.pkl"
    plain_lrc = joblib.load(plain_lrc_path)

    flat = img.reshape(1, -1)

    proba = plain_lrc.predict_proba(flat)
    pred_class = int(plain_lrc.predict(flat)[0]) 
    confidence = float(np.max(proba))

    return pred_class, confidence, proba


def predict_images_in_folder(folder_path, text_widget=None):


    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    results = []

    for fname in image_files:
        try:
            img_path = os.path.join(folder_path, fname)
            pred_idx, conf, _ = predict_image_single(img_path)
            label = categories[pred_idx]
            results.append((fname, label, conf))

            if text_widget:
                text_widget.insert(
                    END, f"{fname} → {label} (conf={conf:.3f})\n"
                )
                text_widget.see(END)

        except Exception as e:
            if text_widget:
                text_widget.insert(END, f"{fname} error: {e}\n")
                text_widget.see(END)

    df = pd.DataFrame(results, columns=["Image", "Predicted_Label", "Confidence"])
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "batch_predictions.csv")
    df.to_csv(csv_path, index=False)

    if text_widget:
        text_widget.insert(END, f"\nSaved results to {csv_path}\n")
        text_widget.see(END)

    return df

def predict_folder():
    text.delete('1.0', END)
    folder_path = filedialog.askdirectory(title="Select folder containing images")
    if not folder_path:
        return

    def target():
        write_output(f"Starting batch prediction for: {folder_path}\n")
        text.see(END)

        df = predict_images_in_folder(folder_path, text_widget=text)

        write_output("Batch prediction finished.\n")
        text.see(END)

        try:
            thumbs_win = tk.Toplevel(main)
            thumbs_win.title("Prediction Thumbnails")

            for idx, row in df.head(12).iterrows():
                img_path = os.path.join(folder_path, row["Image"])
                pil = Image.open(img_path).convert("RGB")
                pil.thumbnail((160, 160), Image.LANCZOS)

                imgtk = ImageTk.PhotoImage(pil)
                lbl = tk.Label(thumbs_win, image=imgtk)
                lbl.image = imgtk
                lbl.grid(row=idx // 3, column=(idx % 3) * 2, padx=5, pady=5)

                tk.Label(
                    thumbs_win,
                    text=f"{row['Predicted_Label']} | Confidence: {row['Confidence']*100:.2f}%"
                ).grid(row=idx // 3, column=(idx % 3) * 2 + 1)

        except Exception as e:
            write_output(f"Thumbnail error: {e}\n")
            text.see(END)

    threading.Thread(target=target, daemon=True).start()


# ------------------ DB helper ---------------------
def connect_db():
    return pymysql.connect(host=os.getenv("DB_HOST"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"), database=os.getenv("DB_NAME"))


# ------------------ Signup/Login ------------------
def signup(role):
    def register_user():

        import re

        username = username_entry.get().strip()
        password = password_entry.get().strip()

        # Clear previous errors
        username_error.config(text="")
        password_error.config(text="")

        # Empty fields
        if not username or not password:

            username_error.config(text="")
            password_error.config(text="")

            if not username:
                username_error.config(
                    text="Username cannot be empty"
                )

            if not password:
                password_error.config(
                    text="Password cannot be empty"
                )

            return

        # Username length
        if len(username) < 4 or len(username) > 20:
            username_error.config(
                text="Username: 4–20 characters required"
            )
            return

        # Username allowed chars
        if not re.match(
            r"^[A-Za-z0-9_]+$",
            username
        ):
            username_error.config(
                text="Only letters, numbers and _ allowed"
            )
            return

        # Password validation
        password_pattern = (
            r"^(?=.*[a-z])"
            r"(?=.*[A-Z])"
            r"(?=.*\d)"
            r"(?=.*[@$!%*?&])"
        )

        if len(password) < 8:
            password_error.config(
                text="Password must contain minimum 8 characters"
            )
            return

        if not re.search(
            password_pattern,
            password
        ):
            password_error.config(
                text="Need uppercase, lowercase, number & special character"
            )
            return

        try:

            conn = connect_db()
            cursor = conn.cursor()

            # Check existing username
            cursor.execute(
                "SELECT * FROM users WHERE username=%s",
                (username,)
            )

            existing = cursor.fetchone()

            if existing:

                username_error.config(
                    text="Username already exists. Try another"
                )

                conn.close()
                return

            # Hash password
            hashed_password = bcrypt.hashpw(
                password.encode('utf-8'),
                bcrypt.gensalt()
            )

            # Insert new user
            cursor.execute(
                "INSERT INTO users(username,password,role) VALUES(%s,%s,%s)",
                (
                    username,
                    hashed_password.decode('utf-8'),
                    role
                )
            )

            conn.commit()
            conn.close()

            messagebox.showinfo(
                "Success",
                f"{role} Signup Successful!"
            )

            signup_window.destroy()

        except Exception as e:

            messagebox.showerror(
                "Database Error",
                str(e)
            )

    signup_window=tk.Toplevel(main)
    signup_window.geometry("380x350")
    signup_window.title(f"{role} Signup")

    Label(signup_window,text="Username").pack(pady=5)
    username_entry=tk.Entry(signup_window)
    username_entry.pack(pady=5)
    username_error = tk.Label(
    signup_window,
    text="",
    fg="red",
    font=("Segoe UI",8)
    )
    username_error.pack()

    Label(signup_window,text="Password").pack(pady=5)
    password_entry=tk.Entry(signup_window,show="*")
    password_entry.pack(pady=5)
    password_error = tk.Label(
    signup_window,
    text="",
    fg="red",
    font=("Segoe UI",8)
    )
    password_error.pack()

    #signup button styling and setting
    signup_btn = tk.Button(
        signup_window,
        text="Signup",
        command=register_user,
        font=("Segoe UI",9),
        bg="#f5f5f5",
        fg="black",
        activebackground="#dfdfdf",   # hover click color
        activeforeground="black",
        relief="flat",
        bd=0,
        cursor="hand2"
    )
    signup_btn.pack(pady=10, ipadx=15, ipady=5)
    # Hover effect
    signup_btn.bind(
        "<Enter>",
        lambda e: signup_btn.config(bg="#e0e0e0")
    )
    signup_btn.bind(
        "<Leave>",
        lambda e: signup_btn.config(bg="#f5f5f5")
    )

current_user = None

def login(role):
    def verify_user():

        username = username_entry.get().strip()
        password = password_entry.get().strip()

        login_error.config(text="")

        if not username or not password:

            login_error.config(
                text="Please enter all fields"
            )

            return

        try:

            conn = connect_db()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM users WHERE username=%s AND role=%s",
                (username,role)
            )

            user = cursor.fetchone()

            conn.close()

            if not user:

                login_error.config(
                    text="Username does not exist"
                )

                return

            # password column index
            stored_password = user[2]

            if not bcrypt.checkpw(
                password.encode('utf-8'),
                stored_password.encode('utf-8')
            ):

                login_error.config(
                    text="Incorrect password"
                )

                return

            global current_user
            current_user = username

            messagebox.showinfo(
                "Success",
                f"{role} Login Successful!"
            )

            login_window.destroy()

            if role=="Admin":
                show_admin_buttons()

            elif role=="Farmer":
                show_user_buttons()


        except Exception as e:

            messagebox.showerror(
                "Database Error",
                str(e)
            )

    login_window=tk.Toplevel(main)
    login_window.geometry("380x350")
    login_window.title(f"{role} Login")

    Label(login_window,text="Username").pack(pady=5)
    username_entry=tk.Entry(login_window)
    username_entry.pack(pady=5)
    
    Label(login_window,text="Password").pack(pady=5)
    password_entry=tk.Entry(login_window,show="*")
    password_entry.pack(pady=5)
    login_error = tk.Label(
        login_window,
        text="",
        fg="red",
        font=("Segoe UI",8)
    )

    login_error.pack()

    #login button styling and setting
    login_btn = tk.Button(
        login_window,
        text="Login",
        command=verify_user,
        font=("Segoe UI",9),
        bg="#f5f5f5",
        fg="black",
        activebackground="#dfdfdf",   # hover click color
        activeforeground="black",
        relief="flat",
        bd=0,
        cursor="hand2"
    )
    login_btn.pack(pady=10, ipadx=15, ipady=5)
    # Hover effect
    login_btn.bind(
        "<Enter>",
        lambda e: login_btn.config(bg="#e0e0e0")
    )
    login_btn.bind(
        "<Leave>",
        lambda e: login_btn.config(bg="#f5f5f5")
    )

def save_prediction_result(
    username,
    disease,
    confidence
):

    try:

        conn = connect_db()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO results
            (username,disease,confidence)
            VALUES(%s,%s,%s)
            """,
            (
                username,
                disease,
                float(confidence)
            )
        )

        conn.commit()
        conn.close()

        print("Result saved successfully")

    except Exception as e:

        print(
            "Result Save Error:",
            e
        )

def analyze_cauliflower_image(image_path):

    api_key = os.getenv("GEMINI_API_KEY")

    try:
        if not os.path.exists(image_path):
            return json.dumps({"error": "File not found"})

        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

        payload = {
            "contents":[
                {
                    "role":"user",
                    "parts":[
                        {"inline_data":{"mime_type":"image/jpeg","data":image_data}},

                        {"text":
                        "Analyze this image.\n\n"

                        "TASK 1:\n"
                        "Check whether this is a cauliflower plant image.\n\n"

                        "TASK 2:\n"
                        "If cauliflower:\n"
                        "- Detect disease presence\n"
                        "- Estimate severity (Mild/Moderate/Severe)\n"
                        "- Identify affected part (Leaf/Curd/Stem)\n\n"

                        "IMPORTANT:\n"
                        "Never return null values.\n"
                        "If uncertain choose the closest option.\n"
                        "Affected part MUST be Leaf, Curd, or Stem.\n"
                        "Severity MUST be Mild, Moderate, or Severe.\n\n"

                        "Return STRICT JSON:\n"
                        "{\n"
                        "\"is_cauliflower\":true/false,\n"
                        "\"image_type\":\"Cauliflower or Other\",\n"
                        "\"disease_present\":true/false,\n"
                        "\"severity\":\"Mild/Moderate/Severe\",\n"
                        "\"affected_part\":\"Leaf/Curd/Stem\"\n"
                        "}"
                        }
                    ]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        res_json = response.json()
        text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]

        print("\n====== GEMINI RAW RESPONSE ======")
        print(text_response)
        print("=================================\n")

        match = re.findall(r"\{.*\}", text_response, re.DOTALL)

        if not match:
            return json.dumps({"error": "No JSON found"})

        return json.dumps(json.loads(match[0]), indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)})


# --------------------------------------------------
#  TKINTER PREDICTION FUNCTION
# --------------------------------------------------

def predict():    
    text.delete('1.0', END)
    write_output("Starting Prediction on Test Image...\n")

    # --------------------------------------------------
    # STEP 1: SELECT TEST IMAGE
    # --------------------------------------------------
    test_image_path = filedialog.askopenfilename( initialdir="TestImages", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not test_image_path:
        write_output("No image selected.\n")
        return

    # --------------------------------------------------
    # STEP 2: RUN XAI  DETECTION
    # --------------------------------------------------
    result = analyze_cauliflower_image(test_image_path)

    # Convert JSON string to dictionary
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            write_output("Error: XAI returned invalid JSON.\n")
            return

    print("XAI Result:", result)

    # --------------------------------------------------
    # CASE C: XAI FAILED 
    # --------------------------------------------------
    if "error" in result:
        write_output("XAI Failed: Showing Only TL Prediction.\n")

        pred_idx, conf, _ = predict_image_single(test_image_path)
        prediction = categories[pred_idx]
        save_prediction_result(current_user, prediction, conf)
        write_output("Classifier Prediction: {prediction}\n")

        plt.figure(figsize=(14, 6))

        # 121 → Original Image
        plt.subplot(121)
        orig = cv2.imread(test_image_path)
        if len(orig.shape) == 3:  
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")


        # 122 → TL Prediction
        plt.subplot(122)
        img_cls = cv2.imread(test_image_path)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_cls, (500, 500))
        cv2.putText(img_cls, f"Predicted: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        plt.imshow(img_cls)
        plt.title("TL Prediction")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --------------------------------------------------
    # CASE A: 
    # --------------------------------------------------
    if result.get("is_cauliflower") == True:
        write_output("Running model classifier...\n")

        # CLASSIFIER PREDICTION
        pred_idx, conf, _ = predict_image_single(test_image_path)
        prediction = categories[pred_idx]
        save_prediction_result(current_user, prediction, conf)
        write_output(f"Classifier Prediction: {prediction}\n")

        # ---------- ANALYSIS WHITE PANEL ----------
        width, height = 800, 600
        result_img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(result_img)

        try:
            font_big = ImageFont.truetype("arial.ttf", 28)
        except:
            font_big = ImageFont.load_default()

        y = 30
        draw.text((30, y), "XAI Analysis Results", fill="black", font=font_big)
        y += 50

        for key, value in result.items():
            draw.text((30, y), f"{key}: {value}", fill="black", font=font_big)
            y += 40

        result_np = np.array(result_img)

        # ---------- DISPLAY 3 SUBPLOTS ----------
        plt.figure(figsize=(18, 6))

        # 131 → Original Image
        plt.subplot(131)
        orig = cv2.imread(test_image_path)
        if len(orig.shape) == 3:  
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        # 132 → XAI Result Panel
        plt.subplot(132)
        plt.imshow(result_np)
        plt.title("XAI Result")
        plt.axis("off")

        # 133 → Classifier Result
        plt.subplot(133)
        img_cls = cv2.imread(test_image_path)
        img_cls = cv2.cvtColor(img_cls, cv2.COLOR_BGR2RGB)
        img_cls = cv2.resize(img_cls, (500, 500))
        cv2.putText(img_cls, f"Classified: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        plt.imshow(img_cls)
        plt.title("Classification Result")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

    # --------------------------------------------------
    # CASE B: NORMAL XAI OUTPUT
    # --------------------------------------------------
    else:
        write_output(f"Detected Image Type = {result.get('image_type')}\n")

        width, height = 800, 600
        result_img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(result_img)

        try:
            font_big = ImageFont.truetype("arial.ttf", 28)
        except:
            font_big = ImageFont.load_default()

        y = 30
        draw.text((30, y), "Image Analysis Result", fill="black", font=font_big)
        y += 50

        draw.text((30, y), "XAI Analysis Results",
                fill="black",
                font=font_big)
        y += 50

        draw.text((30,y),
                f"Image Type: {result.get('image_type')}",
                fill="black",
                font=font_big)
        y += 40

        draw.text((30,y),
                f"Disease: {prediction}",
                fill="black",
                font=font_big)
        y += 40

        draw.text((30,y),
                f"Confidence: {conf:.2f}%",
                fill="black",
                font=font_big)
        y += 40

        draw.text((30,y),
                f"Severity: {result.get('severity')}",
                fill="black",
                font=font_big)
        y += 40

        draw.text((30,y),
                f"Affected Part: {result.get('affected_part')}",
                fill="black",
                font=font_big)

        result_np = np.array(result_img)

        # ---------- DISPLAY 2 SUBPLOTS ----------
        plt.figure(figsize=(14, 6))

        # 121 → Original Image
        plt.subplot(121)
        orig = cv2.imread(test_image_path)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig = cv2.resize(orig, (500, 500))
        plt.imshow(orig)
        plt.title("Original Image")
        plt.axis("off")

        # 122 → XAI Result
        plt.subplot(122)
        plt.imshow(result_np)
        plt.title("Detected Image Type")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
        return

 
# --------------------------------------------------
#  TELEGRAM BOT FUNCTIONS
# --------------------------------------------------
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    MessageHandler, ContextTypes, filters
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "🥦 Send a cauliflower leaf image.\n\n"
        "I can detect:\n"
        "✔ Cauliflower disease\n"
        "✔ Severity\n"
        "✔ Affected part\n"
        "✔ AI Model prediction"
    )

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):

    photo = update.message.photo[-1]
    file = await photo.get_file()

    path = "input.jpg"
    await file.download_to_drive(path)

    await update.message.reply_text("🔍 Analyzing image...")

    # Main model prediction
    pred_idx, conf, _ = predict_image_single(path)

    print("\n========== DEBUG ==========")
    print("Predicted index:", pred_idx)
    print("Prediction:", categories[pred_idx])
    print("Confidence:", conf)
    print("===========================\n")

    prediction = categories[pred_idx]

    try:
        result = analyze_cauliflower_image(path)

        if isinstance(result, str):
            result = json.loads(result)

        severity = result.get("severity", "Unknown")
        affected = result.get("affected_part", "Unknown")

        await update.message.reply_text(
            f"🥦 Disease: {prediction}\n"
            f"📊 Confidence: {conf:.2f}\n"
            f"⚠ Severity: {severity}\n"
            f"🍀 Affected Part: {affected}"
        )

    except Exception as e:

        print("XAI Error:", e)

        await update.message.reply_text(
            f"🥦 Disease: {prediction}\n"
            f"📊 Confidence: {conf:.2f}\n"
            f"⚠ XAI details unavailable"
        )

import threading

def Telegram_Bot():
    global model_folder, categories, transform, model

    write_output("🤖 Starting Telegram Bot...\n")

    def run_bot():
        TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        app = ApplicationBuilder().token(TOKEN).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.PHOTO, handle_image))

        print("🤖 Telegram Bot running in background...")
        app.run_polling()

    threading.Thread(target=run_bot, daemon=True).start()

    write_output("🤖 Telegram Bot is active and waiting for images...\n")


def close():
    main.destroy()
    
    
import tkinter as tk
from tkinter import filedialog, Text, ttk, Scrollbar, Label, Button
from PIL import Image, ImageTk

# =============================
# Main Window
# =============================
main = tk.Tk()
main.title("Cauliflower Leaf Disease Detection")

screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")
main.state("zoomed") 
bg_img = Image.open("background.png")

def resize_bg(event):
    if event.widget != main:
        return
    resized = bg_img.resize((event.width, event.height))
    bg_photo = ImageTk.PhotoImage(resized)
    background_label.config(image=bg_photo)
    background_label.image = bg_photo


background_label = tk.Label(main)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
main.bind("<Configure>", resize_bg)

# =============================
# Dashboard Logic
# =============================
def close():
    main.destroy()
    #text.delete('1.0', END)

def clear_buttons():
    for widget in main.winfo_children():

        # Keep permanent widgets
        if widget in [background_label, title, text, scroll]:
            continue

        widget.destroy()

def clear_textbox():
    text.configure(state='normal')
    text.delete('1.0', END)
    text.configure(state='disabled')

# Button Style
font1 = ('Segoe UI', 11)
btn_style = {
    "font": font1,
    "bg": "#F8F9F9",       # soft white
    "fg": "black",
    "activebackground": "#D6DBDF",  # slightly darker on click
    "activeforeground": "black",
    "relief": "flat",      # removes ugly border
    "bd": 0,
    "cursor": "hand2"
}
def on_enter(e):
    e.widget['bg'] = "#D6DBDF"
def on_leave(e):
    e.widget['bg'] = "#F8F9F9"

def show_admin_buttons():
    clear_buttons()
    clear_textbox()

    #Buttons logic
    btn1 = Button(
        main,
        text="Upload Dataset",
        command=uploadDataset,
        **btn_style
    )
    btn1.place(relx=0.02, rely=0.18, relwidth=0.22,height=38)
    btn1.bind("<Enter>", on_enter)
    btn1.bind("<Leave>", on_leave)

    btn2 = Button(
    main,
    text="Image Preprocessing",
    command=preprocessDataset,
    **btn_style
    )
    btn2.place(relx=0.02,rely=0.25,relwidth=0.22,height=38)
    btn2.bind("<Enter>", on_enter)
    btn2.bind("<Leave>", on_leave)
    
    btn3 = Button(
    main,
    text="Build LRC",
    command=train_and_evaluate_LR,
    **btn_style
    )
    btn3.place(relx=0.02,rely=0.32,relwidth=0.22,height=38)
    btn3.bind("<Enter>", on_enter)
    btn3.bind("<Leave>", on_leave)
    
    btn4 = Button(
    main,
    text="Build ANN",
    command=train_and_evaluate_ann,
    **btn_style
    )
    btn4.place(relx=0.02,rely=0.39,relwidth=0.22,height=38)
    btn4.bind("<Enter>", on_enter)
    btn4.bind("<Leave>", on_leave)
    
    btn5 = Button(
    main,
    text="Build DTC",
    command=train_and_evaluate_DTC,
    **btn_style
    )
    btn5.place(relx=0.02,rely=0.46,relwidth=0.22,height=38)
    btn5.bind("<Enter>", on_enter)
    btn5.bind("<Leave>", on_leave)
    
    btn6 = Button(
    main,
    text="Build InceptionResNet LRC",
    command=train_and_evaluate_hybrid_inceptionresnet_tk,
    **btn_style
    )
    btn6.place(relx=0.02,rely=0.53,relwidth=0.22,height=38)
    btn6.bind("<Enter>", on_enter)
    btn6.bind("<Leave>", on_leave)
    
    btn7 = Button(
    main,
    text="Performance Comparison",
    command=show_model_comparison,
    **btn_style
    )
    btn7.place(relx=0.02,rely=0.60,relwidth=0.22,height=38)
    btn7.bind("<Enter>", on_enter)
    btn7.bind("<Leave>", on_leave)
    
    btn8 = Button(
    main,
    text="Telegram Bot",
    command=Telegram_Bot,
    **btn_style
    )
    btn8.place(relx=0.02,rely=0.67,relwidth=0.22,height=38)
    btn8.bind("<Enter>", on_enter)
    btn8.bind("<Leave>", on_leave)
    
    logout_btn = Button(
    main,
    text="Logout",
    command=show_login_screen,
    font=font1,
    bg="#ff0505",
    fg="white",
    relief="flat",
    bd=0,
    cursor="hand2"
    )
    logout_btn.place(relx=0.02, rely=0.74, relwidth=0.15, height=38)
    # Hover effect
    logout_btn.bind(
        "<Enter>",
        lambda e: logout_btn.config(bg="#CF1020")
    )
    logout_btn.bind(
        "<Leave>",
        lambda e: logout_btn.config(bg="#ff0505")
    )

def show_user_buttons():
    clear_buttons()
    clear_textbox()
    
    btn1 = Button(
    main,
    text="Prediction",
    command=predict,
    **btn_style
    )
    btn1.place(relx=0.02,rely=0.25,relwidth=0.22,height=38)
    btn1.bind("<Enter>", on_enter)
    btn1.bind("<Leave>", on_leave)
    
    btn2 = Button(
    main,
    text="Batch Prediction",
    command=predict_folder,
    **btn_style
    )
    btn2.place(relx=0.02,rely=0.32,relwidth=0.22,height=38)
    btn2.bind("<Enter>", on_enter)
    btn2.bind("<Leave>", on_leave)
    
    logout_btn = Button(
    main,
    text="Logout",
    command=show_login_screen,
    font=font1,
    bg="#ff0505",
    fg="white",
    relief="flat",
    bd=0,
    cursor="hand2"
    )
    logout_btn.place(relx=0.02, rely=0.40, relwidth=0.15, height=38)
    # Hover effect
    logout_btn.bind(
        "<Enter>",
        lambda e: logout_btn.config(bg="#CF1020")
    )
    logout_btn.bind(
        "<Leave>",
        lambda e: logout_btn.config(bg="#ff0505")
    )
    

def show_login_screen():
    clear_buttons()
    clear_textbox()

    #Admin buttons
    adm1 = tk.Button(
        main,
        text="Admin Signup",
        command=lambda: signup("Admin"),
        **btn_style
    )
    adm1.place(relx=0.12, rely=0.85, relwidth=0.15)
    adm1.bind("<Enter>", on_enter)
    adm1.bind("<Leave>", on_leave)

    adm2 = tk.Button(
        main,
        text="Admin Login",
        command=lambda: login("Admin"),
        **btn_style
    )
    adm2.place(relx=0.32, rely=0.85, relwidth=0.15)
    adm2.bind("<Enter>", on_enter)
    adm2.bind("<Leave>", on_leave)

    #Farmer buttons
    farm1 = tk.Button(
        main,
        text="Farmer Signup",
        command=lambda: signup("Farmer"),
        **btn_style
    )
    farm1.place(relx=0.52, rely=0.85, relwidth=0.15)
    farm1.bind("<Enter>", on_enter)
    farm1.bind("<Leave>", on_leave)

    farm2 = tk.Button(
        main,
        text="Farmer Login",
        command=lambda: login("Farmer"),
        **btn_style
    )
    farm2.place(relx=0.72, rely=0.85, relwidth=0.15)
    farm2.bind("<Enter>", on_enter)
    farm2.bind("<Leave>", on_leave)
    
    #Exit button
    exit_btn = tk.Button(
        main,
        text="Exit",
        command=close,
        font=font1,
        bg="#ff0505",
        fg="white",
        relief="flat",
        bd=0,
        cursor="hand2"
    )
    exit_btn.place(relx=0.45, rely=0.93, relwidth=0.10)
    # Hover effect
    exit_btn.bind(
        "<Enter>",
        lambda e: exit_btn.config(bg="#CF1020")
    )
    exit_btn.bind(
        "<Leave>",
        lambda e: exit_btn.config(bg="#ff0505")
    )


# =============================
# Title
# =============================
title_font = ('Segoe UI', 20, 'bold')
title = Label(
    main,
    text="AI-Based Cauliflower Disease Detection System",
    font=title_font,
    # bg="#8fda44",
    bg="#8cc751",
    fg="#ffffff"
)
title.place(relx=0.5, rely=0.02, anchor='n', relwidth=1, height=60)


# =============================
# Output Text Box
# =============================

font_text = ('Segoe UI', 10)

text = Text(
    main,
    font=font_text,
    bg="#ffffff",        # soft mint white
    fg="black",
    bd=0,                  # remove ugly border
    relief="flat",
    padx=15,               # horizontal padding
    pady=15,               # vertical padding
    insertbackground="black",
    wrap='word',
    state='disabled'
)

scroll = Scrollbar(main, command=text.yview)

text.configure(yscrollcommand=scroll.set)

text.place(
    relx=0.40,
    rely=0.15,
    relwidth=0.55,
    relheight=0.60
)

scroll.place(
    relx=0.95,
    rely=0.15,
    relheight=0.60
)

show_login_screen()

main.mainloop()