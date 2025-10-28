# Rangkuman Lengkap: Advanced Computer Vision

**Program Studi Teknik Informatika S-2**  
**Mata Kuliah: Advanced Computer Vision**

**Data Mahasiswa**
**Nama : Daniel Giovanni Damara**
**NIM : 241012000118**
**Kelas : 03MKME002**

---

## Daftar Isi

1. [Pengantar](#pengantar)
2. [Visi Komputer Lanjutan: Dasar dan Penerapan](#1-visi-komputer-lanjutan-dasar-dan-penerapan)
3. [Konsep Dasar Citra Digital dan Manipulasi dengan Python](#2-konsep-dasar-citra-digital-dan-manipulasi-dengan-python)
4. [Filtering Citra Digital: Domain Spasial dan Frekuensi](#3-filtering-citra-digital-domain-spasial-dan-frekuensi)
5. [Jaringan Saraf Tiruan untuk Computer Vision](#4-jaringan-saraf-tiruan-untuk-computer-vision)
6. [Pembelajaran Mesin Lanjutan dalam Computer Vision](#5-pembelajaran-mesin-lanjutan-dalam-computer-vision)
7. [Convolutional Neural Network (CNN)](#6-convolutional-neural-network-cnn)
8. [Kesimpulan](#kesimpulan)
9. [Referensi](#referensi)

---

## Pengantar

Computer Vision atau visi komputer merupakan salah satu bidang yang paling dinamis dalam kecerdasan buatan dan ilmu komputer. Bidang ini berfokus pada pengembangan teknik-teknik yang memungkinkan komputer untuk "melihat" dan "memahami" konten visual seperti gambar dan video, mirip dengan cara sistem visual manusia bekerja.

Dokumen ini merangkum materi pembelajaran Advanced Computer Vision yang mencakup enam topik utama, mulai dari konsep dasar hingga implementasi teknik pembelajaran mesin modern seperti Convolutional Neural Networks (CNN). Setiap bagian dirancang untuk memberikan pemahaman teoritis yang kuat sekaligus keterampilan praktis dalam implementasi menggunakan Python.

---

## 1. Visi Komputer Lanjutan: Dasar dan Penerapan

### 1.1 Definisi dan Konsep Fundamental

Visi komputer adalah cabang ilmu komputer yang fokus pada pengembangan teknik untuk memungkinkan komputer memperoleh pemahaman tingkat tinggi dari gambar atau video digital. Berbeda dengan pengolahan citra yang berfokus pada transformasi gambar, visi komputer berusaha mengekstrak informasi bermakna dari data visual.

**Tahapan dalam Visi Komputer:**
- **Akuisisi Citra**: Proses mendapatkan citra digital dari dunia nyata melalui kamera atau sensor
- **Pengolahan Citra**: Manipulasi dan peningkatan kualitas citra digital
- **Analisis Citra**: Ekstraksi fitur dan informasi bermakna dari citra
- **Pemahaman Citra**: Interpretasi konten visual dan pengambilan keputusan

### 1.2 Sejarah dan Evolusi

Perkembangan visi komputer dapat dibagi menjadi beberapa era penting:

- **1960-an**: Awal mula visi komputer sebagai disiplin ilmu dengan fokus pada pengenalan pola
- **1970-an**: Pengembangan algoritma dasar untuk deteksi tepi dan segmentasi
- **1980-1990-an**: Penerapan metode statistik dan geometris
- **2000-an**: Kemajuan dalam pembelajaran mesin dan library open source seperti OpenCV
- **2010-sekarang**: Revolusi deep learning dengan CNN, R-CNN, YOLO

### 1.3 Aplikasi dalam Dunia Nyata

Visi komputer telah diterapkan dalam berbagai bidang:

**Kesehatan:**
- Diagnosis penyakit melalui analisis citra medis (X-ray, MRI, CT Scan)
- Deteksi kanker dari citra patologi
- Analisis retina untuk deteksi penyakit mata

**Transportasi:**
- Kendaraan otonom (self-driving cars)
- Sistem bantuan pengemudi (ADAS)
- Pemantauan lalu lintas dan pengenalan plat nomor

**Industri:**
- Inspeksi kualitas produk otomatis
- Pemantauan proses produksi
- Kontrol robot dalam perakitan

**Keamanan:**
- Sistem pengawasan dan pengenalan wajah
- Deteksi perilaku mencurigakan
- Analisis video forensik

### 1.4 Dasar Citra Digital

Citra digital merupakan representasi diskrit dari citra kontinu yang dapat diproses oleh komputer. Citra digital terdiri dari piksel (picture elements) yang disusun dalam format grid dua dimensi.

**Karakteristik Utama:**
- **Resolusi**: Jumlah piksel dalam citra (misalnya 1920×1080)
- **Kedalaman Warna**: Jumlah bit yang digunakan untuk merepresentasikan warna setiap piksel
- **Model Warna**: RGB, CMYK, HSV, Grayscale

### 1.5 Python untuk Visi Komputer

Python menjadi bahasa pemrograman pilihan untuk visi komputer karena:
- Sintaks yang mudah dipahami
- Ekosistem library yang kaya (OpenCV, PIL, scikit-image)
- Integrasi dengan teknologi machine learning
- Komunitas yang aktif

**Library Utama:**
- **OpenCV**: Pustaka komprehensif untuk computer vision
- **Pillow (PIL)**: Manipulasi citra dasar
- **NumPy**: Operasi matriks dan array
- **Matplotlib**: Visualisasi data dan citra

### 1.6 Operasi Dasar pada Citra

**Konversi Grayscale:**
```python
import cv2
img = cv2.imread('gambar.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**Histogram Analysis:**
Histogram menggambarkan distribusi intensitas piksel dalam citra dan berguna untuk:
- Analisis kontras citra
- Deteksi under/over exposure
- Segmentasi berbasis threshold

**Thresholding:**
Teknik untuk mengubah citra grayscale menjadi citra biner dengan memisahkan objek dari latar belakang.

---

## 2. Konsep Dasar Citra Digital dan Manipulasi dengan Python

### 2.1 Representasi Citra Digital

Citra digital dapat didefinisikan sebagai fungsi f(x,y) dimana x dan y adalah koordinat spasial, dan f adalah intensitas atau tingkat keabuan pada koordinat tersebut. Dalam implementasi praktis, citra digital direpresentasikan sebagai matriks M×N.

**Interpretasi Matematis:**
- Untuk citra grayscale: Matriks 2D dengan nilai 0-255
- Untuk citra RGB: Tensor 3D dengan dimensi (tinggi × lebar × 3)

### 2.2 Karakteristik Citra Digital

#### 2.2.1 Resolusi

**Resolusi Spasial:**
- Mengacu pada jumlah piksel per satuan panjang (ppi - pixel per inch)
- Resolusi 300 ppi adalah standar minimum untuk pencetakan berkualitas tinggi

**Resolusi Piksel:**
- Jumlah total piksel dalam citra (lebar × tinggi)
- Contoh: 1920×1080, 4K (3840×2160)

**Resolusi Radiometrik:**
- Jumlah bit untuk menyimpan informasi warna per piksel
- 8-bit grayscale: 256 tingkat keabuan
- 24-bit RGB: 16,7 juta warna

#### 2.2.2 Format File

**Format Lossless:**
- PNG: Mendukung transparansi, ideal untuk grafik web
- TIFF: Format fleksibel untuk pengarsipan
- BMP: Tanpa kompresi, ukuran file besar

**Format Lossy:**
- JPEG: Kompresi tinggi, ideal untuk foto
- WebP: Format modern dengan kompresi lebih baik
- JPEG 2000: Penyempurnaan JPEG

### 2.3 Transformasi Dasar Citra

#### 2.3.1 Resizing (Pengubahan Ukuran)

Metode interpolasi untuk resizing:
- **Nearest Neighbor**: Cepat tapi kualitas rendah
- **Bilinear**: Keseimbangan antara kecepatan dan kualitas
- **Bicubic**: Kualitas tinggi, lebih lambat
- **Lanczos**: Kualitas terbaik untuk downsampling

```python
import cv2
img = cv2.imread('gambar.jpg')
resized = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
```

#### 2.3.2 Cropping (Pemotongan)

```python
# Menggunakan array slicing
cropped = img[y_start:y_end, x_start:x_end]
```

#### 2.3.3 Rotasi

```python
height, width = img.shape[:2]
center = (width // 2, height // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
```

### 2.4 Manipulasi Kontras dan Kecerahan

Fungsi untuk mengubah kontras dan kecerahan:
```python
adjusted = np.clip(image * contrast + brightness, 0, 255).astype(np.uint8)
```

- **Contrast**: Faktor pengali (1.0 = tidak berubah)
- **Brightness**: Nilai yang ditambahkan (-255 sampai 255)

### 2.5 Konversi Model Warna

Model warna yang umum digunakan:

**RGB (Red-Green-Blue):**
- Model aditif untuk layar digital
- Setiap piksel terdiri dari 3 komponen: R, G, B

**HSV (Hue-Saturation-Value):**
- Memisahkan informasi warna dari intensitas
- Berguna untuk segmentasi berbasis warna

**Grayscale:**
- Representasi dengan satu kanal intensitas
- Konversi: Gray = 0.299R + 0.587G + 0.114B

**LAB:**
- L: Lightness, a: green-red, b: blue-yellow
- Perceptually uniform color space

### 2.6 Analisis Histogram

Histogram menggambarkan distribusi intensitas piksel dalam citra:

```python
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
```

**Aplikasi Histogram:**
- Deteksi under/over exposure
- Histogram equalization untuk peningkatan kontras
- Analisis distribusi warna

### 2.7 Ekstraksi Fitur Dasar

**Deteksi Tepi:**
- **Sobel**: Mendeteksi tepi berdasarkan gradien intensitas
- **Canny**: Multi-stage algorithm untuk deteksi tepi optimal
- **Laplacian**: Deteksi tepi berbasis turunan kedua

**Local Binary Pattern (LBP):**
- Ekstraksi tekstur lokal
- Robust terhadap perubahan iluminasi
- Aplikasi: Pengenalan wajah, analisis tekstur

---

## 3. Filtering Citra Digital: Domain Spasial dan Frekuensi

### 3.1 Konsep Dasar Filtering

Filtering citra digital adalah teknik pemrosesan yang bertujuan untuk memodifikasi atau meningkatkan citra dengan menekankan atau menekan informasi tertentu. Filtering dapat dilakukan pada:
- **Domain Spasial**: Manipulasi langsung pada piksel
- **Domain Frekuensi**: Setelah transformasi Fourier

### 3.2 Filtering Domain Spasial

#### 3.2.1 Konvolusi

Konvolusi adalah operasi fundamental dalam filtering spasial yang melibatkan kernel (filter) dan citra input:

```
g(x,y) = Σ Σ h(s,t) × f(x-s, y-t)
```

**Komponen Utama:**
- **Kernel/Mask/Filter**: Matriks kecil (3×3, 5×5, 7×7)
- **Padding**: Menangani piksel tepi
- **Stride**: Jarak pergeseran kernel

#### 3.2.2 Low-Pass Filter (Smoothing)

Low-pass filter menekan komponen frekuensi tinggi (detail) dan mempertahankan frekuensi rendah.

**Average Filter:**
```python
kernel_size = 5
average_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
average_filtered = cv2.filter2D(img, -1, average_kernel)
```

**Gaussian Filter:**
```python
gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
```

**Aplikasi:**
- Pengurangan noise
- Pra-pemrosesan untuk deteksi tepi
- Peningkatan hasil segmentasi

#### 3.2.3 High-Pass Filter (Sharpening)

High-pass filter menekankan detail dan tepi dengan mempertahankan komponen frekuensi tinggi.

**Sobel Filter:**
```python
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
```

**Laplacian Filter:**
```python
laplacian = cv2.Laplacian(img, cv2.CV_64F)
```

**Aplikasi:**
- Deteksi tepi dan kontur
- Peningkatan detail
- Segmentasi objek

#### 3.2.4 Bilateral Filter

Filter non-linear yang mengurangi noise sambil mempertahankan tepi:

```python
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
```

**Keunggulan:**
- Menghaluskan area homogen
- Mempertahankan tepi yang tajam
- Hasil yang terlihat natural

### 3.3 Filtering Domain Frekuensi

#### 3.3.1 Transformasi Fourier

Transformasi Fourier mengubah representasi citra dari domain spasial ke domain frekuensi, memungkinkan analisis komponen frekuensi.

**Discrete Fourier Transform (DFT):**
```python
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
```

**Karakteristik:**
- Frekuensi rendah: Area homogen (pusat spektrum)
- Frekuensi tinggi: Tepi, detail, noise (tepi spektrum)
- Komponen DC: Nilai rata-rata intensitas

#### 3.3.2 Jenis Filter Frekuensi

**Ideal Low-Pass Filter:**
- Memotong semua frekuensi di atas cutoff D₀
- Transisi tajam, dapat menyebabkan ringing artifacts

**Butterworth Low-Pass Filter:**
- Transisi halus antara pass dan stop band
- Parameter n mengontrol ketajaman transisi

**Gaussian Low-Pass Filter:**
- Transisi sangat halus dengan kurva Gaussian
- Tidak ada ringing artifacts

**High-Pass Filter:**
- Kebalikan dari low-pass filter
- Menekan frekuensi rendah, mempertahankan detail

#### 3.3.3 Implementasi Filter Frekuensi

```python
# Membuat mask Gaussian low-pass
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols, 2), np.float32)
r = 30  # Radius cutoff

center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask[:,:,0] = mask_area.astype(np.float32)
mask[:,:,1] = mask_area.astype(np.float32)

# Terapkan filter
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
```

### 3.4 Perbandingan Filtering Spasial dan Frekuensi

| Aspek | Filtering Spasial | Filtering Frekuensi |
|-------|------------------|---------------------|
| **Kompleksitas** | Sederhana untuk kernel kecil | Memerlukan FFT dan IFFT |
| **Efisiensi** | Efisien untuk filter kecil | Efisien untuk filter besar |
| **Fleksibilitas** | Terbatas pada ukuran kernel | Dapat dirancang dengan fleksibel |
| **Interpretasi** | Intuitif dan mudah dibayangkan | Abstrak, memerlukan pemahaman frekuensi |

### 3.5 Studi Kasus

**Peningkatan Kualitas Citra Medis:**
1. Gunakan Gaussian filter untuk mengurangi noise
2. Terapkan unsharp masking untuk meningkatkan kontras
3. Kombinasikan hasil untuk citra optimal

**Deteksi Tepi untuk Segmentasi:**
1. Pra-pemrosesan dengan Gaussian blur
2. Deteksi tepi dengan Canny
3. Post-processing dengan operasi morfologi
4. Segmentasi objek menggunakan kontur

---

## 4. Jaringan Saraf Tiruan untuk Computer Vision

### 4.1 Konsep Dasar JST

Jaringan Saraf Tiruan (JST) atau Artificial Neural Network (ANN) adalah model komputasi yang terinspirasi dari struktur dan fungsi jaringan saraf biologis dalam otak manusia.

**Karakteristik Utama:**
- Kemampuan belajar dari data (learning)
- Kemampuan menggeneralisasi (generalization)
- Toleransi terhadap kesalahan (fault tolerance)
- Komputasi paralel

### 4.2 Struktur Dasar JST

**Komponen Neuron Buatan:**
1. **Input (x)**: Sinyal masukan dari neuron lain atau lingkungan
2. **Bobot (w)**: Parameter yang dapat dilatih
3. **Fungsi Penjumlahan**: Σ(wᵢ × xᵢ) + b
4. **Fungsi Aktivasi**: Menentukan output neuron
5. **Output (y)**: Hasil keluaran neuron

**Lapisan dalam JST:**
- **Input Layer**: Menerima data input
- **Hidden Layer**: Memproses informasi
- **Output Layer**: Menghasilkan prediksi akhir

### 4.3 Fungsi Aktivasi

Fungsi aktivasi mengenalkan non-linearitas ke dalam model:

**Sigmoid:**
```
σ(x) = 1 / (1 + e^(-x))
```
- Output: 0 hingga 1
- Aplikasi: Output biner, hidden layer (jarang digunakan karena vanishing gradient)

**Tanh:**
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Output: -1 hingga 1
- Lebih baik dari sigmoid untuk hidden layer

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```
- Paling populer untuk CNN
- Sederhana dan efisien
- Mengatasi vanishing gradient

**Leaky ReLU:**
```
f(x) = max(αx, x), α = 0.01
```
- Mengatasi dying ReLU problem

**Softmax:**
```
softmax(xᵢ) = e^(xᵢ) / Σ e^(xⱼ)
```
- Output sebagai probabilitas
- Digunakan pada layer output untuk klasifikasi multi-kelas

### 4.4 Arsitektur JST untuk Computer Vision

**Multilayer Perceptron (MLP):**
- Jaringan feedforward sederhana
- Setiap neuron terhubung dengan semua neuron di lapisan sebelumnya
- Cocok untuk klasifikasi citra sederhana dengan dimensi rendah

**Convolutional Neural Network (CNN):**
- Dirancang khusus untuk data citra
- Menggunakan operasi konvolusi untuk ekstraksi fitur spasial
- Mengeksploitasi lokalitas spasial dan invariansi translasi

**Recurrent Neural Network (RNN):**
- Memiliki koneksi feedback
- Untuk analisis video atau sequence of images

### 4.5 Proses Pembelajaran JST

#### 4.5.1 Algoritma Backpropagation

Backpropagation adalah algoritma pembelajaran terawasi yang terdiri dari 4 tahap:

**1. Forward Propagation:**
- Input data dimasukkan ke jaringan
- Data diproses melalui semua lapisan
- Menghasilkan output prediksi

**2. Kalkulasi Error:**
- Menghitung perbedaan antara output prediksi dan target
- Menggunakan fungsi loss (MSE, cross-entropy)

**3. Backward Propagation:**
- Menghitung gradien error terhadap setiap bobot
- Mempropagasi error dari output ke input

**4. Pembaruan Bobot:**
- w_new = w_old - learning_rate × gradien
- Menggunakan optimizer (SGD, Adam, dll.)

### 4.6 Fungsi Loss dan Optimizer

**Fungsi Loss:**
- **Mean Squared Error (MSE)**: Untuk regresi
- **Binary Cross-Entropy**: Untuk klasifikasi biner
- **Categorical Cross-Entropy**: Untuk klasifikasi multi-kelas
- **Sparse Categorical Cross-Entropy**: Untuk label integer

**Optimizer:**
- **SGD (Stochastic Gradient Descent)**: Sederhana, konvergensi lambat
- **Adam**: Adaptif, konvergensi cepat, paling populer
- **RMSprop**: Baik untuk RNN
- **Adagrad**: Adaptif untuk data jarang

### 4.7 Regularisasi dan Pencegahan Overfitting

**Dropout:**
- Menonaktifkan sebagian neuron secara acak selama pelatihan
- Probabilitas umum: 0.5 (50% neuron dimatikan)
- Mencegah co-adaptation neuron

**Batch Normalization:**
- Menormalkan output layer
- Mempercepat pelatihan dan meningkatkan stabilitas
- Memberikan efek regularisasi

**Early Stopping:**
- Menghentikan pelatihan ketika validation loss mulai meningkat
- Mencegah overfitting pada data latih

**Data Augmentation:**
- Meningkatkan dataset dengan transformasi
- Rotasi, flipping, scaling, perubahan kecerahan
- Memperkaya data latih tanpa mengumpulkan data baru

### 4.8 Implementasi JST dengan Keras

```python
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist

# 1. Persiapan Data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 2. Definisi Model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 3. Kompilasi Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Pelatihan Model
history = model.fit(
    train_images, train_labels,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# 5. Evaluasi Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Akurasi pada test set: {test_acc:.4f}')
```

### 4.9 Dataset untuk Computer Vision

**MNIST:**
- 70.000 citra digit tulisan tangan (0-9)
- Format grayscale 28×28 piksel
- "Hello World" dalam computer vision

**CIFAR-10/100:**
- CIFAR-10: 60.000 citra berwarna 32×32 dalam 10 kelas
- CIFAR-100: 100 kelas dengan 600 citra per kelas

**ImageNet:**
- 14 juta citra dalam 20.000+ kategori
- Subset ILSVRC: 1,2 juta citra dalam 1.000 kelas

**COCO (Common Objects in Context):**
- 330.000 citra dengan 1,5 juta instance objek
- 80 kategori objek
- Untuk deteksi, segmentasi, dan captioning

---

## 5. Pembelajaran Mesin Lanjutan dalam Computer Vision

### 5.1 Definisi Pembelajaran Mesin

Pembelajaran mesin adalah cabang dari kecerdasan buatan yang fokus pada pengembangan algoritma dan model statistik yang memungkinkan komputer "belajar" dari data tanpa perlu diprogram secara eksplisit.

**Karakteristik:**
- Kemampuan adaptasi terhadap data baru
- Peningkatan performa seiring pertambahan data
- Generalisasi dari contoh-contoh yang dipelajari

### 5.2 Jenis-Jenis Pembelajaran Mesin

#### 5.2.1 Pembelajaran Terawasi (Supervised Learning)

Algoritma belajar dari data berlabel untuk memprediksi output yang diinginkan.

**Komponen:**
- Data pelatihan: {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
- x: vektor fitur, y: label target
- Model mempelajari fungsi f sehingga f(x) ≈ y

**Tipe Masalah:**
- **Klasifikasi**: Memprediksi kategori diskrit
- **Regresi**: Memprediksi nilai kontinu

**Algoritma Populer:**
1. **K-Nearest Neighbors (KNN)**: Klasifikasi berdasarkan mayoritas kelas k tetangga terdekat
2. **Support Vector Machine (SVM)**: Mencari hyperplane optimal yang memisahkan kelas
3. **Decision Tree & Random Forest**: Struktur pohon untuk keputusan berdasarkan fitur
4. **Neural Networks**: Jaringan neuron untuk masalah kompleks

#### 5.2.2 Pembelajaran Tidak Terawasi (Unsupervised Learning)

Algoritma belajar dari data tanpa label untuk menemukan pola atau struktur tersembunyi.

**Tipe Masalah:**
- **Clustering**: Mengelompokkan data berdasarkan kesamaan
- **Reduksi Dimensi**: Mengurangi jumlah fitur sambil mempertahankan informasi
- **Deteksi Anomali**: Mengidentifikasi outlier atau pola tidak normal
- **Pembelajaran Representasi**: Menemukan representasi yang lebih baik

**Algoritma Populer:**
1. **K-Means Clustering**: Membagi data menjadi k cluster berdasarkan jarak
2. **Hierarchical Clustering**: Membangun hirarki cluster
3. **PCA (Principal Component Analysis)**: Reduksi dimensi dengan mempertahankan varians maksimum
4. **Autoencoders**: Neural network untuk pembelajaran representasi

#### 5.2.3 Pembelajaran Penguatan (Reinforcement Learning)

Agen belajar melalui interaksi dengan lingkungan untuk memaksimalkan reward kumulatif.

**Komponen:**
- Agen: Entitas yang belajar dan membuat keputusan
- Lingkungan: Dunia tempat agen berinteraksi
- State: Representasi situasi lingkungan
- Action: Keputusan yang diambil agen
- Reward: Sinyal umpan balik dari lingkungan

### 5.3 Metrik Evaluasi

#### 5.3.1 Metrik Klasifikasi

**Akurasi:**
```
Akurasi = (TP + TN) / (TP + TN + FP + FN)
```

**Presisi:**
```
Presisi = TP / (TP + FP)
```

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
```

**F1-Score:**
```
F1 = 2 × (Presisi × Recall) / (Presisi + Recall)
```

**ROC-AUC:**
- Area di bawah kurva Receiver Operating Characteristic
- Mengukur kemampuan model membedakan kelas

#### 5.3.2 Metrik Regresi

- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² (Coefficient of Determination)**: Proporsi varians yang dijelaskan

#### 5.3.3 Metrik Clustering

- **Silhouette Score**: Mengukur kohesi cluster
- **Davies-Bouldin Index**: Rasio dispersi intra-cluster dengan inter-cluster
- **Calinski-Harabasz Index**: Rasio dispersi antar-cluster dengan dalam-cluster

### 5.4 Implementasi dengan Scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dan split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Standardisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

### 5.5 Perbandingan Supervised vs Unsupervised

| Aspek | Supervised Learning | Unsupervised Learning |
|-------|--------------------|-----------------------|
| **Data Input** | Data berlabel | Data tidak berlabel |
| **Tujuan** | Memprediksi output | Menemukan pola tersembunyi |
| **Evaluasi** | Akurasi, F1-score | Silhouette score, interpretasi |
| **Aplikasi CV** | Klasifikasi, deteksi objek | Clustering, reduksi dimensi |
| **Tantangan** | Membutuhkan banyak label | Evaluasi sulit, interpretasi subjektif |

### 5.6 Transfer Learning

Transfer learning memanfaatkan model pre-trained pada dataset besar (seperti ImageNet) untuk tugas baru dengan data terbatas.

**Pendekatan:**
1. **Feature Extraction**: Gunakan CNN pre-trained sebagai ekstractor fitur
2. **Fine-tuning**: Latih ulang beberapa lapisan terakhir
3. **Full Training**: Gunakan bobot pre-trained sebagai inisialisasi

**Keunggulan:**
- Mengurangi kebutuhan data pelatihan hingga 10-100 kali
- Mempercepat konvergensi
- Sering menghasilkan performa lebih baik

### 5.7 Pengenalan Objek dengan Machine Learning

Pengenalan objek menggunakan pipeline berikut:

**1. Ekstraksi Fitur:**
- Metode tradisional: SIFT, HOG, SURF
- Deep learning: Pembelajaran otomatis fitur melalui CNN

**2. Arsitektur Model:**
- R-CNN, Fast R-CNN, Faster R-CNN untuk deteksi objek
- YOLO dan SSD untuk deteksi real-time
- Mask R-CNN untuk segmentasi instans

**3. Post-processing:**
- Non-Maximum Suppression (NMS)
- Filtering berdasarkan confidence threshold

---

## 6. Convolutional Neural Network (CNN)

### 6.1 Pendahuluan CNN

Convolutional Neural Network (CNN) telah menjadi tulang punggung revolusi deep learning dalam computer vision. Model ini terinspirasi dari struktur visual cortex mamalia dan dirancang khusus untuk mengolah data grid seperti citra.

**Keunggulan CNN:**
- Mampu mendeteksi fitur spasial secara hierarkis
- Mengurangi jumlah parameter melalui weight sharing
- Invariant terhadap translasi objek
- Ekstraksi fitur otomatis

### 6.2 Arsitektur Dasar CNN

Struktur CNN terdiri dari rangkaian lapisan:

**1. Input Layer:**
- Menerima citra input dalam format tensor
- Dimensi: [tinggi × lebar × saluran warna]

**2. Convolutional Layers:**
- Mengekstrak fitur spasial melalui operasi konvolusi
- Menggunakan filter/kernel yang dapat dilatih

**3. Pooling Layers:**
- Mereduksi dimensi spasial
- Meningkatkan invariansi terhadap translasi

**4. Fully Connected Layers:**
- Mengintegrasikan semua fitur
- Klasifikasi akhir

### 6.3 Convolution Layer

#### 6.3.1 Konsep Dasar

Convolution layer melakukan operasi konvolusi antara input dan kernel untuk menghasilkan feature map. Kernel yang berbeda mendeteksi fitur yang berbeda: tepi, tekstur, atau pola kompleks.

#### 6.3.2 Parameter Kritis

**Filter Size (Ukuran Kernel):**
- Umum: 3×3, 5×5, atau 7×7
- Semakin besar ukuran, semakin luas area reseptif

**Stride:**
- Jumlah piksel pergeseran filter
- Stride besar mengurangi dimensi output

**Padding:**
- Valid padding: Tanpa padding
- Same padding: Mempertahankan dimensi spasial

**Jumlah Filter:**
- Menentukan jumlah feature map
- Lebih banyak filter → lebih banyak fitur terdeteksi

### 6.4 Activation Function dalam CNN

**ReLU (Rectified Linear Unit):**
```python
f(x) = max(0, x)
```
- Paling umum digunakan
- Mengatasi vanishing gradient
- Komputasi sederhana

**Leaky ReLU:**
```python
f(x) = max(αx, x), α = 0.01
```
- Mengatasi dying ReLU problem
- Memungkinkan gradien kecil saat x < 0

### 6.5 Pooling Layer

**Max Pooling:**
- Mengambil nilai maksimum dari setiap region
- Paling umum digunakan
- Memperkuat fitur dominan

**Average Pooling:**
- Menghitung rata-rata dari setiap region
- Mempertahankan konteks spasial
- Memperhalus representasi

**Parameter:**
- Ukuran pool: Umumnya 2×2
- Stride: Biasanya sama dengan ukuran pool

### 6.6 Fully Connected Layer dan Output

**Flattening:**
- Mengubah feature maps 3D menjadi vektor 1D
- Contoh: 7×7×64 → 3136 elemen

**Fully Connected Layers:**
- Memproses vektor fitur
- Mengintegrasikan informasi spasial dan semantik
- Aktivasi: ReLU

**Output Layer:**
- Jumlah neuron = jumlah kelas
- Aktivasi: Softmax untuk multi-kelas
- Menghasilkan probabilitas untuk setiap kelas

### 6.7 Teknik Regularisasi

**Dropout:**
- Menonaktifkan sebagian neuron secara acak
- Probabilitas umum: 0.5 pada fully connected layers
- Mencegah overfitting

**Batch Normalization:**
- Menormalkan output layer
- Mempercepat pelatihan
- Meningkatkan stabilitas

**Data Augmentation:**
- Transformasi citra: rotasi, flipping, scaling
- Perubahan kecerahan dan kontras
- Memperkaya data latih

### 6.8 Arsitektur CNN Populer

**1. LeNet (1998):**
- Pelopor CNN modern oleh Yann LeCun
- Untuk pengenalan digit tulisan tangan
- Arsitektur sederhana: 2 conv + 2 pooling + 3 FC layers

**2. AlexNet (2012):**
- Memenangkan ILSVRC 2012
- 5 conv layers + 3 FC layers
- Memperkenalkan ReLU, dropout, dan GPU paralel

**3. VGG (2014):**
- Arsitektur mendalam dan seragam
- Menggunakan konvolusi 3×3 secara konsisten
- VGG-16 dan VGG-19 masih populer

**4. ResNet (2015):**
- Memperkenalkan skip connection
- Mengatasi vanishing gradient pada jaringan sangat dalam
- Memungkinkan pelatihan hingga ratusan lapisan

**5. Inception/GoogLeNet:**
- Menggunakan inception modules
- Efisien dalam parameter
- Paralel processing dengan berbagai ukuran kernel

**6. EfficientNet:**
- Optimasi scaling seimbang
- Efisien dalam komputasi
- State-of-the-art performance

### 6.9 Implementasi CNN dengan Keras

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model():
    model = models.Sequential([
        # Layer konvolusi pertama
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Layer konvolusi kedua
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Layer konvolusi ketiga
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten dan fully connected layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Kompilasi model
model = build_cnn_model()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)
```

### 6.10 Visualisasi Feature Maps

Visualisasi feature maps membantu memahami apa yang "dilihat" oleh CNN:

```python
def visualize_feature_maps(model, img):
    layer_outputs = [layer.output for layer in model.layers 
                     if isinstance(layer, layers.Conv2D)]
    activation_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=layer_outputs
    )
    activations = activation_model.predict(img[np.newaxis, ...])
    
    # Plot feature maps untuk setiap layer
    for i, activation in enumerate(activations):
        # Visualisasi code...
        pass
```

**Observasi:**
- Lapisan awal: Mendeteksi fitur sederhana (tepi, warna)
- Lapisan dalam: Mendeteksi fitur kompleks (tekstur, pola)

### 6.11 Transfer Learning dengan CNN

```python
from tensorflow.keras.applications import VGG16

# Load model pre-trained
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Tambahkan layer kustom
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 6.12 Analisis Performa

**Efek Jumlah Filter:**
- Peningkatan dari 32 ke 64 filter: +5-7% akurasi
- Peningkatan dari 64 ke 128 filter: +2% akurasi (marginal improvement)
- Trade-off: Akurasi vs waktu komputasi

**Efek Pooling Layer:**
- Max pooling: Akurasi terbaik (+4% vs tanpa pooling)
- Average pooling: Sedikit lebih baik dari tanpa pooling
- Global pooling: Mengurangi akurasi (kehilangan info spasial)

### 6.13 Tantangan dan Limitasi

**1. Kebutuhan Data Besar:**
- CNN membutuhkan banyak data untuk generalisasi
- Solusi: Transfer learning, data augmentation

**2. Komputasi Intensif:**
- Pelatihan membutuhkan GPU untuk model besar
- Solusi: Model compression, quantization

**3. Sensitif terhadap Hyperparameter:**
- Learning rate, jumlah filter, arsitektur
- Solusi: Hyperparameter tuning, AutoML

**4. Interpretabilitas Rendah:**
- CNN sebagai "black box"
- Solusi: Visualisasi feature maps, Grad-CAM, LIME

### 6.14 Tren Terkini

**1. Vision Transformers (ViT):**
- Adaptasi arsitektur Transformer untuk visi
- Hasil menjanjikan untuk dataset besar
- Alternatif untuk CNN

**2. EfficientNet dan Neural Architecture Search:**
- Optimasi arsitektur otomatis
- Scaling seimbang: depth, width, resolution

**3. Self-Supervised Learning:**
- Belajar dari data tanpa label
- Mengurangi ketergantungan pada data berlabel

**4. Lightweight Models:**
- MobileNet, ShuffleNet untuk perangkat mobile
- Model compression dan quantization

---

## Kesimpulan

Advanced Computer Vision merupakan bidang yang sangat dinamis dan terus berkembang dengan pesat. Dari dasar-dasar pemrosesan citra digital hingga arsitektur deep learning modern seperti CNN, setiap konsep membangun fondasi yang kuat untuk memahami dan mengimplementasikan solusi computer vision di dunia nyata.

### Poin-Poin Kunci:

**1. Fondasi yang Kuat:**
- Pemahaman tentang citra digital dan operasi dasar sangat penting
- Python dengan library seperti OpenCV, NumPy, dan TensorFlow menjadi toolkit standar
- Filtering dan transformasi citra adalah preprocessing yang esensial

**2. Pembelajaran Mesin:**
- Supervised learning cocok untuk tugas dengan data berlabel
- Unsupervised learning berguna untuk eksplorasi data
- Transfer learning mempercepat pengembangan dengan data terbatas

**3. Deep Learning dengan CNN:**
- CNN adalah arsitektur terbaik untuk pemrosesan citra
- Komponen utama: convolution, pooling, dan fully connected layers
- Regularisasi penting untuk mencegah overfitting

**4. Aplikasi Praktis:**
- Computer vision diterapkan di berbagai bidang: kesehatan, transportasi, industri, keamanan
- Dataset publik (MNIST, CIFAR, ImageNet) memudahkan pembelajaran dan benchmarking
- Transfer learning memungkinkan penerapan praktis dengan sumber daya terbatas

### Arah Masa Depan:

1. **Vision Transformers**: Arsitektur baru yang menantang dominasi CNN
2. **Self-Supervised Learning**: Mengurangi ketergantungan pada data berlabel
3. **Edge AI**: Implementasi computer vision pada perangkat dengan sumber daya terbatas
4. **Explainable AI**: Meningkatkan interpretabilitas model deep learning
5. **Multimodal Learning**: Integrasi visi dengan modalitas lain (teks, audio)

Computer vision akan terus menjadi area penelitian yang aktif dan penting, dengan aplikasi yang semakin luas dalam kehidupan sehari-hari, dari kendaraan otonom hingga diagnosis medis, dari augmented reality hingga robotika.

---

## Referensi

### Buku

1. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Chollet, F. (2021). *Deep Learning with Python, Second Edition*. Manning Publications.
5. Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2021). *Dive into Deep Learning*.
6. Solem, J. E. (2012). *Programming Computer Vision with Python*. O'Reilly Media.
7. Bradski, G., & Kaehler, A. (2008). *Learning OpenCV*. O'Reilly Media.

### Paper dan Jurnal

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.
4. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*.
5. Szegedy, C., et al. (2015). Going deeper with convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 1-9.
6. Buades, A., Coll, B., & Morel, J. M. (2005). A non-local algorithm for image denoising. *IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2, 60-65.
7. Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. *Sixth International Conference on Computer Vision*, 839-846.

### Dokumentasi Online

1. OpenCV Documentation: https://docs.opencv.org/
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Keras Documentation: https://keras.io/
4. PyTorch Documentation: https://pytorch.org/docs/
5. Pillow Documentation: https://pillow.readthedocs.io/
6. NumPy Documentation: https://numpy.org/doc/
7. Scikit-learn Documentation: https://scikit-learn.org/
8. Scikit-image Documentation: https://scikit-image.org/
9. Matplotlib Documentation: https://matplotlib.org/

### Tutorial dan Kursus Online

1. PyImageSearch: https://www.pyimagesearch.com/
2. LearnOpenCV: https://learnopencv.com/
3. Coursera: "Deep Learning Specialization" by Andrew Ng
4. Coursera: "Convolutional Neural Networks" by deeplearning.ai
5. Fast.ai: Practical Deep Learning for Coders
6. Udacity: Computer Vision Nanodegree
7. edX: "Deep Learning Fundamentals" by IBM
8. Stanford CS231n: Convolutional Neural Networks for Visual Recognition

### Dataset

1. MNIST: http://yann.lecun.com/exdb/mnist/
2. CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html
3. ImageNet: http://www.image-net.org/
4. COCO: https://cocodataset.org/
5. Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/
6. Open Images Dataset: https://storage.googleapis.com/openimages/web/index.html
7. Labeled Faces in the Wild: http://vis-www.cs.umass.edu/lfw/

### Repositori GitHub

1. OpenCV: https://github.com/opencv/opencv
2. TensorFlow: https://github.com/tensorflow/tensorflow
3. PyTorch: https://github.com/pytorch/pytorch
4. Keras: https://github.com/keras-team/keras
5. Computer Vision Awesome List: https://github.com/jbhuang0604/awesome-computer-vision

---

## Tentang Dokumen

**Dokumen:** Rangkuman Advanced Computer Vision  
**Program Studi:** Teknik Informatika S-2  
**Mata Kuliah:** Advanced Computer Vision  
**Topik Bahasan:**
- Visi Komputer Lanjutan
- Citra Digital dan Manipulasi
- Filtering Citra Digital
- Jaringan Saraf Tiruan
- Pembelajaran Mesin Lanjutan
- Convolutional Neural Network

**Disclaimer:** Dokumen ini disusun sebagai rangkuman materi pembelajaran dan referensi akademis. Untuk implementasi praktis, selalu rujuk dokumentasi resmi library dan paper terkini.

---

**© 2023 - Advanced Computer Vision Study Materials**  
*Disusun untuk keperluan pembelajaran dan referensi akademis*
