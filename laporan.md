*Nama:* Dila Fadilatu Nisa
*NIM:* 122430135 
*Mata Kuliah:* Kecerdasan Buatan  

---

# 1. Uji Coba Arsitektur Dasar
Langkah awal yang saya lakukan adalah menjalankan keseluruhan arsitektur dasar dari repository asli untuk memastikan bahwa sistem dapat berfungsi dengan baik pada perangkat saya serta untuk mengamati tingkat akurasi yang dihasilkan. Proses ini melibatkan eksekusi tiga file utama, yaitu:
- datareader.py
- model.py
- train.py

Setelah menjalankan ketiganya, saya memperoleh hasil Val Acc pada sekitaran  nilai 76.24%

# 2. Mencoba mengubah hyperparameter pada Arsitektur Dasar
Langkah selanjutnya, saya melakukan percobaan dengan memodifikasi beberapa hyperparameter pada arsitektur dasar untuk melihat pengaruhnya terhadap kinerja model. Tujuannya adalah untuk menemukan kombinasi parameter yang dapat meningkatkan akurasi dan stabilitas proses pelatihan. Adapun nilai-nilai hyperparameter yang digunakan adalah sebagai berikut:
### Hyperparameter
- EPOCHS = 32 
- BATCH_SIZE = 16 
- LEARNING_RATE = 0.001

Setelah menjalankannya saya memperoleh hasil Val Acc pada sekitaran nilai 79.82%

# 3. Mencoba Membuat Model ResNet dengan beberapa variasi hyperparameter yang berbeda-beda
Pada tahap ini, saya melakukan pengembangan dengan mengganti arsitektur dasar SimpleCNN menjadi arsitektur ResNet18 untuk mengevaluasi peningkatan performa model terhadap dataset yang sama. Tujuannya adalah untuk melihat apakah penggunaan residual learning dapat membantu model mengatasi masalah vanishing gradient serta meningkatkan akurasi pada proses pelatihan dan validasi.

Lalu selanjutnya saya mencoba model resnet dengan 3 variasi hyperparameter yang masing masing menghasilkan nilai Val Acc yang berbeda beda. Seluruh file ini diberi nama "train_baru" "train_baru_2" "train_baru_3".

### Hyperparameter train_baru
- EPOCHS = 16
- BATCH_SIZE = 16
- LEARNING_RATE = 0.0004
    → Hasil Val Acc sebesar 80.64%, menunjukkan peningkatan dibandingkan model dasar, meskipun belum mencapai performa optimal.

### Hyperparameter train_baru_2
- EPOCHS = 30
- BATCH_SIZE = 16
- LEARNING_RATE = 1e-4
    → Hasil Val Acc tertinggi sebesar 86.71%, yang menunjukkan bahwa pelatihan dengan jumlah epoch lebih banyak dan learning rate yang lebih kecil memberikan stabilitas serta kemampuan generalisasi yang lebih baik.

### Hyperparameter train_baru_2
- EPOCHS = 16
- BATCH_SIZE = 32
- LEARNING_RATE = 1e-4
    → Hasil Val Acc sebesar 79.88%, sedikit menurun dibandingkan dua percobaan lainnya, kemungkinan karena ukuran batch yang lebih besar membuat pembaruan bobot menjadi kurang sensitif terhadap variasi data mini-batch.

Dari ketiga percobaan tersebut, konfigurasi pada train_baru_2 terbukti memberikan hasil terbaik. Hal ini menunjukkan bahwa penggunaan learning rate yang kecil dengan jumlah epoch yang lebih banyak mampu memberikan keseimbangan antara stabilitas pelatihan dan kemampuan model dalam mengenali pola pada data validasi.

# 4. Mencoba Membuat Model DenseNet dengan Hyperparameter yang Berbeda-beda
Setelah memperoleh hasil yang cukup baik menggunakan *ResNet18*, saya melanjutkan eksperimen dengan menggunakan arsitektur *DenseNet* sebagai model utama berikutnya. Tujuan dari percobaan ini adalah untuk mengevaluasi sejauh mana konsep *dense connectivity* dapat meningkatkan performa model dalam mendeteksi pola fitur pada dataset yang sama.

Berbeda dengan ResNet yang menggunakan *shortcut connection* untuk menambahkan hasil dari lapisan sebelumnya, *DenseNet* menghubungkan setiap lapisan dengan semua lapisan berikutnya melalui mekanisme *feature concatenation*. Pendekatan ini memungkinkan setiap lapisan menerima masukan dari seluruh lapisan sebelumnya, sehingga memperkaya representasi fitur dan mengurangi risiko *vanishing gradient*.

Selain mengganti arsitektur, saya juga melakukan beberapa variasi *hyperparameter* untuk mencari konfigurasi terbaik yang memberikan keseimbangan antara akurasi dan efisiensi pelatihan. Setiap percobaan disimpan dalam file terpisah dengan nama `train_densenet.py`dan `train_densenet_2.py`.

### Hyperparameter train_densenet
- EPOCHS = 20
- BATCH_SIZE = 16
- LEARNING_RATE = 2e-4
- PRETRAINED = True
- FREEZE_BACKBONE_EPOCHS = 2
    → Hasil *Val Acc* sebesar *85.87%*, menunjukkan bahwa penggunaan *pretrained weights* membantu mempercepat konvergensi dan meningkatkan performa model dibandingkan pelatihan dari awal.

### Hyperparameter train_densenet_2
- EPOCHS = 30`
- BATCH_SIZE = 16`
- LEARNING_RATE = 2e-4`
- PRETRAINED = True`
- FREEZE_BACKBONE_EPOCHS = 2`
    → Hasil *Val Acc* tertinggi sebesar *87.21%*, memperlihatkan bahwa dengan jumlah *epoch* yang lebih banyak, model mampu mempelajari fitur dengan lebih mendalam tanpa mengalami overfitting, terutama karena lapisan awal tetap dibekukan selama beberapa *epoch* pertama untuk menjaga stabilitas pelatihan.

Secara keseluruhan, eksperimen ini menunjukkan bahwa *DenseNet* dengan *pretrained weights* memberikan hasil yang kompetitif dan bahkan melampaui performa *ResNet18* pada konfigurasi terbaik. Arsitektur yang padat dan efisien dalam memanfaatkan fitur antar-lapisan menjadikannya salah satu kandidat kuat untuk model klasifikasi citra medis yang memerlukan detail fitur tinggi.



