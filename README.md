# Laporan Proyek Machine Learning – Mario Valerian Rante Ta'dung

---

## 1. Domain Proyek

Stroke merupakan salah satu penyebab utama kematian dan disabilitas di seluruh dunia. Berdasarkan laporan World Health Organization (WHO), setiap tahunnya terjadi sekitar 15 juta kasus stroke secara global, dengan 5 juta orang meninggal dunia dan 5 juta lainnya mengalami disabilitas permanen akibat penyakit ini [1]. Di Indonesia sendiri, beban akibat stroke tidak kalah mengkhawatirkan. Berdasarkan data Riskesdas 2018 yang dirilis oleh Kementerian Kesehatan Republik Indonesia, prevalensi stroke nasional mencapai 10,9 per 1.000 penduduk, menjadikannya sebagai penyakit katastropik dengan beban ekonomi dan sosial yang sangat besar [2].

Tak hanya berdampak pada kesehatan individu, stroke juga memberikan beban ekonomi yang signifikan terhadap sistem kesehatan nasional. Menurut studi yang dipublikasikan oleh American Heart Association, total beban ekonomi akibat stroke di tingkat global diperkirakan mencapai lebih dari US$ 721 miliar, termasuk di dalamnya biaya pengobatan, rehabilitasi, serta kerugian produktivitas akibat disabilitas dan kematian dini [3]. Di Indonesia, berdasarkan data BPJS Kesehatan tahun 2022, stroke termasuk dalam lima penyakit dengan biaya klaim tertinggi, mencapai lebih dari Rp2,5 triliun dalam satu tahun [4]. Kondisi ini menunjukkan pentingnya upaya preventif dan deteksi dini terhadap risiko stroke, terutama pada kelompok usia produktif.

Faktor risiko stroke pada dasarnya dapat dikenali dan dimitigasi lebih awal. Beberapa faktor seperti usia lanjut, hipertensi, diabetes, hiperkolesterolemia, riwayat merokok, dan aktivitas fisik rendah telah lama diketahui sebagai penyebab utama meningkatnya risiko stroke [5]. Namun, dalam praktiknya, banyak kasus baru teridentifikasi saat pasien sudah berada dalam kondisi kronis atau pasca-serangan. Situasi ini disebabkan oleh rendahnya kesadaran masyarakat akan gejala awal stroke serta terbatasnya fasilitas skrining dini, terutama di wilayah-wilayah dengan akses kesehatan terbatas.

Seiring dengan perkembangan teknologi di bidang machine learning (ML), kini tersedia pendekatan yang lebih efisien dan berskala dalam mendeteksi risiko stroke. Algoritma pembelajaran mesin memungkinkan analisis data medis secara otomatis untuk mengidentifikasi pola-pola yang tidak selalu terlihat secara kasat mata. Dengan memanfaatkan variabel seperti umur, jenis kelamin, tekanan darah, kadar glukosa, status merokok, dan kondisi kesehatan terkait lainnya, model prediksi berbasis ML dapat memberikan perkiraan risiko stroke dengan akurasi yang cukup tinggi. Teknologi ini memungkinkan intervensi medis lebih dini, serta pengambilan keputusan yang lebih berbasis data oleh penyedia layanan kesehatan.

Penerapan model prediksi semacam ini sangat relevan di Indonesia, mengingat keterbatasan dalam pemerataan layanan kesehatan dan kurangnya tenaga medis spesialis di banyak daerah. Sistem prediksi yang terintegrasi dalam aplikasi atau sistem informasi kesehatan dapat membantu menyaring individu berisiko tinggi untuk kemudian dirujuk ke pemeriksaan lebih lanjut. Dengan demikian, pendekatan ini tidak hanya meningkatkan efisiensi layanan, tetapi juga membantu menekan angka kejadian stroke dan beban biaya yang menyertainya.

Melalui proyek ini, penulis mencoba membangun sebuah model prediktif sederhana namun efektif untuk memprediksi kemungkinan seseorang mengalami stroke berdasarkan data demografis dan medis. Dengan memanfaatkan dataset terbuka yang tersedia secara publik, diharapkan proyek ini dapat menjadi salah satu langkah awal dalam menjembatani pemanfaatan teknologi kecerdasan buatan dalam bidang kesehatan preventif di Indonesia.

---

## 2. Business Understanding

### 2.1 Problem Statements

- **Pernyataan Masalah 1**  
  Beban penyakit stroke yang tinggi di Indonesia belum diiringi dengan sistem deteksi dini yang efektif, sehingga banyak kasus baru teridentifikasi saat sudah dalam kondisi kronis.

- **Pernyataan Masalah 2**  
  Kurangnya alat prediksi risiko stroke yang mudah diakses dan akurat menyebabkan rendahnya efektivitas screening dini, menghambat upaya pencegahan dan pengurangan angka kematian.

### 2.2 Goals

- **Jawaban Pernyataan Masalah 1**  
  Mengembangkan model prediksi risiko stroke berbasis machine learning dengan akurasi yang memadai untuk membantu deteksi dini risiko stroke.

- **Jawaban Pernyataan Masalah 2**  
  Membuat sistem prediksi yang sederhana, efektif, dan dapat diintegrasikan ke dalam aplikasi kesehatan untuk meningkatkan kemampuan screening awal risiko stroke.

### 2.3 Solution Statements

- **Solution Statement 1**  
  Menerapkan algoritma Random Forest dan Gradient Boosting sebagai baseline model klasifikasi risiko stroke karena performa baik pada data tabular medis dan kemampuannya dalam interpretasi fitur.

- **Solution Statement 2**  
  Melakukan hyperparameter tuning dan feature engineering untuk meningkatkan akurasi dan performa model, serta menggunakan teknik balancing data jika diperlukan untuk menangani ketidakseimbangan kelas.

- **Solution Statement 3**  
  Mengevaluasi model menggunakan metrik akurasi, precision, recall, dan F1-score agar model dapat meminimalkan false negative, yang krusial dalam konteks prediksi risiko stroke.

---

## 3. Data Understanding

Dataset yang digunakan adalah **Stroke Prediction Dataset** dari Kaggle ([https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Terdiri atas 5.110 entri data dengan 2 kelas.

### 3.1 Deskripsi Dataset
| No. | Kolom               | Tipe Data | Jumlah Non-Null | Deskripsi                                                     |
| --- | ------------------- | --------- | --------------- | ------------------------------------------------------------- |
| 1   | `id`                | int64     | 5110            | ID unik untuk setiap data pasien                              |
| 2   | `gender`            | object    | 5110            | Jenis kelamin pasien                                          |
| 3   | `age`               | float64   | 5110            | Usia pasien                                                   |
| 4   | `hypertension`      | int64     | 5110            | 1 jika pasien memiliki hipertensi, 0 jika tidak               |
| 5   | `heart_disease`     | int64     | 5110            | 1 jika pasien memiliki penyakit jantung, 0 jika tidak         |
| 6   | `ever_married`      | object    | 5110            | Status pernikahan pasien                                      |
| 7   | `work_type`         | object    | 5110            | Jenis pekerjaan pasien                                        |
| 8   | `Residence_type`    | object    | 5110            | Jenis tempat tinggal: Urban (perkotaan) atau Rural (pedesaan) |
| 9   | `avg_glucose_level` | float64   | 5110            | Rata-rata kadar glukosa dalam darah                           |
| 10  | `bmi`               | float64   | 4909            | Indeks Massa Tubuh (BMI) pasien                               |
| 11  | `smoking_status`    | object    | 5110            | Status merokok pasien (tidak pernah, pernah, merokok, dll.)   |
| 12  | `stroke`            | int64     | 5110            | Target: 1 jika pasien pernah mengalami stroke, 0 jika belum   |

## Referensi

[1] World Health Organization. (2018). The top 10 causes of death.  
[2] Kementerian Kesehatan Republik Indonesia. (2018). Riset Kesehatan Dasar (Riskesdas) 2018.  
[3] American Heart Association. (2021). Economic Burden of Stroke in the United States.  
[4] BPJS Kesehatan. (2022). Laporan Pengelolaan Program JKN Tahun 2022.  
[5] Harvard Health Publishing. (2022). Stroke: Risk factors you can control.  
