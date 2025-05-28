# Laporan Proyek Machine Learning – [Nama Anda]

---

## 1. Domain Proyek

Stroke merupakan salah satu penyebab utama kematian dan disabilitas di seluruh dunia. Berdasarkan laporan World Health Organization (WHO), setiap tahunnya terjadi sekitar 15 juta kasus stroke secara global, dengan 5 juta orang meninggal dunia dan 5 juta lainnya mengalami disabilitas permanen akibat penyakit ini [1]. Di Indonesia, beban akibat stroke tidak kalah mengkhawatirkan. Data Riskesdas 2018 yang dirilis oleh Kementerian Kesehatan Republik Indonesia menunjukkan prevalensi stroke nasional mencapai 10,9 per 1.000 penduduk, menjadikannya sebagai penyakit katastropik dengan beban ekonomi dan sosial yang sangat besar [2].

Masalah ini harus diselesaikan karena stroke tidak hanya berdampak pada kesehatan individu, tetapi juga menimbulkan beban ekonomi yang signifikan terhadap sistem kesehatan nasional. Studi dari American Heart Association memperkirakan total beban ekonomi akibat stroke di tingkat global mencapai lebih dari US$ 721 miliar, termasuk biaya pengobatan, rehabilitasi, serta kerugian produktivitas akibat disabilitas dan kematian dini [3]. Di Indonesia, stroke termasuk dalam lima penyakit dengan biaya klaim tertinggi BPJS Kesehatan, mencapai lebih dari Rp2,5 triliun dalam satu tahun [4].

Faktor risiko stroke yang dapat dikenali dan dimitigasi sejak dini meliputi usia lanjut, hipertensi, diabetes, hiperkolesterolemia, riwayat merokok, dan aktivitas fisik rendah [5]. Namun, deteksi dini masih belum optimal, terutama karena rendahnya kesadaran masyarakat dan keterbatasan fasilitas skrining.

Perkembangan machine learning (ML) memberikan solusi potensial untuk memprediksi risiko stroke secara lebih efisien dan terukur, yang dapat membantu intervensi dini dan pengambilan keputusan berbasis data. Oleh karena itu, proyek ini bertujuan membangun model prediksi risiko stroke menggunakan data demografis dan medis yang tersedia secara publik.

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

## Referensi

[1] World Health Organization. (2018). The top 10 causes of death.  
[2] Kementerian Kesehatan Republik Indonesia. (2018). Riset Kesehatan Dasar (Riskesdas) 2018.  
[3] American Heart Association. (2021). Economic Burden of Stroke in the United States.  
[4] BPJS Kesehatan. (2022). Laporan Pengelolaan Program JKN Tahun 2022.  
[5] Harvard Health Publishing. (2022). Stroke: Risk factors you can control.  
