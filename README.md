# credit-risk-prediction-vix-idx-partners

## 1. BUSINESS UNDERSTANDING

<p align='center'>
  <img src="https://user-images.githubusercontent.com/115754250/213961560-4e07831b-23bc-4351-ad57-84dde772f281.png">
  <figcaption align="center">Gambar ilustrasi finansial risiko kredit</figcaption>
</p>

* Risiko kredit adalah kerugian yang berhubungan dengan potensi kegagaln dalam memenuhi kewajiban membayar kredit ketika waktu jatuh tempo [1].
* Risiko kredit menjadi salah satu faktor penting dalam proses pemberian kredit atau pinjaman oleh perusahaan finansial.
* Dengan mengambil risiko kredit yang tepat, perusahaan dapat keputusan yang tepat dalam proses pemberian kredit sehingga dapat mengurangi risiko default atau gagal bayar dari peminjam.
* Dalam proyek ini, akan dievaluasi data historis peminjam dan menggunakan Algoritma Machine Learning untuk membuat model prediksi risiko kredit yang efektif.

## 2. DATA
* **Darimana sumber dataset yang akan digunakan?**
  - Data diberikan oleh perusahaan.
* **Apa isi dataset tersebut?**
  - Data pinjaman yang diterima dan ditolak.
* **Berapa jumlah atribut pada dataset tersebut?**
  - Terdapat total **75 atribut** pada dataset. Namun untuk pemodelan machine learning, hanya beberapa atribut yang akan digunakan.
* **Berapa jumlah baris/record pada dataset tersebut?**
  - Terdapat total **466.284 record** pada dataset mentahnya. Namun akan ada perubahan ketika digunakan untuk pemodelan machine learning.
* **Apa yang akan diprediksi?**
  - Tingkat risiko pinjaman berdasarkan data historis status peminjam.

## 3. EXPLORATORY DATA ANALYSIS
### 3.1 Status Peminjaman dan Risiko Kredit
Terdapat 9 status pinjaman pada data mentah sebagaimana ditampilkan pada grafik berikut.

![image](https://user-images.githubusercontent.com/115754250/213966662-a1171bd0-59ff-4bb5-a5ca-811ecd92f8bf.png)

Namun untuk machine learning yang akan dibuat, status pinjaman ini akan diubah menjadi risiko kredit dengan dua tingkat risiko, yaitu:
* **Risiko rendah**, yang direpresentasikan dengan angka **0**. Risiko rendah diambil dari data dengan status pinjaman *Fully Paid*.
* **Risiko tinggi**, yang direpresentasikan dengan angka **1**. Risiko tinggi diambil dari data dengan status pinjaman selain *Fully Paid* dan *Current*

Data dengan status pinjaman *Current* tidak akan digunakan karena pinjaman tersebut masih berjalan dan belum diketahui risikonya.

### 3.2 Nilai Null Pada Tiap Atribut
![image](https://user-images.githubusercontent.com/115754250/213963216-1786f1bc-bd14-4561-957b-c5dbaa5755eb.png)

Atribut dengan jumlah nilai null lebih dari 1/4 total record pada data mentah akan dihapus. Berdasarkan grafik di atas, nama atribut yang ditulis dengan warna merah dan bold akan dihapus.

### 3.3 Nilai Unik Pada Tiap Atribut
![image](https://user-images.githubusercontent.com/115754250/213963318-0b2d962e-82d1-4cce-9158-6e01ea8b1ee1.png)

Atribut yang terlalu unik, atau dengan kata lain atribut yang jumlah nilai uniknya sama dengan total record pada data mentah akan dihapus. Berdasarkan grafik di atas, nama atribut yang ditulis dengan warna merah dan bold akan dihapus.

### 3.4 Nilai Unik Pada Atribut Dengan Data Kategorik
![image](https://user-images.githubusercontent.com/115754250/213963750-b0a9d20b-a34b-4ff9-9d9c-852ef9d11ebe.png)

Atribut dengan nilai katogorik akan sulit diolah jika memiliki nilai unik yang terlalu banyak. Oleh karena itu, pada proyek ini atribut dengan data kategorik yang memiliki nilai unik lebih dari 100 akan dihapus. Berdasarkan grafik di atas, nama atribut yang ditulis dengan warna merah dan bold akan dihapus.

### 3.5 Atribut Dengan Data Kategorik
#### 3.5.1 Atribut tanggal
Atribut *issue_d* akan digunkaan untuk membuat atribut *risk_score* dengan nilai 0 (FICO) untuk peminjaman sebelum 5 November 2013 dan nilai 1 (Vantage) untuk peminjaman setelah 5 November 2013.

Atribut *last_pymnt_d dihapus karena memiliki terlalu banyak nilai unik.

#### 3.5.2 Atribut dengan 2 nilai unik
Semua atribut dengan data kategorik yang hanya memiliki 2 nilai unik akan diubah menjadi nilai 0 dan 1.

#### 3.5.3 Atribut *emp_length*
Atribut panjang pekerjaan (*emp_length*) yang ditulis dalam satuan tahun akan diubah menjadi data numerik dengan mengambil digit nomor tahunnya.

### 3.6 Atribut Dengan Data Kstegorik
#### 3.6.1 Deteksi outliers
Pada proyek ini, deteksi outlier akan dilakukan dengan menggunakan metode Z-Score. Kemudian outliers akan dihapus.

#### 3.6.2 Korelasi antar atribut
Atribut yang memiliki korelasi dengan atribut lain di atas 0,8 dan korelasi dengan atribut target (*loan_status*) di atas 0,1 akan digunakan untuk pemodelan machine learning.

### 3.7 Data yang akan digunakan untuk machine learning.
* Terdiri dari 23 kolom dan 140.627 baris.
* 16 kolom numerik, 6 kolom kategorik, dan 1 kolom target.
* Data numerik akan dinormalisasi sehingga rentang nilainya menjadi 0 sampai dengan 1.
* Data kategorik akan diubah menjadi data numerik dengan one-hot encoding.
* Data latih dan data uji akan dipisah dengan rasio 80% : 20%.

## 4. PEMODELAN MACHINE LEARNING
### 4.1 Decision Tree
Decision tree adalah algoritma machine learning yang menggunakan seperangkat aturan untuk membuat keputusan dengan struktur seperti pohon yang memodelkan kemungkinan hasil, biaya sumber daya, utilitas dan kemungkinan konsekuensi atau resiko.

Konsepnya adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, yang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan. [2]

### 4.2 Hyperparameter Tuning
* Pemilihan hyperparameter terbaik akan dilakukan menggunakan **GridSearchCV*.
* Hyperparameter yang dipilih:
  - criterion: *'gini'* dan *'entropy'*
  - max_depth: *2*, *3*, dan *4*
  
### 4.3 Hasil
![image](https://user-images.githubusercontent.com/115754250/213964850-0e450386-2623-4773-a9b5-1f60475b721f.png)

Diperoleh akurasi:
* **97,6%** pada data latih, dan
* **97,4%** pada data uji

Model dapat memprediksi risiko kredit nasabah secara tepat degan kemungkinan lebih dari 95%.

**Precision**
* **95%** kemungkinan model berhasil memprediksi dengan benar pinjaman dengan **risiko tinggi** sebagai **risiko tinggi**.

**Recall**
* **92%** kasus pinjaman dengan **risiko tinggi** dari total pinjaman dengan **risiko tinggi** yang sebenarnya dikenali oleh model.

**F1-Score**
* **94%** keseimbangan antara precision dan recall yang menunjukan kinerja model sangat baik dengan nilai di atas 90%.

## 5. KESIMPULAN
* Model machine learning yang dibuat untuk prediksi risiko kredit bekerja dengan sangat baik. Hasil evaluasi menunjukkan skor precision, recall, dan f1 yang tinggi, yang menandakan bahwa model kami mampu mengenali pinjaman dengan risiko tinggi dengan baik, serta memiliki tingkat akurasi yang tinggi dalam prediksi risiko pinjaman.
* Exploratory data analysis menunjukkan fitur-fitur yang digunakan memiliki korelasi yang tinggi dengan variabel target membuat model berjalan pada performa terbaiknya.
* Secara keseluruhan, model yang dikembangkan dapat digunakan oleh perusahaan untuk memprediksi risiko kredit dengan baik, sehingga dapat digunakan untuk mengambil keputusan keuangan yang lebih baik. Kami akan terus mengevaluasi dan meningkatkan model ini untuk meningkatkan kinerja dan akurasi prediksi.

## 6. REFERENSI
[1] https://www.ocbcnisp.com/id/article/2022/02/24/risiko-kredit-adalah
[2] https://dqlab.id/pahami-metode-decision-tree-sebagai-algoritma-data-science

# THANK YOU
<table>
  <thead>
    <tr>
      <th>Sosial Media</th>
      <th>Pemilik</th>
      <th>Logo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LinkedIn</th>
      <td>Muhafidz Ahmad Halim</th>
      <td><a href="https://www.linkedin.com/in/muhafidz-ahmad-halim/"><img src="https://user-images.githubusercontent.com/115754250/213965797-c07b0ad3-c5e7-4c9e-9f09-5076141f57f4.png" width="30" height="30"/></a></td>
    </tr>
    <tr>
      <td>Instagram</td>
      <td>Muhafidz</td>
      <td><a href="https://www.instagram.com/anumuhafidz/"><img src="https://cdn4.iconfinder.com/data/icons/social-messaging-ui-color-shapes-2-free/128/social-messaging-ui-color-shapes-2-13-512.png" width="30" height="30"></a></td>
    </tr>
  </tbody>
</table>
