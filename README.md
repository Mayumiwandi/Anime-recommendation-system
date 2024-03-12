# Laporan Proyek Machine Learning - Ahmad Wandi
Ini adalah proyek akhir sistem rekomendasi untuk memenuhi submission dicoding. Proyek ini membangun model berbasis _content based filtering_ yang dapat menentukan top-N rekomendasi anime dan model _K-Nearest Neighbor_.
## Project Overview

![inbox_11299784_caaff69976c0a1e97c7d55eb82383680_static-assets-upload6207184415643227018](https://github.com/Mayumiwandi/My-Learn/assets/84662810/0ce1eb4b-9fe5-4b65-bf81-6f4c4f6cc774)

[Referensi gambar](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data)



Anime, animasi khas Jepang yang digemari banyak orang, kini tersedia di berbagai platform streaming. Namun, banyaknya pilihan anime justru membuat pengguna kesulitan menemukan yang sesuai selera. Penelitian ini mengusulkan sistem rekomendasi untuk mengatasi masalah tersebut.[[1](https://openlibrary.telkomuniversity.ac.id/home/catalog/id/185894/slug/anime-rekomendasi-menggunakan-collaborative-filtering.html)] Sistem ini akan menganalisis riwayat tontonan pengguna, genre favorit, dan rating yang diberikan. Dengan demikian, sistem dapat merekomendasikan anime yang sesuai preferensi masing-masing pengguna. Selain itu, faktor lain seperti popularitas, ulasan pengguna, dan rekomendasi komunitas juga akan dipertimbangkan.[[2](https://repository.uinjkt.ac.id/dspace/bitstream/123456789/45316/1/Ida%20Aisyah.pdf)]
Sistem ini menawarkan keuntungan signifikan bagi pengguna dan penyedia layanan streaming. Pengguna dapat menemukan anime favorit baru, menjelajahi genre baru, dan menemukan anime yang sesuai suasana hati mereka. Sementara bagi perusahaan, sistem ini dapat meningkatkan jumlah penonton, menyediakan konten yang beragam, meningkatkan kepuasan pengguna, dan memahami preferensi pengguna terkait anime yang diinginkan. Singkatnya, sistem rekomendasi dapat menjadi solusi efektif untuk membantu pengguna menemukan anime yang sesuai selera dan meningkatkan pengalaman menonton mereka secara keseluruhan.[[3](https://www.researchgate.net/publication/274712918_Rekomendasi_Anime_dengan_Latent_Semantic_Indexing_Berbasis_Sinopsis_Genre)]

## Business Understanding
Pengembangan sistem rekomendasi anime memiliki potensi untuk memberikan banyak manfaat bagi pengguna dan platform streaming anime. Sistem ini dapat membantu pengguna menemukan anime yang sesuai dengan selera mereka dengan lebih mudah dan efisien, dan dapat membantu platform meningkatkan engagement pengguna, kepuasan pengguna, dan efisiensi platform.[[4](https://jurnal.stkippgritulungagung.ac.id/index.php/jipi/article/view/4222)]
### Problem Statements
- Bagaimana cara membuat sistem rekomendasi anime yang merekomendasikan pengguna berdasarkan genre anime?
- Dengan menggunakan data rating yang dimiliki pengguna, bagaimana perusahaan jasa streaming dapat merekomendasikan anime yang belum pernah ditonton pengguna?
- Bagimana membuat model sistem rekomendasi Cosine Similarity dan K-Nearest Neighbor?
- Bagaimanna cara mengukur nilai perfoma model sistem rekomendasi yang telah dibangun?

### Goals
Untuk menjawab permasalahan tersebut dibuatlah sistem rekomendasi dengan tujuan sebagai berikut:

- Menghasilkan rekomendasi anime sebanyak Top-N Rekomendasi kepada pengguna berdasarkan genre.
- Menghasilkan beberapa rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton.
- Membuat model sistem rekomendasi Cosine Similarity dan K-Nearest Neighbor berdasarkan fitur yang telah dipilih dari dataset
- Mengukur perfoma model sistem rekomendasi dengan menggunakan metrik evaluasi

### Solution Approach
Menganalisis data dengan melakukan Exploratory Data Analysis dan melakukan visualisasi.
Agar didapatkan model prediksi yang baik maka dilakukanlah _data cleaning_ berupa menghapus missing value , memeriksa apakah ada data yang duplikat ,dan Menghapus tanda baca alfanumerik dan Hapus link (URL). Melakukan _One hot encoding_  untuk mengubah data kategorikal menjadi nilai numerik, Untuk mengetahui perfoma model dilakukan pengecekan performa dengan metrik evaluasi seperti _Precission, Calinski Harabasz Score, dan Davies Bouldin Score_.
## Data Understanding
### EDA - Deskripsi Variabel

| Jenis    | Keterangan                                                |
|----------|-----------------------------------------------------------|
| Title    | Anime Dataset 2023                                        |
| Source   |[Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data?select=anime-filtered.csv)                                                  |
| Maintainer | [Sajid](https://www.kaggle.com/dbdmobile)                                                   |
| License  | Database: Open Database, Contents: Database Contents      |
| Visibility | Publik                                                  |
| Tags     | Arts and Entertainment, Movies and TV Shows, Anime and Manga, Popular Culture, Japan |
| Usability | 10.00                                                     |


Berikut informasi pada dataset: Kumpulan dataset ini dikumpulkan dari platform [MyAnimeList](https://myanimelist.net/)  , komunitas online populer dan database untuk penggemar anime dan manga. Platform ini menyediakan informasi berharga tentang acara anime, profil pengguna, dan skor pengguna untuk berbagai anime. Dataset yang digunakan pada proyek kali ini disediakan secara publik di kaggle dengan nama datasets yaitu: _Anime Dataset 2023_ . Dataset ini dapat diunduh di Kaggle : [Anime Dataset 2023](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset/data?select=anime-filtered.csv) .


**Berikut informasi pada dataset** :
 - Datasets berupa file csv (Comma-Seperated Values).
 - Dataset berupa 6 buah file CSV yaitu: 
    * anime-dataset-2023.csv  
    * anime-filtered.csv      
    * final_animedataset.csv  
    * user-filtered.csv       
    * users-details-2023.csv  
    * users-score-2023.csv

Pada model kali ini dataset yang digunakan adalah file `anime-filtered.csv`
 - Dataset memiliki 14952 sample dengan 25 fitur.
 - Dataset memilik 15 fitur `object`, 8 fitur `int64`, dan 2 fitur `float64`.
 - Terdapat *Missing value* pada fitur  `sypnopsis sebanyak 1350 dan Ranked sebanyak 1721`.
 - Tidak ada data yang duplikat.

### Variable - variable pada dataset
Kolom datasets anime memiliki informasi berikut:
* **`anime_id`:** ID unik untuk setiap anime (angka atau kode pengenal).
* **`Name`:** Judul anime dalam bahasa aslinya.
* **`Score`:** Skor atau rating yang diberikan kepada anime.
* **`Genres`:** Genre anime, dipisahkan dengan koma (misalnya, Aksi, Komedi, Fantasi).
* **`English name`:** Judul anime dalam bahasa Inggris (jika tersedia).
* **`Japanese name`:** Judul anime dalam bahasa Jepang (jika tersedia).
* **`Synopsis`:** Deskripsi singkat atau ringkasan plot anime.
* **`Type`:** Jenis anime (misalnya, TV Series, Movie, OVA, dll.).
* **`Episodes`:** Jumlah episode dalam anime.
* **`Aired`:** Tanggal penayangan anime.
* **`Premiered`:** Musim dan tahun penayangan perdana anime.
* **`Producers`:** Perusahaan produksi atau produser anime.
* **`Licensors`:** Pihak yang memiliki lisensi anime (misalnya, platform streaming).
* **`Studios`:** Studio animasi yang mengerjakan anime.
* **`Source`:** Sumber materi anime (misalnya, manga, light novel, original).
* **`Duration`:** Durasi setiap episode anime.
* **`Rating`:** Batasan usia untuk menonton anime.
* **`Ranked`:** Peringkat anime berdasarkan popularitas atau kriteria lain.
* **`Popularity`:** Peringkat popularitas anime.
* **`Members`:** Jumlah anggota yang telah menambahkan anime ke daftar mereka di platform.
* **`Favorites`:** Jumlah pengguna yang menandai anime sebagai favorit.
* **`Watching`:** Jumlah anime yang sedang ditonton oleh pengguna.
* **`Completed`:** Jumlah anime yang telah selesai ditonton oleh pengguna.
* **`On Hold`:** Jumlah anime yang ditunda oleh pengguna.
* **`Dropped`:** Jumlah anime yang dihentikan oleh pengguna.


![rating-rekomendasi](https://github.com/Mayumiwandi/My-Learn/assets/84662810/b9d423b6-a978-43b3-bcf4-0b12c20e766c)

Gambar 1. Rating


![download](https://github.com/Mayumiwandi/My-Learn/assets/84662810/4a121818-d02a-45eb-8b2c-6d1cffd3dddf)

Gambar 2. Categories distribution

Berdasarkan _Gambar 1. Rating_ dapat kita lihat rata-rata Rating adalah `6.5`, minimal rating adalah `1.8`, dan maxsimal rating adalah `9.1`. Dan berdasarkan _Gambar 2. Categories distribution_ anime terdiri dari 6 tipe berupa TV, OVA, Movie, Special, ONA, Music. _TV (Television Series)_ yang ditayangkan di televisi dengan episode yang bervariasi, _OVA (Original Video Animation)_ yang dirilis untuk media rumahan, seperti DVD, _Movie_ yang dirilis di bioskop dengan durasi yang lebih panjang, _Special_ yang seringkali menjadi bonus atau tambahan dari seri TV atau film, _ONA (Original Net Animation)_ yang didistribusikan secara daring, dan Music yang dibuat untuk menyoroti perilisan album musik atau single.


![download (1)](https://github.com/Mayumiwandi/My-Learn/assets/84662810/6c8ef782-c945-4ed9-baf7-6b11e0b6ad45)

Gambar 3. Top 10 Anime Community


![download (3)](https://github.com/Mayumiwandi/My-Learn/assets/84662810/1bfabba0-6fc3-4ba3-b039-6e7d6bb6f470)


Gambar 4. Top Rating Tertinggi

Berdasarkan _Gambar 3. Top 10 Anime Community_ dapat dilihat Top 10 Komunitas anime Death Note menjadi komunitas terbanyak pertama, disusul dengan Shingeki no Kyojin kedua, Fullmetal Alchemist: Brotherhood ketiga, Sword Art Online keempat, One Punch Man kelima, Boku no Hero Academia keenam, Tokyo Ghoul ketujuh, Naruto kedelapan, Steins Gate kesembilan, dan No Game No Life kesepuluh. Informasi ini dapat digunakan pengembang sistem dalam merekomendasikan anime yang populer kepada penggunanya. Banyaknya anggota komunitas anime menandakan bahwa anime cukup favorit dan populer di kalangan pengguna.

Dan pada _Gambar 4. Top Rating Tertinggi_  dapat dilihat rating tertinggi pertama adalah Fullmetal Alchemist: Brotherhood, dan kedua Shingeki no Kyojin: Final Season, ketiga Steins Gate, keempat Shingeki no Kyojin Season 3 Part 2, kelima Hunter x Hunter(2011), keenam Gintama°, ketujuh Gintama', kedelapan Ginga Eiyuu Densetsu, kesembilan Gintama': Enchousen dan kesepuluh adalah 3-gatsu no Lion 2nd Season. Informasi ini dapat digunakan pengembang sistem dalam merekomendasikan anime yang populer kepada penggunanya. Hal ini dikarenakan semakin banyaknya kontribusi peringkat, semakin banyak pula pengguna yang menonton anime tersebut (populer).

## Data Preparation

Pada proses Data Preparation dilakukan _text cleaning_ untuk membersihkan teks dari tanda baca dan URL. Untuk data yang _missing value_ adalah dengan menerapkan metode _dropping_ menggunakan drop(). adapun alasan mengapa Metode _dropping_ ini digunakan karna data yang akan di hapus tidak mempengaruhi model. Yang awalnya jumlah dataset sebanyak `14952` dan dengan menghapus jumlah _missing value_ dataset sekarang menjadi `13229`. Untuk membangun sistem rekomendasi pada proyek kali ini digunakan fitur `Name, Score, Genres, Type, Studios`. Sistem rekomendasi berbasis genre, atribut yang akan digunakan yakni `Name dan genres`. Sistem rekomendasi dengan metode collaborative filtering, atribut yang digunakan yakni: `Name, Score dan Type`. Dan dilakukan _one-hot encoding_ untuk mengubah fitur `Type`dan `Score`, digunakan untuk mengubah variabel kategorikal menjadi bentuk yang lebih mudah dipahami oleh model pembelajaran mesin.

## Modeling 
Pada proyek ini hanya gunakan Model Cosine Similarity dan K-Nearest Neighbor. Kedua algoritma ini akan mempelajari kesamaan antar data dalam fitur yang ada.

### Cosine similarity

_Cosine similarity_ adalah metode untuk mengukur seberapa mirip dua vektor dalam ruang multidimensi. Ini adalah pengukuran kosinus sudut antara dua vektor yang dimensi dan magnitudonya direpresentasikan sebagai titik dalam ruang. Nilai similaritas kosinus berkisar antara -1 hingga 1, di mana nilai 1 menunjukkan kedua vektor sepenuhnya sejajar (100% mirip), 0 menunjukkan vektor tegak lurus (tidak ada keterkaitan), dan -1 menunjukkan kedua vektor sepenuhnya berlawanan arah (100% tidak mirip). Metode ini sering digunakan dalam pemrosesan teks dan pengelompokan data untuk menentukan tingkat kesamaan antara dokumen atau fitur dalam dataset. [[5](https://medium.com/geekculture/cosine-similarity-and-cosine-distance-48eed889a5c4)]

Cosine Similarity dituliskan dalam rumus: 

$$Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)$$ 

dimana: 
- (A·B)menyatakan produk titik dari vektor A dan B.
- ||A|| mewakili norma Euclidean (magnitudo) dari vektor A.
- ||B|| mewakili norma Euclidean (magnitudo) dari vektor B.

Untuk melakukan pengujian model, digunakan potongan kode berikut.
```python
anime_recommendations('One Piece')
```
| Name                                             | Genres                                                       |
|--------------------------------------------------|--------------------------------------------------------------|
| One Piece Episode of Sorajima                   |Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power.         |
| One Piece Episode of Merry Mou Hitori no Nakama no Monogatari| Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power.          |
| One Piece Movie 14 Stampede                      | Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power. |
| One Piece Episode of East Blue  Luffy to 4 nin no Nakama no Daibouken| Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power.  |
| One Piece Episode of Sabo  3 Kyoudai no Kizuna Kiseki no Saikai to Uketsugareru Ishi| Action, Adventure, Comedy, Drama, Fantasy, Shounen, Super Power   |

Table 1. Hasil Pengujian Model Content Based Filtering (dengan Filter Genres).

Berdasarkan _Table 1. Hasil Pengujian Model Content Based Filtering (dengan Filter Genres)._  Sistem telah berhasil merekomendasikan top 5 persen anime yang mirip dengan **One Piece**, yang termasuk beberapa film dan seri dari **One Piece** itu sendiri. Ini berarti bahwa jika seorang pengguna menyukai **One Piece**, maka sistem dapat memberikan rekomendasi untuk seri atau film lain dalam waralaba **One Piece**. Dengan pendekatan ini, sistem mengidentifikasi anime-anime yang memiliki kemiripan dalam genre dengan **One Piece**, sehingga memungkinkan pengguna untuk menemukan konten yang sesuai dengan preferensi mereka berdasarkan kesukaan mereka terhadap **One Piece**.

Kelebihan _Cosine Similarity_:
- Kompleksitas yang rendah, membuatnya efisien dalam perhitungan.
- Cocok digunakan pada dataset dengan dimensi yang besar karena tidak terpengaruh oleh jumlah dimensi.

Kekurangan _Cosine Similarity_:
- Hanya memperhitungkan arah dari vektor, tanpa memperhitungkan magnitudo (besarnya).
- Perbedaan dalam magnitudo vektor tidak sepenuhnya diperhitungkan, yang berarti nilai-nilai yang sangat berbeda dapat dianggap mirip jika arah vektornya sama.


### K-Nearest Neighbor
_K-Nearest Neighbor (KNN)_ adalah salah satu algoritma paling sederhana dalam klasifikasi data. Algoritma ini mudah dipahami karena mengelompokkan data berdasarkan jarak terdekat dengan tetangga lainnya. Dalam KNN, kita mempertimbangkan sejumlah tetangga terdekat (ditentukan oleh parameter K) untuk menentukan kelas atau label dari data yang akan diklasifikasikan. Ketika K=1, algoritma hanya memperhitungkan satu tetangga terdekat atau satu rekaman data dengan karakteristik terdekat.[[6](https://medium.com/bee-solution-partners/cara-kerja-algoritma-k-nearest-neighbor-k-nn-389297de543e)] 

K-Nearest Neighbor dituliskan dalam rumus:

 $$Euclidean Distance (P, Q) = sqrt(∑(Pi - Qi)^2)$$

dimana:
- Pi mewakili fitur ke-i dari titik data P.
- Qi mewakili fitur ke-i dari titik data Q (titik data dari kumpulan data D).
- ∑ merupakan simbol penjumlahan pada semua fitur titik data.

berikut merupakan hasil pengujian model _K-Nearest Neighbor_ dengan _metrik Euclidean Distance_: 

Apabila pengguna menyukai aplikasi:_**Neon Genesis Evangelion Death  Rebirth**_

Berikut ini adalah aplikasi yang juga mungkin akan disukai :
| Anime Name                                   | Similarity Score |
|----------------------------------------------|------------------|
| Neon Genesis Evangelion Death Rebirth        | 100.0%           |
| Neon Genesis Evangelion The End of Evangelion| 98.94%           |
| Kekkaishi TV                                 | 98.59%           |
| Doraemon Doraemon Comes Back                 | 98.59%           |
| Dr Slump Aralechan                           | 98.59%           |

Tabel 2. Hasil Pengujian Model K-Nearest Neighbor

Berdasarkan _Tabel 2. Hasil Pengujian Model K-Nearest Neighbor_, kita dapat melihat bahwa model _K-Nearest Neighbor_ memberikan rekomendasi Anime berdasarkan kemiripan fitur-fitur seperti 'Name', 'Score', 'Type', dan 'Studios'. Hasil rekomendasi untuk Anime yang mirip dengan `Neon Genesis Evangelion Death  Rebirth` berdasarkan berdasarkan fitur-fitur yang dipelajari memberikan hasil rekomendasi aplikasi serupa yaitu: _Neon Genesis Evangelion Death Rebirth, Neon Genesis Evangelion The End of Evangelion, Kekkaishi TV, Doraemon Doraemon Comes Back, Dr Slump Aralechan_. Seperti tampak pada _Tabel 2. Hasil Pengujian Model K-Nearest Neighbor_ dengan tingkat kemiripan dalam persentase berturut-turut senilai 100.0%, 98.94%, 98.59%, 98.59%, 98.59% . Tentunya model ini akan sangat membantu pengguna menemukan aplikasi yang mirip dengan **_Neon Genesis Evangelion Death Rebirth_**.

Kelebihan KNN:
- Pelatihan sangat cepat.
- Sederhana dan mudah dipelajari.
- Tahan terhadap data pelatihan yang memiliki derau.
- Efektif jika data pelatihan besar.

Kekurangan KNN:
- Penentuan nilai k menjadi bias dalam model.
- Komputasi yang kompleks, terutama pada data besar atau dimensi fitur tinggi.
- Keterbatasan memori karena harus menyimpan semua data pelatihan.
- Rentan terhadap atribut yang tidak relevan yang dapat memengaruhi hasil klasifikasi.


## Evaluation
Metrik evaluasi digunakan untuk menilai seberapa baik performa suatu model. Dalam konteks ini, beberapa metrik evaluasi yang umum digunakan untuk mengukur kinerja model antara lain Precission, Calinski Harabasz Score, dan Davies Bouldin Score. Metrik-metrik ini bertujuan untuk memberikan gambaran tentang seberapa baik model bekerja dalam melakukan tugas tertentu, seperti klasifikasi atau klastering data.

### Precission
_Precission_ adalah metrik yang penting untuk mengevaluasi kinerja model pengelompokan. Metrik ini membantu dalam memahami seberapa akurat model dalam mengidentifikasi data positif. Nilai presisi yang tinggi menunjukkan bahwa model jarang membuat prediksi positif yang salah, sehingga prediksi positifnya dapat lebih dipercaya.[[7](https://esairina.medium.com/memahami-confusion-matrix-accuracy-precision-recall-specificity-dan-f1-score-610d4f0db7cf)] 

_Precission_ dituliskan dalam rumus:

$$Presisi = \frac{TP}{TP + FP}$$

dimana: 
- TP (True Positive): Jumlah data yang diprediksi positif dan memang benar-benar positif.
- FP (False Positive): Jumlah data yang diprediksi positif, namun kenyataannya adalah negatif.


*Interpretasi* dari hasil presisi berdasarkan _Table 1. Hasil Pengujian Model Content Based Filtering (dengan Filter Genres)_. menunjukkan bahwa presisi model rekomendasi Top-5 adalah sempurna, yaitu 5/5 atau 100%. Ini menandakan bahwa model tersebut memberikan rekomendasi dengan tingkat presisi yang sangat tinggi, yakni 100%. Ini sesuai dengan hasil pengujian yang menunjukkan bahwa model mampu memberikan rekomendasi dengan nama dan genre yang mirip dengan anime `One Piece`, seperti _Action, Adventure, Comedy, Drama, Fantasy_, Shounen, dan Super Power. Hasil rekomendasi menampilkan lima aplikasi dengan genre yang serupa dengan `One Piece`.

### Calinski-Harabasz score
Calinski-Harabasz score adalah metrik evaluasi untuk algoritme pengelompokan yang mengukur seberapa baik pengelompokan memisahkan data ke dalam kelompok-kelompok yang kompak dan terpisah. Didefinisikan sebagai rasio antara sebaran dalam cluster dan sebaran antar cluster, semakin tinggi nilai CH, semakin baik kinerja pengelompokan tersebut, tanpa memerlukan informasi tentang label kebenaran dasar.[[8](https://medium.com/@haataa/how-to-measure-clustering-performances-when-there-are-no-ground-truth-db027e9a871c)] 

Rumus  Calinski-Harabasz Score (CH) adalah:

$$CH = \frac{B}{W} \times \frac{N - k}{k - 1}$$

Di mana:
- \( B \) adalah sebaran antar cluster (between-cluster scatter).
- \( W \) adalah sebaran dalam cluster (within-cluster scatter).
- \( N \) adalah jumlah total data.
- \( k \) adalah jumlah cluster.

Untuk melakukan pengujian model, digunakan potongan kode berikut.
```python
calinski_harabasz_score(data_new, animedf_name)
```
Dan didapatkan score dari pengujian model.
```
3.1613291729405617
```
Hasil evaluasi menunjukkan bahwa kluster dalam model ini masih belum terpisahkan dengan baik, yang tercermin dari nilai skor Calinski-Harabasz (CH) yang relatif rendah sebesar `3.1613291729405617`. Kondisi ini mengindikasikan adanya potensi untuk rekomendasi yang kurang sesuai pada beberapa aplikasi, yang mungkin tidak sepenuhnya sesuai dengan preferensi pengguna. Oleh karena itu, perlu dilakukan peninjauan lebih lanjut atau penyesuaian pada model untuk meningkatkan pemisahan kluster dan akurasi rekomendasi.

### Davies Bouldin Score
 _**Davies Bouldin Score (DB)**_ adalah metrik evaluasi kinerja pengelompokan yang mengukur rata-rata kesamaan setiap cluster dengan cluster yang paling mirip dengan membandingkan jarak dalam cluster terhadap jarak antar cluster. Dengan skor minimum nol, semakin rendah nilai DB, semakin baik kinerja pengelompokannya, menunjukkan cluster yang lebih dekat satu sama lain dan kurang tersebar. Berbeda dari sebagian besar metrik, DB tidak memerlukan pengetahuan apriori tentang label kebenaran dasar, mirip dengan Silhouette Score, namun memiliki formulasi yang lebih sederhana, memberikan cara efisien untuk mengevaluasi pengelompokan tanpa memerlukan pengetahuan tambahan tentang struktur data.[[9](https://ieeexplore.ieee.org/document/4766909)] 
 
 Rumus Davies-Bouldin Score (DB) adalah:


$$DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{R_i + R_j}{d(c_i, c_j)} \right)$$

Di mana:
- \( k \) adalah jumlah cluster.
- \( R_i \) adalah radius dalam cluster ke-i.
- \( d(c_i, c_j) \) adalah jarak antara pusat cluster ke-i (\( c_i \)) dan pusat cluster ke-j (\( c_j \)).

Davies-Bouldin didefinisikan sebagai rata-rata dari nilai-nilai R, di mana setiap nilai R adalah rasio antara jumlah dari radius dalam cluster (dalam pengertian jarak, misalnya Euclidean distance) dan jarak antara pusat cluster, dengan pusat-pusat yang lain. Rasio ini digunakan untuk mengevaluasi kemiripan setiap cluster dengan cluster lain.

Untuk melakukan pengujian model, digunakan potongan kode berikut.
```python
davies_bouldin_score(data_new, animedf_name)
```
Dan didapatkan score dari pengujian model.
```python
0.7864266764751376
```

Hasil evaluasi Davies-Bouldin (DB) menunjukkan bahwa model ini memiliki skor yang relatif cukup kecil, dengan nilai DB sebesar `0.7864266764751376` Hal ini menandakan bahwa model sudah memiliki separasi kluster yang cukup baik. Sebagai hasilnya, rekomendasi anime memiliki kualitas yang baik, mempertimbangkan bahwa pengelompokan kluster dalam model sudah cukup efektif dalam memisahkan data. Hal ini terbukti dengan hasil rekomendasi aplikasi yang sudah cukup baik.
 
 
## Referensi
1. Iklil jayaperwira,dkk.(2023). Anime Rekomendasi Menggunakan Collaborative Filtering. Jurnal e-Proceeding of Engineering.Vol.10, No.3 Juni 2023. Tersedia: [tautan.](https://openlibrary.telkomuniversity.ac.id/home/catalog/id/185894/slug/anime-rekomendasi-menggunakan-collaborative-filtering.html)
2. Ida Aisyah. (2019). ANIME DAN GAYA HIDUP MAHASISWA (Studi pada Mahasiswa yang Tergabung dalam Komunitas Japan Freak UIN Jakarta). Skripsi UNIVERSITAS ISLAM NEGERI SYARIF HIDAYATULLAH JAKARTA. Tersedia: [tautan.](https://repository.uinjkt.ac.id/dspace/bitstream/123456789/45316/1/Ida%20Aisyah.pdf)
3. Rudy Aditya Abarja, Hapnes Toba. (2015). Rekomendasi Anime dengan Latent Semantic Indexing Berbasis Sinopsis Genre. Tersedia: [tautan.](https://www.researchgate.net/publication/274712918_Rekomendasi_Anime_dengan_Latent_Semantic_Indexing_Berbasis_Sinopsis_Genre)
4. Nazhif Muafa Roziqiin, M. Faisal. (2024). SISTEM REKOMENDASI PEMILIHAN ANIME MENGGUNAKAN USER-BASED COLLABORATIVE FILTERING. Jurnal Ilmiah Penelitian dan Pembelajaran Informatika.Vol 9, No 1 (2024). Tersedia: [tautan.](https://jurnal.stkippgritulungagung.ac.id/index.php/jipi/article/view/4222)
5. Sindhu Seelam. (2021). Machine Learning Fundamentals: Cosine Similarity and Cosine Distance. Published in Geek Culture. Tersedia: [tautan.](https://medium.com/geekculture/cosine-similarity-and-cosine-distance-48eed889a5c4)
6. Asep Maulana Ismail. (2018). Cara Kerja Algoritma k-Nearest Neighbor (k-NN). Published in Bee Solution Partners. Tersedia: [tautan.](https://medium.com/bee-solution-partners/cara-kerja-algoritma-k-nearest-neighbor-k-nn-389297de543e)
7. Rina. (2023). Memahami Confusion Matrix: Accuracy, Precision, Recall, Specificity, dan F1-Score untuk Evaluasi Model Klasifikasi. Tersedia: [tautan.](https://esairina.medium.com/memahami-confusion-matrix-accuracy-precision-recall-specificity-dan-f1-score-610d4f0db7cf)
8. Haitian Wei. (2020). How to measure clustering performances when there are no ground truth?. Tersedia: [tautan.](https://medium.com/@haataa/how-to-measure-clustering-performances-when-there-are-no-ground-truth-db027e9a871c)
9. Davies, David L. Bouldin, Donald W. (1979). A Cluster Separation Measure. IEEE Transactions on Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227. Tersedia: [tautan.](https://ieeexplore.ieee.org/document/4766909)









_