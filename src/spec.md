Spesifikasi
Implementasikan suatu modul FFNN yang memenuhi ketentuan-ketentuan berikut:
FFNN yang diimplementasikan dapat menerima jumlah neuron dari tiap layer (termasuk input layer dan output layer).
FFNN yang diimplementasikan dapat menerima fungsi aktivasi dari tiap layer. Pilihan fungsi aktivasi yang harus diimplementasikan adalah sebagai berikut:
Nama Fungsi Aktivasi
Definisi Fungsi
Linear

ReLU

Sigmoid

Hyperbolic Tangent (tanh)

Softmax

FFNN yang diimplementasikan dapat menerima fungsi loss dari model tersebut. Pilihan loss function yang harus diimplementasikan adalah sebagai berikut:
Nama Fungsi Aktivasi
Definisi Fungsi
MSE

Binary
Cross-Entropy

Categorical
Cross-Entropy

Catatan:
Binary cross-entropy merupakan kasus khusus categorical cross-entropy dengan kelas sebanyak 2.
Log yang digunakan merupakan logaritma natural (logaritma dengan basis e).
Terdapat mekanisme untuk inisialisasi bobot tiap neuron (termasuk bias). Pilihan metode inisialisasi bobot yang harus diimplementasikan adalah sebagai berikut:
Zero initialization
Random dengan distribusi uniform.
Menerima parameter lower bound (batas minimal) dan upper bound (batas maksimal).
Menerima parameter seed untuk reproducibility.
Random dengan distribusi normal.
Menerima parameter mean dan variance.
Menerima parameter seed untuk reproducibility.
Instance model yang diinisialisasikan harus bisa menyimpan bobot tiap neuron (termasuk bias).
Instance model yang diinisialisasikan harus bisa menyimpan gradien bobot tiap neuron (termasuk bias).
Instance model memiliki method untuk menampilkan distribusi bobot dari tiap layer.
Menerima masukan berupa list of integer (bisa disesuaikan ke struktur data lain sesuai kebutuhan) yang menyatakan layer mana saja yang distribusinya akan di-plot.
Instance model memiliki method untuk menampilkan distribusi gradien bobot dari tiap layer.
Menerima masukan berupa list of integer (bisa disesuaikan ke struktur data lain sesuai kebutuhan) yang menyatakan layer mana saja yang distribusinya akan di-plot.
Instance model memiliki method untuk save dan load.
Model memiliki implementasi forward propagation dengan ketentuan sebagai berikut:
Dapat menerima input berupa batch.
Model memiliki implementasi backward propagation untuk menghitung perubahan gradien:
Dapat menangani perhitungan perubahan gradien untuk input data batch.
Gunakan konsep chain rule untuk menghitung gradien tiap bobot terhadap loss function.
Berikut merupakan turunan pertama untuk setiap fungsi aktivasi:
Nama Fungsi Aktivasi
Turunan Pertama
Linear

ReLU

Sigmoid

Hyperbolic Tangent (tanh)

Softmax

Berikut merupakan turunan pertama untuk setiap fungsi loss terhadap bobot suatu FFNN (lanjutkan sisanya menggunakan chain rule):
Nama Fungsi Loss
Definisi Fungsi
MSE

Binary Cross-Entropy

Categorical Cross-Entropy

Model memiliki implementasi metode regularisasi L1 dan L2.
Model memiliki implementasi weight update dengan menggunakan gradient descent untuk memperbarui bobot berdasarkan gradien yang telah dihitung, berikut persamaannya:

Implementasi untuk pelatihan model harus memenuhi ketentuan berikut:
Dapat menerima parameter berikut:
Batch Size
Learning Rate
Jumlah Epoch
Verbose
Verbose 0 berarti tidak menampilkan apa-apa selama pelatihan.
Verbose 1 berarti hanya menampilkan progress bar beserta dengan kondisi training loss dan validation loss saat itu.
Proses pelatihan mengembalikan histori dari proses pelatihan yang berisi training loss dan validation loss tiap epoch
