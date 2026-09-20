# Low-Rank Scalar-Gated MAML (LRSGMAML)

Bu bağımsız submission, `baselines/sgmaml/` uygulamasına öğrenilebilir, statik
bir low-rank gradyan dönüşümü ekler. SGMAML'a göre değişen yöntem bileşeni,
Conv2d ağırlıklarının support gradyanlarına eklenen bu dönüşümdür. Backbone,
episodik veri protokolü, BN davranışı, model seed'i (98), first-order support
gradyanları, Adam, learning rate'ler, iki tasklık meta-batch, beş inner-step,
clipping, validation sıklığı ve en iyi checkpoint seçme kuralı korunur.

## Güncelleme

Her fast-weight tensörü için mevcut skalar gate korunur:

```text
m_j = sigmoid(a_j),  a_j başlangıcı = 4.0
```

Bir Conv2d ağırlığı `W_j: [C_out, C_in, k_h, k_w]` için önce SGMAML'ın
mevcut elementwise gradient clipping'i uygulanır. Ardından:

```text
G_j = clipped_grad_j.reshape(C_out, -1)
r_j = min(configured_rank, C_out)
U_j, V_j: [C_out, r_j]
R_j = U_j @ (V_j.T @ G_j)
G_transformed_j = m_j * G_j + R_j
W_next_j = W_j - inner_lr * G_transformed_j.reshape_as(W_j)
```

Kod, U sıfırken SGMAML ile aynı kayan nokta işlem sırasını korumak için önce
`W_j - inner_lr * m_j * clipped_grad_j`, ardından `-inner_lr * R_j`
işlemini uygular. Düzeltme ayrıca gate ile çarpılmaz. `C_out × C_out` bir
matris veya bütün ağırlıkları kapsayan dev bir dönüşüm oluşturulmaz.

Stem, residual ve projection/skip Conv2d ağırlıkları dahildir. Parametreler,
gerçek `nn.Conv2d` modüllerinden bulunan parametre isimleriyle eşleştirilir;
fast-weight isim sırası ve Conv2d şekilleri doğrulanır. BN parametreleri, Conv2d
bias'ları ve classifier weight/bias yalnız SGMAML'ın skalar güncellemesini
kullanır. Classifier way sayısıyla yeniden boyutlandırılabilir; backbone
faktörleri değişmez. Her task, saklanan aynı initialization'dan başlar.

Bu ResNet-18'de 20 Conv2d ağırlığı vardır:

```text
sum(C_out) = 64 + 4 * (64 + 128 + 256 + 512) + (128 + 256 + 512)
           = 4,800
ek parametre = sum_j 2 * C_out_j * min(rank, C_out_j)
rank=4 için = 38,400
```

Bu sayı SGMAML'a ek U/V parametreleridir; mevcut 62 skalar gate aynen kalır.

## Başlangıç, öğrenme ve RNG

U sıfır, V ise ortalaması sıfır ve standart sapması `1/sqrt(C_out)` olan
normal dağılımla başlatılır. Başlangıçta `R_j=0` olduğundan güncelleme
SGMAML ile eşleşir. İlk outer backward'da U genellikle gradyan alır. V'nin
ilk gradyanının sıfır olması beklenir, çünkü bu gradyan U ile çarpılır.
U sıfırdan ayrıldığında V de öğrenebilir. İki faktörü birlikte sıfır
başlatmak her iki öğrenme yolunu da kapatır.

Faktör oluşturma, ayrı ve seed'i belirlenmiş bir CPU `torch.Generator`
kullanır. Oluşturma ve checkpoint yükleme sırasında global CPU/CUDA RNG
akışı ilerletilmez. Mevcut model ve episodik classifier oluşturmanın kendi
RNG tüketimi SGMAML'daki gibi devam eder.

Burada "statik", faktörlerin task veya inner-step girdisinden üretilmemesi
anlamına gelir. Gate ve U/V, tasklar ve inner-step'ler arasında ortaktır ve
query loss üzerinden outer optimizer tarafından öğrenilir. First-order
support gradyanları korunurken gate/U/V işlemleri hesaplama grafiğinde
kalır. Initialization, gate ve U/V aynı outer optimizer ve task gradyanı
biriktirme yoluna dahildir. Inner-loop yalnız fast-weight'leri adapte eder.
Validation ve meta-test sırasında gate/U/V değerleri sabit tutulur.

## Config ve checkpoint

`config.json`, SGMAML config'inin şu ek alanı içeren kopyasıdır:

```json
"low_rank": {"rank": 4}
```

Rank pozitif tamsayı olmalıdır. Yeni normalization, clipping, regularization
veya öğrenilebilir ek ölçek yoktur.

Mevcut beş checkpoint dosyasına `low_rank_transport.pickle` eklenir.
Dosya format sürümünü, dönüşüm türünü, rank'ı, gerçek parametre isimlerini,
beklenen şekilleri ve öğrenilmiş U/V değerlerini içerir. Eksik veya uyumsuz
dönüşüm dosyası açıklayıcı hatayla reddedilir; rastgele faktörlerle devam
edilmez. En iyi validation noktasında initialization, gate ve U/V birlikte
kopyalanır. Daha sonra kaydedilen learner bu eşleşmiş snapshot'ı kullanır.

## Tanı kayıtları

Her mevcut validation turunda, o turun ilk en fazla üç task'ının bütün
inner-step'lerinden tanılar alınır; ayrıca veri veya task örneklenmez:

- Sigmoid gate istatistikleri.
- Conv katmanı başına U ve V normları.
- `||R_j||_F / (||sigmoid(a_j) * G_j||_F + eps)`.
- Clipping sonrası girdi gradyanı ve dönüştürülmüş gradyan arasındaki cosine
  similarity; sıfır normlu durumlar açıkça işaretlenir.

Tanılar detach edilmiş değerlerden üretilir; hesaplama grafiği saklamaz.
Kayıtlar `[LRSGMAML diagnostics]` etiketiyle yazdırılır. Runner logger'ı bir
`logs_dir` sağladığında validation kayıtları bu dizinde JSONL olarak da
tutulur. Kaydedilen model klasöründeki `low_rank_diagnostics.json`, validation
geçmişini içerir. Notebook'un model klasörünü bütünüyle kopyalayan mevcut
artifact işlemi bu dosyayı da kapsar; yeni W&B grafik entegrasyonu eklenmez.

Başlangıçta düzeltme oranı sıfır olmalıdır. Sıfır olmayan gradyanlarda pozitif
skalar gate yönü değiştirmediği için cosine similarity yaklaşık 1'dir.
Eğitim ilerledikçe oran ve cosine, dönüşümün ne ölçüde devreye girdiğini
gösterir. U/V normları tek başına yeterli değildir: `U*c` ve `V/c` aynı
dönüşümü verir. Tanı geçmişinin son kaydı ile seçilen en iyi checkpoint aynı
validation turuna ait olmak zorunda değildir.

## Çalıştırma örneği

Verinin önceden `/content/meta_album_feedback` dizinine hazırlandığı Colab
ortamında, repo kökünden mevcut runner kullanılabilir. Model seed'i baseline
içindeki 98, aşağıdaki data seed'i 93'tür. Çıktı klasörleri bu baseline'a
ayrılmıştır:

```bash
python -u -m cdmetadl.run \
    --seed=93 \
    --input_data_dir=/content/meta_album_feedback \
    --submission_dir=/content/AG-Meta_Album/baselines/lrsgmaml \
    --output_dir_ingestion=/content/ag_meta_outputs/lrsgmaml_feedback_data_seed_93/ingestion \
    --output_dir_scoring=/content/ag_meta_outputs/lrsgmaml_feedback_data_seed_93/scoring \
    --test_tasks_per_dataset=100 \
    --overwrite_previous_results=False \
    --verbose=True \
    --save_train_raw_outputs=False
```

## Mantıksal değerlendirme

Bu, SGMAML üzerinde kontrollü biçimde denenebilecek makul bir hipotezdir.
Skalar gate yalnız bir tensörün adım büyüklüğünü değiştirirken low-rank terim,
çıkış kanalları arasında gradyan bilgisi karıştırabilir ve güncellemenin
yönünü değiştirebilir. Az sayıda ek parametre, aynı protokol ve başlangıçta
SGMAML'a eşit güncelleme, yöntemi karşılaştırmayı kolaylaştırır. Görevler
arasında ortak faydalı kanal ilişkileri varsa iyileşme sağlayabilir.

İç optimizasyondaki gradyan dönüşümlerini öğrenmek için ilgili bir araştırma
örneği [Meta-Curvature](https://arxiv.org/abs/1902.03356) çalışmasıdır.
Buradaki çıkış kanalı low-rank düzeltmesi ayrı bir tasarım seçimidir;
bu bağlantı LRSGMAML için başarı veya özgünlük kanıtı sunmaz.

Başarımı garanti değildir. `m*I + U@V.T` simetrik veya pozitif tanımlı
olmaya zorlanmaz; güncellemeyi büyütebilir, zayıflatabilir veya support
gradyanına ters yön oluşturabilir. Clipping dönüşümden önce olduğundan
dönüştürülmüş gradyanı sınırlandırmaz. Statik faktörler bütün domain'lere
ortak olduğundan bazı domain'lerde yararlı yönler diğerlerinde zararlı
olabilir. First-order yaklaşımı, support gradyanının fast-weight'e
bağımlılığını türevlemez; öğrenilmiş dönüşümün gradient tahmini de bu
yaklaşımın sınırlarını taşır. Rank=4 bir başlangıç tercihi, optimal olduğuna
dair kanıt değildir.

Bu nedenle sonuç, aynı veri/model seed'leri ve protokolle SGMAML'a karşı
query/validation başarımı ve domain bazlı sonuçlarla değerlendirilmelidir.
Daha düşük training loss veya büyüyen faktör normları tek başına başarı
kanıtı değildir. Yöntem burada tarif edildiği şekliyle uygulanmıştır;
bu riskleri bastırmak için ek bir kısıt veya normalizasyon eklenmemiştir.

## Dosyalar ve doğrulama durumu

- `model.py`: submission arayüzü, outer eğitim, validation, birlikte best
  snapshot alma, save/load ve task adaptasyonu.
- `helpers_lrsgmaml.py`: mevcut yardımcılar ve scalar + low-rank inner update.
- `low_rank_transport.py`: isim eşleştirmesi, U/V, RNG korumalı oluşturma,
  checkpoint doğrulama ve tanı toplama.
- `network.py`, `weight_names.py`, `api.py`, `metadata`: SGMAML'dan kopyalanan
  backbone ve submission sözleşmesi.
- `config.json`: mevcut config ve `low_rank.rank`.
- `tests/test_lrsgmaml.py`: küçük ağlarla başlangıç eşitliği, gradient yolu,
  optimizer/buffer üyeliği, checkpoint, any-way, RNG, tanı ve snapshot testleri.

Kullanıcının talebi doğrultusunda bu ekleme sırasında eğitim veya test
çalıştırılmamıştır. Kontrol, kaynak kodu ve diff incelemesiyle sınırlıdır.
Testlerin geçmesi, gerçek veri/GPU davranışı, hız/bellek maliyeti ve yöntemin
SGMAML'a göre başarı kazancı henüz doğrulanmış değildir.
