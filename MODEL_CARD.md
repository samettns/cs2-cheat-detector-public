# Model kartı — Model v1

Bu depo **hazır eğitilmiş Model v1** ile çıkarım (analiz) yapar. Modeli eğiten ve değerlendiren tam araştırma hattı
(çift seçimi, çapraz doğrulama, dondurma, duyarlılık analizleri) **bu public sürümde bulunmaz**; burada yalnız
eğitilmiş model, onun referans dosyaları ve analiz için gereken kod vardır.

> Model skoru **kalibre edilmiş bir hile olasılığı değildir** ve tek başına bir hile kararı veya suçlama
> dayanağı olarak kullanılamaz. Bkz. README → *Sorumlu kullanım*.

## Özet

| alan | değer |
|---|---|
| dosya | `models/v1/model_B.joblib` (scikit-learn, `joblib`) |
| algoritma | `HistGradientBoostingClassifier` (`class_weight="balanced"`, `learning_rate=0.05`, `max_depth=3`, `max_iter=200`, `random_state=0`) |
| scikit-learn sürümü | 1.9.1 (`requirements-lock.txt` ile aynı) |
| girdi | öldürme başına 15 özellik (`models/v1/features.json`, sıra önemlidir) |
| çıktı | öldürme skoru (0–1); oyuncu skoru = en az 3 öldürmenin skor ortalaması |
| eksik değer | doldurulmaz (modelin yerel NaN desteği) |
| durum | dondurulmuş; özellik, eşik ve ağırlıklar bu sürümde değişmez |

## Eğitim verisi

- **Kaynak:** CS2CD — Counter-Strike 2 Cheat Detection (Hugging Face, CC-BY-4.0). Atıf ve lisans ayrıntısı:
  [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
- **Seçim:** her biri bir hileli + bir hilecisiz maçtan oluşan **160 eşleştirilmiş çift** (aynı harita, aynı mod,
  benzer uzunluk). Çift listesi: `data/pairs.csv` (50) + `data/pairs_new.csv` (110). Toplam 320 maç, 27 889 öldürme,
  en az 3 öldürmeli 1 960 oyuncu.
- **Etiketler:** oyuncu düzeyinde. Hileci örnekleri yalnız hileli maçlardaki hileci etiketli oyunculardan, temiz
  örnekleri yalnız hilecisiz maçlardan alındı. Hileli maçlardaki "temiz" etiketli oyuncular güvenilir olmadığı için
  eğitimde kullanılmadı.
- **Ham veri bu depoda yoktur;** yalnız türetilmiş özellik/skor özetleri vardır (`data/reference/v1/`).

## Özellikler

| grup | özellik | anlamı |
|---|---|---|
| nişan hareketi | `speed_mean`, `speed_max`, `speed_last_mean` | öldürmeden önceki 1 sn / son 250 ms nişan hızı |
| | `jitter`, `still_ratio`, `pitch_range_last` | titreklik, hareketsiz nişan oranı, son 250 ms dikey nişan aralığı |
| | `after_move` | öldürmeden sonraki 500 ms nişan dönüşü |
| atış ve silah | `shots`, `is_sniper`, `is_pistol` | son 1 sn atış sayısı, silah sınıfı |
| görünürlük | `inv_angle_med`, `inv_frac_lt10` | rakip görünmezken nişanın rakibe açısı; 10°'den yakın olma oranı |
| | `prefov_angle` | rakip görünür olmadan ~190 ms önceki nişan açısı |
| | `wallbang`, `reaction_ms` | duvar arkasından isabet; görünür olma → ilk isabet süresi |

**Görünürlük.** "Rakip görünür müydü?" sorusu oyunun `spotted` bayrağıyla değil, harita geometrisi (`.tri`)
üzerinde göz yüksekliğinden ışın izleme (ray-cast, awpy) ile cevaplanır. `spotted` tek başına kullanılmaz: dürbünlü
silahlarda yanlı davranır. Ray-cast'in, delmeyen isabetlerde hedefi "görünür" bulma oranı CS2CD'de harita başına
%95'in üstündeydi. Ölçümü bozan küçük bir bölge (Anubis'teki su kanalı) maskelenmiştir.

## İnceleme önceliği (karar katmanı)

Oyuncu stili, öldürmelerin silah sınıfı paylarına göre belirlenir (sniper payı > %50 → *sniper ağırlıklı*;
otomatik payı > %50 → *otomatik ağırlıklı*; diğerleri *diğer*). Skor, **aynı stildeki temiz CS2CD oyuncularının**
skor dağılımıyla karşılaştırılır (stil içi yüzdelik). Bunun nedeni, tek bir global eşiğin sniper stilini
sistematik olarak fazla işaretlemesidir.

| öncelik | kural |
|---|---|
| yüksek | skor > stil içi temiz oyuncuların %95'lik dilimi (hedef temiz yanlış pozitif oranı %5) |
| orta | skor > %90'lık dilim |
| düşük | altında |
| yetersiz veri | 3'ten az öldürme |

Eşikler `models/v1/thresholds.json` içindedir: sniper ağırlıklı 0.863 (164 temiz oyuncu), otomatik ağırlıklı 0.359
(920), diğer 0.669 (310). **İnceleme noktası:** tek bir öldürmenin skoru, aynı stildeki temiz öldürmelerin %95'lik
diliminden yüksekse işaretlenir (`data/reference/v1/reference.json`). Açıklamalar ("neden bu öncelik?") her
özelliğin aynı stildeki temiz referans içindeki yüzdeliğinden üretilir; model içinden açıklama (ör. SHAP) değildir.

## Değerlendirme

Değerlendirme maç çifti bazında yapıldı: bir çiftin iki maçı ya eğitimde ya testte. Böylece aynı maçtan sızıntı
olmaz.

| değerlendirme | nişan + silah | + görünürlük (Model v1 yapısı) |
|---|---|---|
| ilk 50 çift, çift bazlı çapraz doğrulama | AUC 0.813 [0.76–0.86] | AUC 0.856 [0.81–0.90] |
| sonraki 110 çift (yalnız ilk 50 çiftle eğitilmiş model) | AUC 0.872 [0.836–0.903] | AUC 0.913 [0.884–0.937] |

(Oyuncu düzeyi AUC, köşeli parantez içinde %95 güven aralığı.)

**Dikkat:**
- 110 çiftlik setin küçük bir kısmı geliştirme sırasında önceden görülmüştü; tamamen bağımsız sayılmamalıdır.
- Model v1 bu iki setin birleşimiyle (160 çift) eğitildiği için ondan bağımsız, hiç kullanılmamış bir test seti
  **kalmamıştır**.
- Bu sonuçlar bir sinyal olduğunu gösterir; kanıt değildir.
- Etiketler gürültülüdür: hileli maçlarda çoğu zaman bütün takım hileci etiketlidir.
- **Sniper ağırlıklı oyuncularda** ayırt etme gücü belirgin biçimde düşüktür (stil içi AUC ~0.75).

## Eğitim maçlarının gösterimi (OOF)

Arayüzdeki CS2CD örnekleri modelin eğitim havuzundandır. Bu maçlar, onları **görmemiş** çapraz doğrulama
katlarının skorlarıyla (out-of-fold, OOF) gösterilir (`data/reference/v1/oof_all_kills.parquet`). Veri setindeki
etiket ayrı bir alanda gösterilir ve model çıktısı değildir.

**Örnek seçimi:** `samples/` altındaki üç örnek, sonuçlara bakmadan önce yazılmış sabit bir kuralla (seed 2026)
seçildi. Kuralın çıktısı `data/reference/v1/jury_set.json`, kodu `src/jury_set.py`. Örnekler: hileci etiketli
oyuncu içeren bir maç, hilecisiz bir maç ve bilinen bir yanlış alarm (sniper ağırlıklı temiz oyuncu).

## Dağılım uyarısı

Analiz edilen demo, modelin eğitildiği temiz maçlardan çok farklıysa arayüz **demo düzeyinde** bir uyarı gösterir.
Uyarı skorları değiştirmez. Referans, 160 temiz eğitim maçının maç başı nişan değişimi (P99) ve medyan tepki süresi
dağılımıdır (`data/reference/v1/distribution_reference.json`).
- %2.5–97.5 aralığı dışı: "dağılım farkı gözlendi"
- %0.5–99.5 dışı: "temiz eğitim referansından farklı"
- tepki süresi için en az 20 ölçüm gerekir

## Amaçlanan kullanım ve kapsam dışı

- **Amaçlanan:** CS2CD benzeri matchmaking maçlarında, bir insanın hangi oyuncu ve anlara önce bakacağını
  sıralamak.
- **Kapsam dışı:** otomatik ceza, ban veya hüküm; bir kişiyi suçlamak veya ifşa etmek.
- **Dikkatle yorumlanmalı:** profesyonel veya çok yüksek becerili oyuncular (yanlış pozitif riski), sniper ağırlıklı
  oyuncular, 64 tick dışındaki demolar, desteklenen 9 harita dışındaki haritalar.

## Güvenlik notu

`model_B.joblib` bir pickle dosyasıdır. Yalnız bu depodan gelen dosyayı yükleyin; güvenmediğiniz bir kaynaktan
gelen `.joblib` dosyasını açmayın.

## Dosya meta verisi hakkında

`models/v1/thresholds.json` ve `data/reference/v1/reference.json` içinde geçen `notes/...` dosya adları ile `K1`,
`K3`, `D26–D27` gibi karar numaraları, yayımlanmamış geliştirme notlarına ait adlardır ve hiçbir içerik taşımaz;
dosyalar, Model v1'in kayıtlı kimlik hash'i değişmesin diye bayt olarak olduğu gibi bırakılmıştır.

## Lisans ve sorumluluk reddi

- Model ağırlıkları (`models/v1/`) ve referans dosyaları **CC-BY-4.0** koşullarıyla, CS2CD'ye atıf ve değişiklik
  bildirimiyle paylaşılır. Ayrıntı: [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
- Model **araştırma ve eğitim amacıyla hazırlanmış ve bu amaçla sunulmaktadır.** Bu, amaçlanan kullanımı anlatan bir
  açıklamadır; CC-BY-4.0 koşullarına ek bir lisans kısıtlaması değildir.
- Model ve dosyalar **"olduğu gibi" (as is)**, hiçbir doğruluk, uygunluk veya başka bir garanti verilmeden sunulur.
- Eğitilmiş model ağırlıklarının hukuki niteliği (veri setinin bir uyarlaması sayılıp sayılmadığı) genel olarak
  netleşmiş değildir. CC-BY-4.0 seçimi ihtiyatlı bir yaklaşımdır; hukuken kesin bir değerlendirme değildir.
- CS2CD'nin üst veri kaynağına (veri kartına göre csstats.gg üzerinden toplanan maç verisi) ilişkin haklar
  geliştirici tarafından **doğrulanamamıştır**.
