# Veri atfı ve lisanslar

Bu depodaki **kod** MIT lisanslıdır ([LICENSE](LICENSE)). CS2CD veri setinden türetilen dosyalar koddan ayrı
değerlendirilir ve aşağıdaki koşullarla paylaşılır.

## Kaynak veri seti

| alan | değer |
|---|---|
| ad | CS2CD — Counter-Strike 2 Cheat Detection |
| yayıncı | CS2CD (Hugging Face) |
| adres | https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection |
| DOI | [10.57967/hf/5654](https://doi.org/10.57967/hf/5654) |
| lisans | [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/) |
| veri toplama belgesi | "Counter-Strike 2 Game data collection with cheat labelling", Mille Mei Zhen Loo & Gert Lužkov (veri kartında bağlantılı) |

Veri kartı, veri setini kullananlardan uygun biçimde atıf istemektedir. Veri kartına göre veri csstats.gg
üzerinden toplanmış ve `.dem` dosyalarındaki hassas bilgiler veri setinden çıkarılmış veya değiştirilmiştir
(oyuncular anonim, `Player_1..10`).

Önerilen atıf:

> CS2CD. *Counter-Strike 2 Cheat Detection* [veri seti]. Hugging Face, doi:10.57967/hf/5654. CC-BY-4.0.

**Değişiklik bildirimi (CC-BY-4.0 §3(a)(1)(B)):** Bu projede CS2CD'nin tick ve olay verisinden öldürme başına
davranış özellikleri çıkarılmış, bu özelliklerle bir sınıflandırıcı eğitilmiş ve değerlendirilmiş, maç başına skor
ve özet çıktıları hesaplanmıştır. **Ham CS2CD verisi bu depoda yeniden dağıtılmaz.**

## Dosya grupları

| dosya | içerik | kaynağı | paylaşım koşulu |
|---|---|---|---|
| `data/pairs.csv`, `data/pairs_new.csv` | eğitimde kullanılan 160 maç çifti: CS2CD maç numarası, harita, mod, dosya boyutu | CS2CD maç listesi ve meta verisi | CC-BY-4.0, atıfla |
| `data/reference/v1/*.parquet` | CS2CD temiz oyuncu/öldürmelerinin özellik değerleri ve çapraz doğrulama skorları; dağılım uyarısı için maç başı özetler | CS2CD verisinden hesaplanmış türev veri | CC-BY-4.0, atıfla |
| `data/reference/v1/*.json` | eşik, açıklama yönleri, dağılım referansı, örnek seçim kuralı çıktısı | yukarıdaki türev veriden hesaplanmış özet sayılar | CC-BY-4.0, atıfla |
| `samples/analysis/` | 3 CS2CD maçının anonim analiz çıktıları (özellikler, konum/olay özetleri, OOF skorları, veri seti etiketi) | CS2CD maçları (zaten anonim) | CC-BY-4.0, atıfla |
| `models/v1/` | eğitilmiş model ağırlıkları (`model_B.joblib`), özellik listesi, eşikler, temiz referans skorları | CS2CD üzerinde eğitilmiş model ve türev özetler | CC-BY-4.0, atıfla (aşağıdaki nota bakın) |

**Model ağırlıkları hakkında not:** Eğitilmiş model ağırlıklarının hukuken veri setinin bir "uyarlaması" (Adapted
Material) sayılıp sayılmadığı genel olarak netleşmiş bir konu değildir. Bu proje ihtiyatlı bir yaklaşımla ağırlıkları
kaynak veriyle aynı koşullarla (CC-BY-4.0, atıf ve değişiklik bildirimiyle) paylaşır. Bu seçim hukuken kesin bir
değerlendirme değildir ve CS2CD'nin kendi verisi üzerindeki haklar hakkında ek bir iddia içermez.

## Sorumluluk reddi

- Model ve türetilmiş dosyalar **araştırma ve eğitim amacıyla hazırlanmış ve bu amaçla sunulmaktadır.** Bu,
  amaçlanan kullanımı anlatan bir açıklamadır; CC-BY-4.0 koşullarına ek bir lisans kısıtlaması değildir.
- Tüm dosyalar **"olduğu gibi" (as is)**, hiçbir doğruluk, uygunluk veya başka bir garanti verilmeden sunulur.
- CS2CD'nin üst veri kaynağına (veri kartına göre csstats.gg üzerinden toplanan maç verisi) ilişkin haklar
  geliştirici tarafından **doğrulanamamıştır**. Bu depo ham CS2CD verisini yeniden dağıtmaz; yalnız türetilmiş
  özellik, özet ve model dosyalarını içerir.

## Depoda olmayanlar

- **Ham CS2CD verisi** (`.parquet` tick tabloları, `.json` olayları): Hugging Face'ten indirilir.
- **Valve dosyaları:** harita geometrisi (`.tri`) ve radar görselleri Valve Corporation'a aittir. Depoda yoktur;
  kullanıcının kendi oyun kurulumundan yerelde üretilir (README → *Harita dosyaları*).
- **Güncel maç demoları** (Valve / FACEIT / HLTV) ve bunlardan türetilmiş hiçbir örnek.

Bu proje Valve Corporation, csstats.gg veya CS2CD yazarlarıyla bağlantılı değildir ve onlar tarafından
onaylanmamıştır. Counter-Strike, Valve Corporation'ın ticari markasıdır.
