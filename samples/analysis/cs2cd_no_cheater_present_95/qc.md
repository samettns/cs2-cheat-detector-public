# Kalite kontrol: cs2cd_no_cheater_present_95

**Model v1 için: UYGUN** 

_Bu rapor oyuncu sınıflandırması ve model skoru içermez. Oyuncular anonimdir (Player_N)._

## Demo
- Kaynak: cs2cd | harita: cs_office (destekli ✅) | patch None | format None
- Tick hızı: **64.0** (beklenen 64) | ardışık tick farkları [(1, 78482), (4, 12), (2, 6)] | süre 20.4 dk
- Round: 16 | maç başlangıç tick'i: 2097 | oyuncu: 10 | takım başına anlık (takım, medyan, max): [(2.0, 5.0, 5), (3.0, 5.0, 5)]

## Öldürmeler
- Bütün ölümler: 118 (maç öncesi: 0, takım öldürme: 2)
- Nişan özelliği hesaplanan: 56 → maç öncesi çıkarılan: 0 → **modele girecek: 56** (8 oyuncu)

## 15 özellik (models/v1/features.json sırası)

| # | özellik | dolu | NaN |
|---|---|---|---|
| 1 | speed_mean | 1.000 | 0.000 |
| 2 | speed_max | 1.000 | 0.000 |
| 3 | speed_last_mean | 1.000 | 0.000 |
| 4 | jitter | 1.000 | 0.000 |
| 5 | still_ratio | 1.000 | 0.000 |
| 6 | pitch_range_last | 1.000 | 0.000 |
| 7 | shots | 1.000 | 0.000 |
| 8 | after_move | 1.000 | 0.000 |
| 9 | is_sniper | 1.000 | 0.000 |
| 10 | is_pistol | 1.000 | 0.000 |
| 11 | inv_angle_med | 1.000 | 0.000 |
| 12 | inv_frac_lt10 | 1.000 | 0.000 |
| 13 | prefov_angle | 0.625 | 0.375 |
| 14 | wallbang | 1.000 | 0.000 |
| 15 | reaction_ms | 0.625 | 0.375 |

NaN'lar doldurulmaz (model native NaN ile eğitildi). Pre-FOV/tepki/görünmezken açı NaN'ları tanım gereği (ör. görünürlük geçişi yok) oluşur; CS2CD'de de benzer oranlarda.

## Görünürlük ve ray-cast
- Görünürlük hesaplanabilir: ✅
- **Eski ölçüt (`bullet_damage`)**: delmeyen isabetlerde görünür 0.912 (n=80); delenlerde 0.276 (n=116). CS2CD 9 haritada delmeyen %95–99.
- Yedek ölçüt: 1.0 (n=28)
- ⚠️ Yedek ölçüt (öldürmeden önceki son örnek anda kurban görünür mü; delmeyen + dumansız öldürmeler) eski bullet_damage ölçütüyle (her delmeyen isabet, isabet anında) doğrudan eşdeğer DEĞİLDİR.

## Diğer
- Nişan değişimi P99 (tick'ten tick'e açı): 6.073 (CS2CD maç başı tipik aralık ~4.8–8.9)
- Harita sınırı dışındaki canlı konum (8 Hz örnek): 0
- Bıçak skini öldürmeleri: toplam 0, modele giren 0. v1 bilinen hatası: bıçak skinleri (knife_*) nişan gerektirmeyen silah listesinde yok → 'otomatik' sınıfına düşer; CS2CD eğitiminde de aynıydı. Model v1 donuk olduğu için düzeltilmedi.

## Uyarılar
- yok
