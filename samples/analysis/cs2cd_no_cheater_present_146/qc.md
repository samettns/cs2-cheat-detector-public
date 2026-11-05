# Kalite kontrol: cs2cd_no_cheater_present_146

**Model v1 için: UYGUN** 

_Bu rapor oyuncu sınıflandırması ve model skoru içermez. Oyuncular anonimdir (Player_N)._

## Demo
- Kaynak: cs2cd | harita: de_inferno (destekli ✅) | patch None | format None
- Tick hızı: **64.0** (beklenen 64) | ardışık tick farkları [(1, 129475), (2, 21), (6, 12)] | süre 33.7 dk
- Round: 21 | maç başlangıç tick'i: 2586 | oyuncu: 10 | takım başına anlık (takım, medyan, max): [(2.0, 5.0, 5), (3.0, 5.0, 5)]

## Öldürmeler
- Bütün ölümler: 151 (maç öncesi: 0, takım öldürme: 0)
- Nişan özelliği hesaplanan: 144 → maç öncesi çıkarılan: 0 → **modele girecek: 144** (10 oyuncu)

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
| 11 | inv_angle_med | 0.972 | 0.028 |
| 12 | inv_frac_lt10 | 0.972 | 0.028 |
| 13 | prefov_angle | 0.799 | 0.201 |
| 14 | wallbang | 1.000 | 0.000 |
| 15 | reaction_ms | 0.799 | 0.201 |

NaN'lar doldurulmaz (model native NaN ile eğitildi). Pre-FOV/tepki/görünmezken açı NaN'ları tanım gereği (ör. görünürlük geçişi yok) oluşur; CS2CD'de de benzer oranlarda.

## Görünürlük ve ray-cast
- Görünürlük hesaplanabilir: ✅
- **Eski ölçüt (`bullet_damage`)**: delmeyen isabetlerde görünür 0.99 (n=519); delenlerde 0.846 (n=26). CS2CD 9 haritada delmeyen %95–99.
- Yedek ölçüt: 1.0 (n=135)
- ⚠️ Yedek ölçüt (öldürmeden önceki son örnek anda kurban görünür mü; delmeyen + dumansız öldürmeler) eski bullet_damage ölçütüyle (her delmeyen isabet, isabet anında) doğrudan eşdeğer DEĞİLDİR.

## Diğer
- Nişan değişimi P99 (tick'ten tick'e açı): 6.244 (CS2CD maç başı tipik aralık ~4.8–8.9)
- Harita sınırı dışındaki canlı konum (8 Hz örnek): 0
- Bıçak skini öldürmeleri: toplam 0, modele giren 0. v1 bilinen hatası: bıçak skinleri (knife_*) nişan gerektirmeyen silah listesinde yok → 'otomatik' sınıfına düşer; CS2CD eğitiminde de aynıydı. Model v1 donuk olduğu için düzeltilmedi.

## Uyarılar
- yok
