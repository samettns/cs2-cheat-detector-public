# Kalite kontrol: cs2cd_with_cheater_present_36

**Model v1 için: UYGUN** 

_Bu rapor oyuncu sınıflandırması ve model skoru içermez. Oyuncular anonimdir (Player_N)._

## Demo
- Kaynak: cs2cd | harita: de_mirage (destekli ✅) | patch None | format None
- Tick hızı: **64.0** (beklenen 64) | ardışık tick farkları [(1, 151766), (2, 24), (3, 13)] | süre 39.5 dk
- Round: 30 | maç başlangıç tick'i: 2799 | oyuncu: 10 | takım başına anlık (takım, medyan, max): [(2.0, 4.0, 5), (3.0, 5.0, 5)]

## Öldürmeler
- Bütün ölümler: 156 (maç öncesi: 0, takım öldürme: 0)
- Nişan özelliği hesaplanan: 151 → maç öncesi çıkarılan: 0 → **modele girecek: 151** (7 oyuncu)

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
| 13 | prefov_angle | 0.278 | 0.722 |
| 14 | wallbang | 1.000 | 0.000 |
| 15 | reaction_ms | 0.278 | 0.722 |

NaN'lar doldurulmaz (model native NaN ile eğitildi). Pre-FOV/tepki/görünmezken açı NaN'ları tanım gereği (ör. görünürlük geçişi yok) oluşur; CS2CD'de de benzer oranlarda.

## Görünürlük ve ray-cast
- Görünürlük hesaplanabilir: ✅
- **Eski ölçüt (`bullet_damage`)**: delmeyen isabetlerde görünür 0.902 (n=41); delenlerde 0.181 (n=160). CS2CD 9 haritada delmeyen %95–99.
- Yedek ölçüt: 0.704 (n=27)
- ⚠️ Yedek ölçüt (öldürmeden önceki son örnek anda kurban görünür mü; delmeyen + dumansız öldürmeler) eski bullet_damage ölçütüyle (her delmeyen isabet, isabet anında) doğrudan eşdeğer DEĞİLDİR.

## Diğer
- Nişan değişimi P99 (tick'ten tick'e açı): 4.86 (CS2CD maç başı tipik aralık ~4.8–8.9)
- Harita sınırı dışındaki canlı konum (8 Hz örnek): 0
- Bıçak skini öldürmeleri: toplam 0, modele giren 0. v1 bilinen hatası: bıçak skinleri (knife_*) nişan gerektirmeyen silah listesinde yok → 'otomatik' sınıfına düşer; CS2CD eğitiminde de aynıydı. Model v1 donuk olduğu için düzeltilmedi.

## Uyarılar
- ⚠️ takım başına anlık oyuncu medyanı [(2.0, 4.0, 5), (3.0, 5.0, 5)] (beklenen 5)
