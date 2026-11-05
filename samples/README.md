# Örnek analizler (anonim)

Arayüz bu klasördeki analizleri `reports/analysis/` ile birlikte listeler.
Ham demo, Steam ID veya gerçek oyuncu adı içermez (`python src/sample_pack.py check`).

Örneklerin hepsi CS2CD veri setinden türetilmiştir: CS2CD — Counter-Strike 2 Cheat Detection
(Hugging Face `CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection`), lisans CC-BY-4.0. Oyuncular veri setinde zaten
anonimdir (`Player_1..10`). Bu klasördeki dosyalar o veriden türetilmiş özellik ve model çıktılarıdır; skorlar
maçı görmemiş kat modellerinden (OOF) gelir.

| örnek | ne gösteriyor |
|---|---|
| `cs2cd_with_cheater_present_36` | veri setinde hileci etiketli oyuncu bulunan maç |
| `cs2cd_no_cheater_present_146` | hilecisiz maç: temiz oyuncularda öncelik dağılımı |
| `cs2cd_no_cheater_present_95` | bilinen yanlış alarm: sniper ağırlıklı temiz oyuncu yüksek öncelik alıyor |
