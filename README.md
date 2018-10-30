# arch
Kaby Lake line
7.1. Introduction
-	a 7. generációs Kaby Lake valójában egy Skylake refresh
-	főbb újdonságai: optane memory, in KBL G series: in package integrated CPU, GPU, HBM2
-	energiaszabályozás: Speed Shift Technology v2
7.3. Főbb innovációk
-	optane memory -> a PCH-hoz köthető:
o	nem felejtő memoria, tipikusan HDD drive cache-ként használják
o	3D XPoint memoria technológián alapul
o	tipikus mérete 16 vagy 32 GB
o	M.2 kártyára van felszerelve, ami 2-4 PCIe sávval van csatlakoztatva
o	használatához szükség van a Rapid Storage Technology driver-re
7.2. Újdonságok energiaszabályozásban
-	Speed Shift Tech. előnyei: gyorsabb frekvencia váltások -> csökkenti a task befejezését (task completion)
7.4. Példa: A 8. generációs mobil Kaby Lake G series
-	a 8. gen Intel Core fejlődésa: KLR (Kaby Lake Refresh) (mobil) -> CL (Coffee Lake) (desktop) -> KL G: Kaby Lake G (mobil)
-	több die összekapcsolása EMIB (Embedded Multi-Die Interconnect Bridge) segítségével
-	az AMD Vega CPU-jának jelentősen nagyobb a teljesítménye, mint az Intel integrált HD Graphics 630-jának
-	az integrált CPU, GPU és HBM2 megoldás további előnye: less board space
-	megjegyzés: Cinebench: cross platform tesztcsomag, ami kiértékeli a számítógép teljesítményét. MAXON animációs szoftverén (Cinema 4D) alapul, amit stúdiók és produkciós házak világszerte használnak 3D-s tartalmak készítése során. Gyakran használjak a processzorok grafikus képességének értékelésére.

 
Coffee Lake Line
9.1. Bevezetés
-	újdonságok: 6C (USB G2, integrated connectivity, Optane 2)
-	energiaszabályozás: In H-series: TVB (Thermal Velocity Boost)
9.3. Példa: DT Coffee Lake S-series második hulláma
CL (Coffee Lake) – 8. generation intel core, desktopok, Broad Consumer Processors
Újdonságai:
-	USB gen. 2 support
-	integrated connectivity
-	enhanced optane support
USB Gen2 support:
-	az USB szabvány fejlődése
o	USB 1.0 (1996) – átviteli sebesség: max. 1.5 MB/s
o	USB 2.0 (2000) – átviteli sebesség: max. 60 MB/s
o	USB 3.0 (2008) – SuperSpeed átviteli ráta max. 625 MB/s
o	USB 3.1 (2013) – két alternatíva van: USB 3.1 Gen 1 és USB 3.1 Gen 2 mely SuperSpeed+ átviteli rátával akár 1.25 GB/s
o	USB 3.2 (2017) – két új SuperSpeed+ átviteli mód akár 1.25 és 2.5 GB/s átviteli rátával USB-C csatlakozón keresztül
-	az USB 3.0 szabványhoz tartozó SuperSpeed nagy átviteli sebességét a SuperSpeed busz teszi lehetővé, ami két serial point-to-point adatsínt és egy föld vonalat valósít meg -> újfajta csatlakozóra volt szükség
 
Integrated Connectivity Support
-	a 802.11ac Wi-Fi, BT és RF blokkok részbeni integrálása a PCH-ra
-	802.11ac Wi-Fi: 2013, 5 GHz frekvencia, 20-40-80-160 MHz sávszélesség, MIMO technology (többször átviteli csatorna, akár 8), 346-800-1733-3466 MB/s adatátviteli sebesség
MIMO technológia alapja
	azért mutatták be (az IEEE 801.11n módosításban), hogy jelentősen növeljék az adatátviteli sebességet
	feltételezi, hogy több adó és vevő antenna van és támogatja több kommunikációs csatornát, melyek egyidejűleg küldhetnek adatot
	az adatfolyamot alfolyamokra osztja, ezeket pedig párhuzamosan küldi át a csatornán, a vevő oldalon pedig összeállítja
	a 801.11n 4 csatornát támogat
Intel’s Integrated Connectivity (CNVi)
	általában van egy Wi-Fi/BT/RF modul, amit a processzortól eltérően helyeznek el
	Integrated Connectivitiy-vel az Intel részben a PCH-ra integrálja a Wi-Fi/BT/RF modult
	ebben az implementációban az intel a Wi-Fi/BT/RF modul funkcionális blokkjait, mint a logika, MAC (Multplier-Accumulator) és memória, a PCH-ra, azon belül egy Pulsar nevű blokkra helyezi, más részel (mint a PHY – Physical layer és RF) maradnak a companion RF modulon (CRF, másnéven Jefferson Peak)
	A CRF modult egy M.2 kártyán implementálják, a PCH-hoz a CNVio nevű interfészen keresztül csatlakozik
Enhanced Optane Support
-	(ism. előző gen.)
-	a 2. gen feltételezi a gyors SSD Boot drive-ot és egy nagy HD adatdrive-ot.
-	a Core i9+, Core i7+, Core i5+ -> a + jelöli az Optane technológia támogatását, 8. generációval mutatták be, sötétkék logója van
