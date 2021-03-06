1))
Arról az alkalmazásról szeretnék egy bemutatót tartani, amely támogatja ill. megvalósítja a teszt vezetések során készített videók automatikus taggelését.

I would like do a short presentation about the application we have been working on. This application supports the automatic tagging of the videos the were recorded during test drives.

2))
Kezdetben a teszt vezetések során két emberre volt szükség: egy vezetőre és egy mérnökre, aki a taggelést kézzel végzi el.
A projekt célja az volt, hogy ezt a taggelést automatizáljuk, hogy a későbbiekben már offline, mesterséges intelligencia segítségével történhessen.

At first, two people were needed during test drives: a driver, and an engineer who did the tagging manually.
The goal of this project was to make this process of tagging automatic, so tagging is done offline with the help of artificial intelligence.

3))
A gépi látás fejlődésének köszönhetően egyre több szabadon hozzáférhető képfelismerő alkalmazás létezik.
A teszt vezetések során fontos az, hogy ismerjük a környezetet (hogy tudjuk azt, milyen objektumok zavarhatják a GNSS jelét). -> Ezt a feladatot pedig megoldhatjuk mesterséges intelligencia segítségével.
Az automatikus taggeléshez a képfelismerés mellett további mérések is felhasználhatók.

There are many publicly accessible applications for image recognition thanks to the rapid developmebnt of computer vision technologies.
It’s beneficial to know our enviroment during test drive evaluation -> this task can be done using AI
Image recognition is not the only way to gather information about the environment, there other measurements that can be used to identify certain objects of interest.

4))
Az alkalmazást python nyelven valósítottuk meg, és felhasználjuk a TensorFlow Object Detection API-ját. 
A TensorFlow egy nyílt forráskódú programkönyvtár, amely támogatja a gépi tanulást, a neurális hálózatok használatát.
Az objektum detektáláshoz Faster R-CNN-t használtunk, már létező módszer, ami azt határozza meg, hogy az adott képkereten hogyan kerülnek kiválasztásra a számunkra fontos régiók.

The application was written in Python, we used TensorFlow’s Object Detection API
TensorFlow is an open source machine learning framework
For object detection we used Faster R-CNN -> which is used which combines rectangular region proposals with convolutional neural network features.

5))
Az alábbi ábra szemlélteti a szoftvernek a felépítését.
Van egy main script, aminek a bemenete a taggelendő videó
Ebből kinyerjük az egyes képkereteket, az Object detection API pedig keretenként szolgáltat eredményeket
Az így kapott eredményeket a main script kigyűjti egy mátrixba (sorok = képkeret száma, oszlopok = detektálandó objektumok)
Végül az eredményeket kiírja egy excel fájlba

Here’s a quick overview of the software.
There is main script, the input is the video file recorded by the measurement camera
The script extracts the frames from the video, and the object detection application provides a result for every frame
The results are stored in a matrix (rows represent the number of the frame, columns represent the detected object)
The results are written into an excel file

6))
A neurális hálózatot a LabelImg nevű program segítségével tanítjuk be.
Minden detektálandó objektumről szükség van néhány száz tanuló képre, ezeknek az annotációja elvégezhető egy grafikus felüleleten
Az így keletkező szöveges fájlok és a képek segítségével pedig már betanítható a hálózat.

The neural network is trained using LabelImg
It provides a graphical interface to annotate images, each object needs a couple hundred example images
Using the annotated images we can train the neural network

7))
Ezen a dián látható néhány példa a detektálás eredményére. Néhány száz tanuló képpel már viszonylag pontos eredményeket érhetünk el.

This slide shows the results. We can achieve relatively precise results using about two hundred images per object.

8))
A kimeneti fájlt az alábbi formában állítjuk elő.
Ezen a képen látszik, hogy éppen elhaladtunk egy gantry alatt.

This slide shows the output file.
After passing a gantry this is the result.

9))
Video

====================================================================================================================
SzA55. A Skylake család II.
(bevezetett innovációk: az ISP, Memory Side Cache, a Speed Shift technológia mint az Enhanced SpeedStep technológia továbbfejlesztése)

* Integrált ISP (Image Signal Processing)
- Komplett kamera + képalkotás megoldás: teljes hardver és szoftver integráció; optimalizált modul/szenzor támogatás (akár 4 kamera, akár 13 MP-es szenzorok); prémium minőségű képalkotás
- Az ISP integrálva van a hatékonyabb működés érdekében

* Memory Side Cache (41. old ábra)
- teljesen koherens cache
- elérhető az IO eszközök és megjelenítő engine számára

* energiafelhasználás szabályozására vonatkozó innovációk
>> Speed Shift technológia
- korábbi technológia: DVFS: ha a munkamenet a P1 teljesítmény állapotnál magasabb teljesítményt kíván, akkor a processzor átveheti az energiagazdálkodás vezetését, és aktiválja a Turbo Boost technológiát; P1...PN állapotok a garantált és a még hatékony teljesítmény szintet jelölik; a PN állapotnál alacsonyabb állapotok csak kritikus feltételek mellett lépnek életbe
- a SpeedStep legnagyobb hátránya a lassú reagálás (kb 30 ms) -> kb ennyi idő kell, amíg az OS észreveszi, hogy a működési feltételek megváltoztak. a SST ezt az időt leredukálja 1 ms-re
- az OR továbbadja az energiagazdálkodás vezérlését a processzornak (ezen belül a PCU-nak) -> a PCU végzi el a P-állapotok kezelését (ez a Speed Shift Tech (intel) vaghy Autonomous Control (intel) vagy Hardware Controlled Performance States (ACPI) -> ez pedig már az összes P állapotot kezelni tudja

>> Duty Cycle control
(le van írva - tételnek nem része)

====================================================================================================================
SzA56. A Kaby Lake család
(bevezetett innovációk: a Speed Shift technológia 2. generációja, az Optane memória és az M.2 interfész szabvány) 

- 7. generáció (Skylake refresh), 14 nm technológia

* Speed Shift v2: a frekviencia változások gyorsabban mennek végbe -> a task befejezése gyorsabb (>2x gyorsabb); a 7. generációs processzorok 15 ms alatt elérik a max. órafrekvenciát

* Optán memória
- a PCH-hoz kötődő innováció
- nem-felejtő memória, amit tipikusan a HDD cache-eként használják
- tipikus mérete 16 vagy 32 GB
- M.2 kártyára van felszerelve, ami 2 vagy 4 PCIe sávval van csatlakoztatva
- Rapdi Storage Technology driver kell a használatához

* M2 interfész szabvány
- interfész specifikáció a belülre felszerelt kiegészítő kártyákhoz
- az mSATA sztenderdet hivatott lecserélni, különböző modul szélességet/hosszúságot kínál
- alkalmas kis eszközökhöz, vékony laptopokhoz és tabletekhez

====================================================================================================================
SzA57. A Kaby Lake G-sorozatú processzorok
(felépítésük, MCM megvalósítás, HBM2 memória, az EMIB technika)

- 8. generációs mobil Kaby Lake G sorozat
- egy Kaby Lake CPU-t, egy AMD grafikus egységet és egy 4 GB HBM2 (High Bandwith Memory) VRAM-ot integrál egy MCM csomagba
- a GPU-t egy x8 PCI 3.0 sávval köti össze a CPU-val, a GPU össze van kapcsolva a HBM2 memóriával is az EMIB-en keresztül

* EMIB
- Embedded Multi-Die Interconnect Bridge -> több heterogén lapkát kapcsol össze költséghatékonyan

- Az AMD Vega GPU jelentősen nagyon teljesítményre képes, mint az Intel integrált HD Graphics 630-ja
- további előny: "less board space"

(Cinebench: corss-platform test rendszer, ami kiértékeli a számítógép teljesítőképességeit)

====================================================================================================================
SzA58. A Coffe Lake S-sorozat 2. generációjának innovációi
(USB Gen. 2 támogatás, integrált kapcsolat (integrated connectivity), továbbfejlesztett Optane támogatás)

* USB standard fejlődése:
- USB 1.0 - átviteli ráta: 1.5 MB/s
- USB 2.0 - átviteli ráta: 60 MB/s
- USB 3.0 - SuperSpeed átviteli rátával 625 MB/s -> az USB 3.0 specifikáció szerint a csatlakozónak kék színűnek kell lennie
- USB 3.1 - két alternatíva: Gen 1 és Gen 2 (utóbbi SuperSpeed+ átviteli rátával 1.25 GB/s)
- USB 3.2 - két új SuperSpeed+ átviteli mód: 1.25 és 2.5 GB/s (USB-C csatlakozóval)

+ A 3.0 SuperSpeed átviteli rátája a SuperSpeed buszon alapul - ennek két új soros, pont-to-point adatvonala van (full duplex átvitelhez) és egy föld vonala (ground line) -> ehhez új csatlakozókra volt szükség

* Integrált kapcsolat támogatás
- részben integrálja a 802.11ac Wi-Fi-t, a BT és RF blokkokat a PHC-ra.

>> MIMO technogia
- A MIMO (többszörös adatátviteli csatorna) technológiát az IEEE 801.11n WiFi fejlesztésénél mutatták be, hogy növelját az adat(átviteli) rátát
- feltételezi, hogy mindkét oldalon (adónál és vevőnél is) több antenna található, és több komm. csatornát támogat, amelyeken kerestül egyidejűleg lehet adatot továbbítani
- a MIMO technológiában az adó feldarabolja a küldendő adatot és ezeket párhuzamosan küldi el a vevőnek az elérhető csatornákon, ahol összerakja az adatrészleteket
- 801.11n 4, a 801.11ac pedig 8 párhuzamos adatolyamot támogat

>> Integrated Connectivity
- általában van egy Wi-Fi/BT/RF modul, ami a processzortól különböző módon van elhelyezve (pl. M.2 kártyán)
- IntCon-nal az Intel ezeket részben a PCH-ra integrálta
- ebben az implementációban az Intel a Wi-Fi/BT/RF drága funkcionális blokkjait (pl. logic, Multiplier-Accumulator) a PCH-ra helyezi egy Puslar nevű blokkba. A modul további részeit (Physical layer, RF) egy társ (companion) RF (CRF) modulon maradnak (Jefferson Peak-nek is nevezik)
- A CRF modul az M.2 kártyán van implementálva és egy szabadolmaztatott buszon keresztül csatlakozik a PCH-hoz (CNVio interfészen keresztül)

- 2017-es Atom vonalban már bemutatta az intel, a profit növelését szeretnék elérni vele

* továbbfejlesztett Optane támogatás
- a 2. gen gyors SSD Boot lemezt és nagy HD adat lemezt tételez fel (a HD boot drive-val szemben) ezért ebben az esetben az optán memória az adatlemez cashe-ként szolgál (és nem a HD boot drive-nak)
- a core i9+ -> + jelölés: támogatja az optán technológiát (sötétkék logó)

====================================================================================================================
SzA59. A Zen-alapú processzor családok áttekintése
(a processzor lapka szegmentálása, ennek előnyei, hátrányai, a Zen-alapú processzor családok áttekintése)

Tervezési paradigmák a többmagos processzorok szegmentálásához

-> Monolítikus implementáció [az összes mag ugyanazon a lapkán lesz implementálva - max. 28 mag]

-> Több csipes modul (Multi-Chip-Module - MCM) [a magok több lapkán vannak implementálva, csatlakoztatva vannak egymsához (mindenki mindenkivel) és csomagba vannak ágyazva - 4x8 mag]



Moduláris processzor design előnyei/hátrányai:

+ nagy számú magok gyártása gazdaságosabb, érdemes a nagyobb lapkákat kisebb lapkákra szegmentálni [32 magos proc gyártásának költsége 0.59-szeresére csökken.]

+ a memóriacsatornák száma és az I/O lineárisan skálázva lesz a lapkaszámmal
+ különböző piaci szegmensek számára lehet processzorokat tervezni azáltal különböző számú lapka implementálásával

- magas a késleltetés lapkák között, ami rontja a teljesítményt



Alapvető építő blokkok a Zen alapú AMD procikban: Zen mag, 4 magoc CCX (Core CompleX), 8 magos Zeppelin Module (2x CCX) - Ezek alapján a Zen alapú processzorok:

-> Ryzen Mobile (Mobil): Egy CCX, Vega GPU
-> Ryzen (DT): Zeppelin chip

-> ThreadRipper (HED): (1. gen) 2 Zeppelin chips IF-fel összekapcsolva, MCM-ként implementálva; (2. gen) 4 Zeppelin chip

-> Epyc (1S/2S server): 4 Zeppelin chip IF-fel összekapcsolva, MCM-ként implementálva



IF (cache koherens összeköttetés)

- CCX modult lapkával, MCM-ben lapkát lapkával, két socketet a 2S szerverben köt össze

- HyperTransport összeköttetés fejlesztése: alacsony késleltetés, magas sávszélesség, alacsony energiafogy. és lehetőség van le és felskálázni



ZEN magok:
ZEN (14 nm) -> ZEN+ (12 nm) -> ZEN2 (7 nm) -> ZEN3 (7nm+)


Zen mag:

- elsődleges cél az IPC növelése volt a teljesítménynövelés helyett -> ehhez az AMD kifejlesztette a SenseMI technologia csomagot

- előnyei a korábbi Excavator architektúrával szemben: 14 nm (vs 28 nm), 52%-os teljesítménynövelés egyszálas munkamenet során; 3.7-szeres növekedés teljesítmény/Watt terén; SMT támogatás

(Bulldozer-nél még két integer block volt, itt már egybe van építve + külön FP)



IPC növelés

Innovációk:
- neurális hálózattal javított elágazás becslés
- okos prefetch
- nagyon micro-op cache

Fejlesztések:
- szélesepp op kiküldés (hat FX végrehajtás, 4 FP végrehajtás - utasítás kibocsátás?)
- a mikroarchitektúra továbbfejlesztése



Zen+ mag: 12LP (low power) technológia:

- a frissített Ryzen DT vonalat ugyanakkora lapkaméreten valósította meg ugyanannyi tranzisztorral mint az eredeti design -> ennek következményeképpen a fekete szilikon terület nőtt -> a vastagabb szilikon javítja a lapka termikus viselkedését ***
- körülbelül 11%-kal kevesebb energiát fogyaszt a Ryzen2000 mint a Ryzen 1000 ugyanazon az órafrekvencián -> ugyanazzal az energiafogy.sal pedig 15%-kkal jobb telj.

====================================================================================================================

SzA60. A CCX mag-blokk és a Zeppelin modul áttekintése


(A CCX mag-blokk összevetése ARM mag cluster-ével, a Zeppelin modul
felépítése)

* CCX építő blokk
- a 4 bites CCX (CPU-Complex) egy alap epítő blokk a Zen alapú processzorokban
- minden magnak van egy 512 kB-os privát L2 cache-e, a 4 magnak van egy 8 MB-os L3 cache ami 4 szeletre van felosztva
- az L3 cache többnyire exklúkzív, victim cashe-ként működik az L2 cacheknek
- 1.4 milliárd tranzisztor van implementálva a lapkán
- az összeköttetések miatt (minden lapka 9ssze van kötve minden lapkával) mindegyik mag hozzáférhez az összes L3 cache-hez ugyanazzal az átlagos késleltetéssel

összehasonlítás?
* Zeppelin modul
- két CCX block össze van kapcsolva Infinity Fabric által


*********
Ryzen desktop vonal:
- 1. gen. Ryzen desktop vonal (vagy Summit Ridge vonal)
- van bennük egy Zeppelin lapka (másnéven Ryzen lapka): két CCX complex  IF-fel összekapcsolva
- AM4 socket van benne

ThreadRipper: 248, 249, 253, 254, 255, 256, 260
2. gen. ThreadRipper architektúrális felépítése:
- alacsony magszámú modelleknek két aktív lapkájuk van (6 vagy 8 mag/lapka) -> mindkettőnek direkt hozzáférése van a memóriához és az IO-hoz
- magas magszámű modellek 4 aktív lapkával rendelkeznek (6 vagy 8 mag/Lapka) -> két lapkának van direkt hozzáférése a memhez és az io-hoz, a másik kettőnek nincs; ezeknek magasabb számítási erőforrásra van szükségük, memória és Io erőforrásuk viszont megegyezik
(bi-modális erőforrás felhasználás)
	-> az ütemező elsősorban a közvetlenül csatlakozó magokhoz jelöl ki feladatokat
	-> túlmelegedés elkerüléséhoz viszont nem fogja az közvetlenül csatlakoztatott lapkán lévő összes magot betölteni
Precision Boost




7=======================================
DIES
1.) 
2.) 11, 13, 15, 16, 17, 21, 25
29, 32, 33, 34, 37, 38, 39, 41
48, 49, 50, 52, 53, 56, 59
63, 65, 67, 68, 69, 70

INTERCONNS
8

MOBILE BOOM
1.)	2, 7, 8, 10, 16, 22, 23									(7)
2.)	25, 26, 28, 36, 38									(5)
3.)	40, 42, 43, 44, 48, 52, 54, 55, 56, 58, 61, 62, 63, 66, 67, 69, 70, 73, 74		(19)
4.)	79, 81, 86, 88, 92, 94, 100, 110, 114, 115, 116, 117, 118, 119, 121, 127, 128, 131	(18)
5.)	133, 135										(2)

ARM
1.)	5										(1)
2.)	21, 23, 24, 31, 36, 48, 49							(7)
3.)	146, 147, 152, 154								(4)
4.)	158, 163, 166, 167, 168, 170, 172, 181, 186					(9)
6.)	261, 264, 265, 266, 269, 270, 273, 275, 276, 277, 278, 283, 286, 289, 290, 305	(16)

PDF Felolvasó: http://digitalisinnovacio.hu/hasznos-programok-pdf-felolvaso-program-magyarul/

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Research papers:
https://projekter.aau.dk/projekter/files/63218113/report.pdf
https://www.irjet.net/archives/V4/i4/IRJET-V4I4275.pdf
http://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
http://www.scitepress.org/Papers/2018/67188/67188.pdf
https://pdfs.semanticscholar.org/5fd7/eca5432e4e4038dfc2ad425fa5f3e8739f0b.pdf
https://ieeexplore.ieee.org/document/7124941
https://e-archivo.uc3m.es/bitstream/handle/10016/7089/traffic_escalera_IVC_2003_ps.pdf
https://arxiv.org/ftp/arxiv/papers/1707/1707.07411.pdf
https://towardsdatascience.com/what-is-the-best-programming-language-for-machine-learning-a745c156d6b7

Haar Cascade
Code example:
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
How training works:
https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html
Blog post tutorial:
https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html
Another tutorial (check it out):
http://christopher5106.github.io/computer/vision/2015/10/19/create-an-object-detector.html

===================================================================================================
How to do the cascade training:
===================================================================================================
Step 1
- Create a folder structure like this:

/img
  img1.jpg
  img2.jpg
bg.txt

File bg.txt:
img/img1.jpg
img/img2.jpg

==========================================
==========================================
  Introduction
  Diagnosting imaging has become a very important tool in medicine. It provides a way to map the anatomy of a patient which then can be used for diagnosis, treatment planning or medical studies. Medical imaging is used in various modalities such as magnetic resonance imaging (MRI), digital mammography, ultrasound or computed tomography (CT). As technology evolves more and more medical images are being used in every day medicine and reliable methods and algorithms are needed to process the large amount of data aquired from the patients. These algorithms often provide a way to delineate regions of interests (anatomical structures, organs).
  
  Definitions
  "Image segmentation is the process of partitioning an image into distinctive regions (segments) with similar properties such as intensity, texture, brightness, and contrast." [x]
  
  Due to the fact that there are numerous image scans that need to be processed manual segmentation would take a long time and it would often leads to errors and this is why automated methods are needed to perform segmentation.
  
[x] https://ieeexplore.ieee.org/document/7881449

https://pdfs.semanticscholar.org/d909/6dc6d76148633f168c929e6e7961f563ca78.pdf

Moreover, due to the numerous image scans for each patient and similar intensity of the nearby organs, manual segmentation is a tedious task and is prone to errors. Therefore, reliable automated algorithms should be utilized to delineate the region of interest.

------------------------------
1.) Vision For Test Drives (->)
Phase I. (2 HC on board)
Bosch as driver
Bosch as engineer
//measurement requires a lot of adjustments, plus tagging needs to be done
Phase II. (2 HC on board)
AMS driver
Bosch as engineer
//measurement requires a lot of adjustments, plus tagging needs to be done)
Phase III. (2 HC on board)
AMS as driver
BEKO/AMS as engineer
//measurements are becoming more stable, some intervention is required, plus tagging needs to be done)
Phase IV. (1 HC on board)
AMS as driver
No engineer
//measurements are stable, driver can start the equipment and then do ~100km of driving. Tagging is done offline by AI.

2)
Computer Vision (CV) has been rapidly evolved in the last decade. Today very sophisticated APIs are publicly accessible for image recognition, which help the engineer to do these tasks quickly and efficiently. 
For test drives evaulation, knowing the environment is beneficial, that’s why an engineer does tagging during the drives. This activity can be replaced by AI, with adequate accuracy.
As many drive data could be available in the measurements, the auto-tagging function can go pure image recognition. (e.g.: tagging based on road turn radius, rain-sensor signal, environment temperature etc.)

3) Pyramid
AI
Neural NetWorks
CNN - widespread deep-learning technology for image recognition
faster r-cnn
(+: shows better accuracy compared to other solutions, detects smaller features well; -: slow)

4) Object Recognition - Implementations
Matlab (Computer vision toolbox)
+ Easy to use, many tutorials are available for object det.
+ Commonly used within Bosch
- Limited options compared to Python
- Relatively slow
Python
+ Faster computiong
+ Commonly used
+ Good object det API
+ Better parallelization
- Portability and coding is more challanging
Google APi
+ ready to use service
- slow, expensive, confidentality issues

5) Object rec. software overview
Input video files from measurement camera -> main scrip -> frame extractor (provides frame of the video for the image recognition) -> TensorFlow API: object det api provides per frame results -> main scrip -> result file contains frame number and detected objects

6) Configuring the AI
Config file is needed for:
number of objects to be detected (tunnel, gantry, …)
max objects per frame
thresholds
training config options
and more

7) Training the AI
Training is done throurg the convenient  „LabelImg”
100 frames gives adaquate image recognition performance
Training a new object takes approx 1 hour 
On one frame more objects can belabelled at the same time

8) Resulkts so far
The system is capable of recognizing the following objects on a highway with high accuracy:
Truck
Gantry
Bridge
Tagging a 1.5 hour-long video takes about 2 hours
If we use more training data we can improve the accuracy and robustness of the modell

9) Example pictures

10) Example file

11) Further improvements
Make Auto-Tagging an integral part of the eval tool or the post-processing 
Tagging can be extended into situation recognition by applying vehicle date from the vehicle-bus
Training the modell to detect objects that are very similar to our currently detected objects (to avoid false detections)
Applying plausabilization to output file for avoiding false positives
Turning frame number into time data

--------
GNSS bevezető - ESA könyve
https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf
--------
Edge









------------------------------------------------------------------------------------------
Example
Throughout this essay I used an abdominal CT scan [source] as an example to show the results of different segmentation techniques. To test the described algorithms I used OpenCV, an open source library for computer vision that provides optimized implementation for many of the introduced methods.

---
Edge-based - additional info
Edge-based segmentation strategies assume that different objects are separated by edges and segmentation is performed by identifying the grey level gradients (this approach can be used for images with multiple color channels as well with slight modifications).

The general model for edge based segmentation consists of image smoothing, local edge detection by determining the magnitude and direction of the gradient, edge correction (methods for edge-correction include edge relaxation, non-maximum surpression, hysteresis thresholding), and grouping detected edges.

--
Region-based

Region-growing
Region based methods are based on the assumption that the neighbors of a pixel share similar features within a region so these methods usually compare one pixel to its neighbors recursively. There are many different region based techniques such as region-growing or split and merge.
	
	Seed Segmentation
	One of the region-growing method is called seed segmentation. The basic idea is that we start with a single pixel called seed pixel and we keep expanding the region by adding pixels that share similar features. This method consits of the following steps:
	1.) Calculate the histogram of the image.
	2.) Smooth the histogram by applying a Gauss-filter.
	3.) Detect peaks and valleys (local maximum and local minimum values) in the histogram. This step is called peakiness test.
	4.) Create binary images using the threshold value found in valleys.
	5.) Use connected component algorithm for every binary image to find connected regions.
	The recursive version of the connected component algorithm consists of the following steps, assuming we're working on a binary image and the pixel values are either one or zero: 
	1) Start processing the image from the top left corner.
	2) If the pixel value is "1" and it hasn't been labeled yet then assign a new label to it.
	3) Check every neighbor of the pixel and if the value of the neighboring pixel is "1" and it hasn't been labelled yet, assign the same label that was used in step 2.
	4) The algorithm ends if every pixel whose value is "1" are processed.
	
	Split & Merge
	Split and Merge first splits non-homogeneous regions into four equal size regions. After each split a test is needed to determine whether each new region is homogeneous or not. If they're not, they need to be splitted again. This step is repeated until there are only homogeneous regions left. The method is using a quadtree data structure, where total regions are the parent nodes and the splits of the region are the child nodes. The next step is merging neighboring regions that share similar properties. The algorithm stops when we can't find any more regions to merge. Figure X shows an example of this segmentation method.
	
	[http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/medium/segment/split.htm]

Additional sources:
http://www.di.univr.it/documenti/OccorrenzaIns/matdid/matdid125113.pdf
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/region.pdf

A Split & Merge implementation
https://stackoverflow.com/questions/7050164/image-segmentation-split-and-merge-quadtrees/14730467#14730467

---
A Paper
https://ieeexplore.ieee.org/document/7881449
SECTION I.
Introduction

The increase in the use of imaging modalities (such as CT and MR images) for clinical purposes necessitates the use of powerful computers to assist radiological experts for diagnosis and treatment planning. Moreover, due to the numerous image scans for each patient and similar intensity of the nearby organs, manual segmentation is a tedious task and is prone to errors. Therefore, reliable automated algorithms should be utilized to delineate the region of interest.

Image segmentation is the process of partitioning an image into distinctive regions (segments) with similar properties such as intensity, texture, brightness, and contrast [1]–[2][3]. Medical image segmentation on the other hand is a difficult process in nature. Since image data and model prototype (the a priori description of features to be analyzed) are typically complex, usual methods utilized in image processing cannot be directly used in medical image analysis. Moreover, in most medical segmentation application, such as tumor segmentation, regions to be segmented are often non-rigid, vary in size and location, and differ from patient to patient [4]. This makes the segmentation process even more complex and accurate and reproducible segmentation is not straightforward [5].

Also segmentation is a key step of most medical image analysis tasks, such as registration, labeling, and motion tracking. A primary example is the segmentation of the heart, especially the left ventricle (LV), from cardiac images. Segmentation of LV is compulsory for computing diagnostic information such as ventricular volume ratio, heart output, and for wall motion analysis which provides information on wall thickening, etc. [6]. Development and implementation of the related segmentation techniques requires detailed understanding of the underlying problems as well as the nature of the data, and scientific or medical interest.

Clinical acceptance of segmentation methods depends on their computational simplicity and the degree of user supervision [7]. Both semiautomatic and fully automatic segmentation methods are proposed and utilized in medical segmentation. This paper presents an overview of the most applied existing segmentation techniques applied after acquisition of the medical images. The following sections are organized as follows: First, section 2 describes manual and automatic segmentation methods. Section 3 briefly gives an overview of supervised and unsupervised segmentation methods. Then, in section 4, current segmentation methods and their challenges and advantages are discussed. Next, evaluation of segmentation methods is presented in section 5. Finally, the paper is concluded and summarized in section 6.
SECTION II.
Manual and Automated Segmentation

Image segmentation methods can be divided into three groups based on the degree of human involvement: manual segmentation, semi-automatic segmentation, and fully automatic segmentation [7], [8]. These groups are explained in the following section.
A. Manual Segmentation

Manual segmentation of an organ is manually painting the organ or drawing the boundaries of the organ which is done by an expert (physician, trained technician, etc.) [7], [9]. Manual segmentation is a tedious and a time consuming task. It also has “inter-rater” (different experts segment the image differently), and “intra-rater” variability (the same person segments the image differently in different times); hence, it is not reproducible [10]. Since manual segmentation serves as the ground truth for other segmentation algorithms (automatic and semi-automatic), it should count for inter and intra-rater variability. Although manual segmentation depends on the rater, it is still widely used in clinical trials where time constraint is not of importance [11], [12].
B. Semi-Automatic/Interactive Segmentation

In Semi automatic segmentation (SAS) methods, having the minimum human interactions is required to initialize the method or correct the segmentation results manually [9]. SAS methods use different strategies to combine human knowledge and computers. There are three different types of work flow which are often implemented as reported by Ramkumar et.al. [9]. One should keep in mind that SAS methods are subjected to inter-rater and intra-rater variability.
C. Fully Automatic Segmentation

In fully automatic methods, the computer does the segmentation without human interaction. These methods incorporate human knowledge as a priori and usually use model-based techniques along with soft computing methods. In automatic segmentation methods, the use of anatomical knowledge such as size, shape, and location is essential for having a robust algorithm. Since humans use high level visual processing and specialized knowledge to perform segmentation tasks, developing a fully automated algorithm with high accuracy is challenging.

Currently, the fully automatic segmentation methods don't have wide acceptance in clinical practice, while they are advantageous in processing a large batch of images mostly for the research community.
SECTION III.
Supervised and Unsupervised Segmentation

Image segmentation methods are classified according to the employed learning method into three categories: supervised, semi-supervised, and unsupervised segmentation. In supervised methods the full training data is labeled but in unsupervised ones the segmentation is done with use of unlabeled data. Semi-supervised methods stand somewhere in between the aforementioned methods. In which, small amount of training data is labeled correctly and the rest is either unlabeled or falsely labeled.

Labeling the data requires an expert's knowledge and also is time consuming. Therefore, it is not feasible to utilize supervised methods for large amount of data. Semi-supervised algorithms provide an alternative solution to supervised methods from the computational cost point of view. They also give more accuracy compared to unsupervised methods [13].

Unsupervised methods usually use clustering procedure and do not depend on labeled training set. K-means and Fuzzy C-means are the two commonly used clustering algorithms [13], [14]. The purpose of clustering is to determine decision boundaries using unlabeled training data [15].
SECTION IV.
Segmentation Methods

Although a wide variety of segmentation methods have been presented, none of them will work perfectly for all imaging applications. Moreover, segmentation techniques are often optimized for specific imaging modality such as MRI or CT. Also, selection of the best suitable segmentation method for an intended application is a difficult task and is dependent to goal of segmentation task. In most cases a combination of different segmentation techniques is necessary. In this paper, we explain common segmentation methods used in recent medical image processing. After describing the procedure of each method, the advantages and drawbacks of that method are discussed. The methods can be classified into four classes: 1) Threshold based methods 2) Region based methods 3) Clustering and classification methods 4) Model-based methods
A. Threshold Based Methods

Thresholding is a simple, fast, and effective segmentation method. In this method, the intensities of objects in the image are compared to one (global threshold) or more thresholds (local thresholds).

When an image scan has a bimodal histogram (most common case when no pathology exists, i.e. normal body tissues), objects can be separated from the background with single threshold, also known as global threshold. However, when the image scan contains abnormality or when we are interested in separating different tissue types simultaneously, local thresholding is utilized, either by using several thresholds or by applying multiple thresholding techniques. The threshold value is obtained from the histogram using the valleys and peaks of the image's histogram.

Since thresholding only deals with the intensity of the pixels, not any other relationship between pixels, such as location, it does not consider the spatial information of images. This leads to sensitivity to noise and intensity inhomogeneities of the image. The mentioned problems cause some falsely brighter regions in the image and fundamental differences in the histogram which leads to more complex segmentation [16]–[17][18].
1) Global Thresholding

In this method the pixel intensities of image are compared to a single threshold value and by setting all the pixels above the threshold to one and below it to zero the image is segmented. This creates a binary image. This can be formulated as in (1).
g(x,y)={10f(x,y)≥Totherwise(1)
View SourceRight-click on figure for MathML and additional features. Where f(x, y) is the pixel intensity at (x, y) location, T is the selected threshold, and g(x, y) is the resulted binary image.

The selection of optimum threshold value is of great importance. One of the primary and useful methods in finding the optimal threshold value automatically is Otsu's thresholding method [18]. This method, first proposed in 1978, is based on discriminant analysis. The method assumes that the image has bimodal histogram [19], [20]. This method uses a grey-level histogram of input image and finds the optimal threshold value such that the overlap between the two class (object and background) is minimized.
2) Local Thresholding

To change the Local (adaptive) thresholding uses local threshold values instead of using a single threshold value for the entire image. In this method, intensity histogram is used to estimate the local threshold values. The estimation of threshold values is generally based on prior knowledge of image. Local threshold values can be estimated by using statistical properties such as mean intensity values used by Shanthi and Kumar [21].

Thresholding is the best choice, when an image contains objects with homogeneous region, or the contrast between the objects and the background is high. The selection of threshold becomes difficult when the image contrast is low or in the presence of noise. In general, threshold-based segmentation methods are used as the first step in a segmentation process, since they are unable to segment most medical images desirably [21]. Boegel et.al.[22] used gradient-based thresholding approach for blood vessel segmentation. In their approach, first, the parameters of a global thresholding algorithm is estimated using an iterative process. Then, a locally adaptive version of the approach is applied using the estimated parameters. The method showed promising results compared to common thresholding methods.
B. Region Based Methods

A group of connected pixels with similar properties form a region which may correspond to an object or part of an object. Region based methods form regions by merging neighborhood pixels. Computation of a region is based on predefined similarity criterion between pixels [23]. Region based methods divide the image into regions where two different regions have no overlap, and the union of the regions makes the original image. In fact, this method is the constrained version of the thresholding method in which the pixels within a region must satisfy the similarity criterion. The region growing segmentation algorithm is one of the most common region based methods used for brain tumor segmentation application.
1) Region Growing

Region growing (RG) is among the simplest interactive region based segmentation methods. At least one seed is required to initialize the algorithm. The seed is located at the structure of interest to be segmented. Neighbors of the seed are compared with the seed by means of similarity criterion one by one, and if any of them satisfies the condition then that neighbor is also added to the region. The process iterates until no more pixel can be added to the region. The similarity criteria can be intensity, or any other features in the image. Also automatic seed finding procedures, such as clustering, may be used to find the initial seeds.

While RG generates connected regions with simplicity, it is sensitive to the seed initialization. Some different methods have been proposed to reduce this sensitivity. Sajadi and Sabzpoushan [24] utilized cellular automata for the initial seed selection with conjunction of traditional RG and applied it for retinal blood vessels extraction. Also RG leaks to nearby tissues when the boundary information is not good enough and the image has poor contrast. RG performs well at homogenous and high contrast regions. Another disadvantage of the RG is that it is sensitive to noise and the segmentation produces disconnected areas in the presence of noise. However, RG is still used for many radiology applications such as lung, bone, and homogenous (isolated) tumor segmentation. It has also been widely used in extracting the potential lesions in mammograms [25].
C. Clustering and Classification Methods

Classification is a segmentation technique, in which training data is used to find patterns in the image. In classification methods, either supervised or unsupervised, classifier is used to cluster pixels in the feature space. Clustering means grouping data into classes such that high intra-class (in the same class) similarity and low inter-class (between different classes) similarity exists. The similarity is determined in terms of appropriate distance measure such as Euclidean distance measure. Each cluster is represented by its centroid or mean and variance. There are many clustering methods proposed in the literature. Basic unsupervised clustering methods are: Fuzzy C-means (FCM), K-means, and Markov Random Fields (MRF). The supervised clustering methods include artificial neural network (ANN), and Bayes methods. In this section FCM, MRF and ANN clustering techniques are presented and analyzed.
1) Fuzzy c-Means (fcm)

This Clustering method is an unsupervised algorithm which divides data into two or more clusters. This algorithm assigns membership to each data point corresponding to each cluster center. This membership is based on the distance between the cluster and the data point. The more near the data is to the cluster's center, the more possible its membership is towards that cluster center. Several FCM applications have been presented for MRI segmentation of different body parts [26], [27]. Since FCM is an iterative method, it is very time consuming. To overcome this problem, some solutions such as Bias corrected FCM (BCFCM) clustering algorithms are proposed. BCFCM segments brain images very quickly and provides good quality, which makes it excellent to support virtual brain endoscopy for brain tumor segmentation [28].
2) Markov Random Fields (MRF)

One of the unsupervised clustering algorithms, which integrate spatial information into clustering procedures, is MRF. In many applications, this reduces both the problem of clusters overlapping and noise effect on clustering results [29]. MRF is capable of handling complex dependencies among data instances, providing a high accuracy on segmentation tasks [30]. In [31] hidden MRFs are used to segment cone beam CT images of tumors.
3) Artificial Neural Networks (ANN)

This algorithm is a supervised clustering method. Mathematical operations are applied to the input nodes (features) of an ANN classifier, and the result is carried out at output nodes. To train ANN the values of parameters (involved in the mathematical operations) must be determined such that the error in the predictions (made by the output nodes) is minimized. ANN approaches are non-parametric techniques, since no parametric distribution (such as Gaussian) is assumed for the data. The use of hidden layers in ANN allows the modeling of non-linear dependencies in the input data. Although ANN training phase is complex, the ability to model non-trivial distributions gathers practical advantages. This is an important property in applications such as heterogeneous tumor segmentations where the simple Gaussian assumption is not appropriate [32], [33].

Many different approaches have been investigated in the classification and clustering algorithms segmentation in both feature and classifier selection. For example Dheeba et.al[35] followed a new classification approach for detection of breast abnormalities in digital mammograms using Particle Swarm Optimized Wavelet Neural Network (PSOWNN). The proposed abnormality detection algorithm is based on extracting Laws Texture Energy Measures from the mammograms and classifying the suspicious regions by applying a PSOWNN pattern classifier. ROC curve is then used to evaluate the performance of the method. Hassanien et.al [35] introduce a hybrid approach that combines the advantages of fuzzy sets, ant-based clustering and multilayer perceptron neural networks (MLPNN) classifier, in conjunction with statistical-based feature extraction technique. The authors applied the method for segmenting the breast cancer lesions from MR images.
D. Deformable Model Methods

The segmentation of 3D image data is a challenging task that has been mainly approached by model-based segmentation techniques as parametric/geometric deformable models. Deformable models have the ability to segment images of anatomic structures by building a connected and continuous model which takes into account a priori knowledge about the location, size, orientation, and shape of these structures.

Deformable models are capable of adjusting with significant variability of biological structures over time and across different individuals [36]. Existing deformable models can be divided into two classes: parametric and geometric. The following sections explain the parametric and geometric deformable models.
1) Parametric Deformable Models

Parametric deformable models (PDM) also known as snakes [37] are curves or surfaces defined within the image domain that move under the influence of internal and external forces. In fact, the PDM contour is a parametric curve (Esnake) defined as the sum of three energy terms shown in (2).

Where Econstraint, is the energy term considering constraint on the curve. Eextrenal, and Emtema1 are the energy terms corresponding to external and internal forces respectively.
Esnake=Einternal+Eexternal+Econstraint(2)
View SourceRight-click on figure for MathML and additional features.

The objective is to define the forces in a clever manner such that the final position of the contour will have a minimum energy. Therefore the problem boils down to an energy minimization problem.

Initialization plays an important role in snakes. In applications where the boundaries of objects are so close to each other, the initial position of the model should be placed close enough to the desired boundary to prevent converging to wrong boundaries.
2) Geometric Deformable Models

In PDMs (snakes, b- snakes, etc.) it is challenging to change the topology of the curve as it evolves. If the shape changes dramatically, curve re-parameterization may also be required. An alternative solution is to use geometric deformable models or Level sets (LS) [38].

LS evolve to fit and track objects of interest by modifying the underlying embedding function instead of modifying curve function as in snakes. These methods improve the initialization of parametric active contours and provide symmetrical placement of the initial contour with respect to the boundaries of interest. Although, in practice it is difficult to achieve good segmentation results because in many applications, objects of interest do not have regular shapes [39].
SECTION V.
Segmentation Evaluation

The validation of segmentation results is an important issue in medical image analysis since it has a direct impact on surgical decisions. There are two ways of evaluating the accuracy of segmentation results: qualitatively and quantitatively.

Qualitative evaluation is done by comparing the results to ground truth and giving some rating schemes.

Dice similarity Coefficient (DSC) is the most commonly used quantitative standard measure in many medical segmentation applications, where the overlap between the ground truth and the segmentation results is calculated. DSC is a number ranging from 0 to 1, with 0 indicating no overlap and 1 indicating complete overlap [40]. The drawback of DSC is that it is unsuitable for comparing segmentation accuracy on objects that differ in size [41]. In fact, DSC is not enough for evaluating the segmentation results [42]. For example, in Fig. 1, both of the segmentation results have similar DSC, but as it is clear, the result of segmentation in Fig. 1(b) is not acceptable due to low sensitivity and specificity. Therefore, other measurements such as sensitivity and specificity are calculated along with DSC to give better evaluation of the segmentation results. Sensitivity measures the overlap, but for a poor segmentation much bigger than the ground truth it can be equal to 1. Specificity is therefore the necessary and complementary part of the sensitivity, but it can be equal to 1 for a very poor segmentation that does not detect the object of interest at all.
Figure 1
Fig. 1

Schematic diagrams of sensitivity and specificity metrics with color-coded condition—test outcome pairs: true positive (tp) (green area), true negative (tn) (white area), false positive (fp) (yellow area), and false negative (fn) (blue area). (a) Sensitivity = 94.69% specificity = 94.19%. (b) Sensitivity = 72.99% specificity = 78.16%. VGT = ground truth segmentation, vtest = performed segmentation (brown area) [42]

DSC is defined in (3), while sensitivity and specificity are defined in (4) and (5). In the equations, VGT is the reference standard segmentation (ground truth), Vseg is the segmentation performed by using any of the methods, |⋅| is the size operator, and ⋂ determines the overlapping area between the VGT and the segmentation. TP indicates true positives (intersection between segmentation and ground truth), FP corresponds to false positives (segmented parts not overlapping the ground truth), FN shows the false negatives (missed parts of the ground truth), and TN stands for true negatives (part of the image beyond the union of segmentation and ground truth).
DSC=2|VGT∩Vseg||VGT|+|Vseg|Sensitivity=TPTP+FNSpecificity=TNTN+FP(3)(4)(5)
View SourceRight-click on figure for MathML and additional features.

There are other complementary evaluation metrics such as Hausdorff distance measures which are used for measuring boundary mismatches in the literature [43].

The evaluation of segmentation algorithms always depends on the specific task. For instance, in the tumor segmentation application the requirements are different when the next procedure is to do surgery compared to the volumetric chemotherapy follow-up assessment.
SECTION VI.
Conclusion

This paper explains a number of current image processing methods which are widely used in medical image segmentation. The appropriate segmentation algorithm should be chosen based on the application of the segmentation. When regions of interest can be distinguished from the background by their intensity, threshold-based or region growing techniques have been employed. On the other hand, when objects can be identified by their shapes, model-based techniques are applied for the segmentation. As recently reported in the literature, several general conclusions can be drawn. First, for the sake of reproducibility, it is important to direct segmentation towards a fully automated method by using human intelligence and prior knowledge about the structure of interest along with the algorithm to improve the segmentation results. Second, the use of some pre- or postprocessing methods has demonstrated improved segmentation results. Third, since to the best of our knowledge there is not a standardized database available for many medical applications, the comparison between different algorithms is difficult. Even if such a unified database exists for some medical challenges (like BRATS database for brain tumor segmentation [44]) only a few of the current algorithms were applied to such databases. Finally, although current segmentation algorithms show promising results, the automatic segmentation methods still need more robust procedures to gain wide acceptance among the clinicians for every day clinic practice.

-----------
Evaluation metrics
For detection use this:
https://pdfs.semanticscholar.org/d051/ba0a904e4b4c45a2af145aa29b8490bbbc5c.pdf
For recognition this might be a good idea to start with:
http://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf


Falso Positive Rate (FPR)
Álpozitív ráta:
Az álpozitív eredmények számának és az álpozitív és valódi negatívok összegények az aránya. Azt méri, hogy a rendszer mennyire jól utasítja el az álpozitívokat.

Detektion Rate (DR)
Detektálási arány:
Valódi pozitív osztva a valódi pozitív és álnegatívok összegével.  A ténylegesen detektálandó objektumokat milyen arányban képes detektálni.

Accuracy
TP + TN / CGT


Precision
Pontosság = a valódi pozitívok osztva a valódi pozitívok és fals pozitívok összegével. A pontosság a detektált objektumok közül a helyesen detekált objektumok aránya.

Recall
Azok közül az objektumok közül, amiket detektálni kellett volna, mennyit tudott detektálni a rendszer. (TP/(TP+FN))



-----------------------
https://projekter.aau.dk/projekter/files/63218113/report.pdf
https://pdfs.semanticscholar.org/0f1e/866c3acb8a10f96b432e86f8a61be5eb6799.pdf


AdaBoost
Az algoritmus elvégzi ezt a műveletet az összes tanuló adaton az összes lehetséges jellemzőre. A végső "erős osztályozó" azoknak az osztályozóknak a kombinációja lesz, amelyek a legjobb eredményt szolgáltatják. A tanulás során minden tanuló adathoz hozzárendel egy súlyt, a nehezen osztályozható képeknél ezt a súlyt megnöveli így ezek nagyobb valószínűséggel kerül a végső erős osztályozóba. Csak annyi jellemzőt vesz figyelembe az erős osztályozó, amennyi elegendő a kívánt eredmény eléréséhez. Azon gyenge osztályozókat, amelyek az esetek legalább 50%-ban helyesen detektálnak, hozzá lehet venni az erős osztályozóhoz.

A teljes detektálás egy kaszkádosított rendszer, amelynek az egyes fokozatai az AdaBoost által tanított erős osztályozók. A későbbi fázisokat csak abban az esetben tanítja be a rendszer, ha a korábbi fázisok mindegyikének megfelelt, ezáltal a későbbi fázisok egyre szigorúbbak. Egy objektumot abban az esetben detektál a rendszer, ha az objektumot tartalmazó részkép minden egyes fázison sikeresen áthaladt. Ha az algoritmus már az első fázisban elveti a részképet, akkor csak az első fázis jellemzőit számolja ki, így csak ritkán fordul első, hogy minden fokozat minden jellemzőjét ki kell számolni. Tanulás során minden fázist csak addig tanít a renszer, amíg a meghatározott teljesítményt el nem éri, ezzel csökkentve a számítási időt.

Detektálás során egy adott részképre vonatkozóan az egyes fázisok eredménye lehet pozitív vagy negatív. Valódi pozitív az eredmény, ha az osztályozó helyesen, míg álpozitív az eredmény, ha tévesen detektált egy objektumot. Álnegatív eredmény abban az esetben következhet be, ha az osztályozó nem képes detektálni egy objektumot, annak ellenére, hogy az szerepel az adott részképen. Ahhoz, hogy a rendszer helyesen működjün, az egyes fázisoknak kevés álnegatív eredményt kell produkálnia, hiszen negatív eredmény esetén az algoritmus megáll, és a későbbi fázisokban már nincs lehetőség ezt korrigálni, míg sok álpozitív eredmény megengedett.

Az OpenCV rendelkezik olyan alkalmazásokkal, amelyek segítségével el lehet végezni a rendszer betanítását, valamit szolgáltat beépített függvényeket a betanított rendszer használatához. 


HASONLÓ RENDSZEREK

http://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
Abstract
—In  this  paper,  we  present  a  computer  vision  based
system  for  fast  robust  Traffic  Sign  Detection  and  Recognition
(TSDR), consisting of three steps. The first step consists on image
enhancement and thresholding using the three components of the
Hue Saturation and Value (HSV) space. Then we refer to distance
to border feature and Random Forests classifier to detect circular,
triangular  and  rectangular  shapes  on  the  segmented  images.
The last step consists on identifying the information included in
the  detected  traffic  signs.  We  compare  four  features  descriptors
which  include  Histogram  of  Oriented  Gradients  (HOG),  Gabor,
Local Binary Pattern (LBP), and Local Self-Similarity (LSS). We
also compare their different combinations. For the classifiers we
have  carried  out  a  comparison  between  Random  Forests  and
Support Vector Machines (SVMs). The best results are given by
the combination HOG with LSS together with the Random Forest
classifier.  The  proposed  method  has  been  tested  on  the  Swedish
Traffic  Signs Data  set and  gives  satisfactory results.

https://www.irjet.net/archives/V4/i4/IRJET-V4I4275.pdf
This  paper  reviews  the  method  for  traffic  sign 
detection  and  recognition.  In  the  section  on  learning
based detection,   we   review   the   Viola   Jones   detector   and   the 
possibility   of   applying   it   to   traffic   sign   detection.   The 
recognition  of  the  detected  traffic  sign  is  handled  by  the 
Histogram  of  Gradient  based  SVM  classifier.  Together  this 
system  is  expected  to  perform  much  better  than  the  other 
systems available. The algorithms when trained with proper 
set  of  images  have  been  noted  to  perform  accurately.  This 
must  hold  true  for  the  traffic  signs  as  well  under  different 
color, lighting, atmospheric conditions.

https://www.irjet.net/archives/V4/i4/IRJET-V4I4275.pdf
The system uses the Viola-Jones algorithm to detect signs, 
which  is  a  very  fast  and  accurate  algorithm  if  trained 
properly. This enables the detection on embedded devices 
possible where low computing power is available. Also the 
system uses the HOG algorithm to extract features to train 
the SVM cascade, which is again very accurate. The features 
extracted  are  then  fed  to  SVM  cascade  instead  of  other 
algorithms like ANN, KNN which are not as accurate as the 
SVM algorithm. Moreover, SVM does not have a K value like 
KNN, which slows it as the value increases.

https://lmb.informatik.uni-freiburg.de/people/bahlmann/data/ba_zh_ra_pe_ko_iv2005
 This paper describes a computer vision based sys-
tem for real-time robust traffic sign detection, tracking, and
recognition. Such a framework is of major interest for driver
assistance in an intelligent automotive cockpit environment. The
proposed approach consists of two components. First, signs are
detected using a set of Haar wavelet features obtained from Ada-
Boost training. Compared to previously published approaches,
our solution offers a generic, joint modeling of color and
shape information without the need of tuning free parameters.
Once detected, objects are efficiently tracked within a temporal
information propagation framework. Second, classification is
performed using Bayesian generative modeling. Making use of the
tracking information, hypotheses are fused over multiple frames.
Experiments show high detection and recognition accuracy and a
frame rate of approximately 10 frames per second on a standard
PC

http://www.cs.sjtu.edu.cn/~shengbin/course/SE/Real-Time%20Detection%20and%20Recognition%20of%20road%20traffic%20signs.pdf
This  paper  proposes  a  novel  system  for  the  auto-
matic  detection  and  recognition  of  traffic  signs.  The  proposed
system  detects  candidate  regions  as  maximally  stable  extremal
regions (MSERs), which offers robustness to variations in lighting
conditions.  Recognition  is  based  on  a  cascade  of  support  vector
machine  (SVM)  classifiers  that  were  trained  using  histogram  of
oriented gradient (HOG) features. The training data are generated
from synthetic template images that are freely available from an
online  database;  thus,  real  footage  road  signs  are  not  required
as training data. The proposed system is accurate at high vehicle
speeds, operates under a range of weather conditions, runs at an
average speed of 20 frames per second, and recognizes all classes
of ideogram-based (nontext) traffic symbols from an online road
sign database. Comprehensive comparative results to illustrate the
performance of the system are presented.


MEGVALÓSÍTÁS
[I - Videófájlok kezelése OpenCV-ben]
Képkockák kinyerése

[II:Haar kaszkád osztályozók betanítása]
Az OpenCV több alkalmazással rendelkezik, amelyek segítségével elvégezhető a betanítás: opencv_createsamples, opencv_annotation, opencv_traincascade, opencv_visualisation. Ezek használata a későbbiekben részletesebben is bemutatásra kerül.
[2:Adatok előkészítése]
A gyenge osztályozók tanításához szükség van egy pozitív és egy negatív mintákból álló halmazra, melyeket a már korábban elkészített felvételekből állítottam össze.
[2.1: Negatív minták]
A negatív minták olyan véletlenszerűen választott képkockák, melyek nem tartalmaznak olyan objektumot, amit detektálni szeretnénk. Ezen képek útvonalait össze kell gyűjteni egy szöveges állományban, ahol minden sor egy kép elérési útvonalát tartalmazza. A negatív képek méretére vonatkozóan csak egyetlen megkötés van: nagyobbnak kell lenniük, mint a tanuló ablakméret, ami a model tanításának egyik fontos paramétere. Az ablakméret határozza meg azt is, hogy a negatív képekből hány almintát generál majd az algoritmus. 
[2.1: Pozitív minták]
A pozitív képek tartalmazzák a detektálandó objektumokatat, ezeket pedig az "opencv_createsamples" alkalmazás segítségével készítjük elő. Az alkalmazás két lehetőséget kínál a pozitív mintákat tartalmazó adathalmaz előállítására:
- A rendelkezésre álló pozitív mintákon néhány egyszerűbb transzformációt végrehajtva további mintákat képes generálni. Ez a megoldási mód megfelelően működhet olyan objektumok esetében, amelyeket mindig ugyanolyan fényviszonyok mellett, ugyanolyan szögből szeretnénk detektálni, viszont ezen feltételek nélkül gyenge eredményeket szolgáltat.
- A képeken kivágja és átméretezi a detektálandó objektumokat és átkonvertálja az OpenCV által kezelhető bináris formátumra. Ezt a megoldást választottam, mert volt elegendő mennyiségű tanuló adatom, ezért a továbbiakban csak ezt a módszert ismertetem.
A pozitív képekből is elő kell állítani egy listát egy szöveges fájlban a negatív képekhez hasonlóan. A különbség az, hogy itt jelölni kell az objektumok pozícióját is a képen az alábbi módon:

Könyvtárstrukt.
/img
  img1.jpg
  img2.jpg
info.dat

img/img1.jpg  1  140 100 45 45
img/img2.jpg  2  100 200 50 50   50 30 25 25

Az útvonalt követő szám a képen található objektumok számát jelöli, melyet az objektumok koordinátái (x, y, szélesség, magasság) követnek. Ha ezek elkészültek, az alkalmazás parancssorból futtatható.

[2.3. Az OpenCV integrált annotációs eszköze]
Az OpenCV rendelkezik egy annotációs eszközzel (opencv_annotation), amely segítségével a pozitív mintákat tároló szöveges fájl könnyen előállítható. Az alkalmazást parancssorból indítva két paramétert vár el, a kimeneti fájl útvonalát és a pozitív képeket tartalmazó mappa útvonalát. Ezt követően grafikus felületen, kézzel jelölhetjük ki az objektumokat, és ezek alapján az alkalmazás automatikusan legenerálja a szükséges szöveges fájlt. 

[3] Kaszkád tanítása
Ha a szükséges adathalmazokat előkészítettük, az opencv_traincascade alkalmazás segítségével tanítjuk be az osztályozókat. Főbb paramétereit az alábbiak szerint állítottam be:
??????????
Miután a tanítás befejeződött, a megadott mappában létrejön egy cascade.xml fájl, ami már használható detektálásra.

[4] OpenCV detektálás folyamata
Ahhoz, hogy a betanított detektort használni tudjuk a programban, be kell tölteni az előző lépésben létrehozott xml kiterjesztésű fájlt. A tényleges detektálást az alábbi függvény valósítja meg:

[III - support vektor machines]
- az alábbi táblák felismerésére vagyunk képesek::
- az adatok elkészítése - miért akkora, stb.
- hog jellemzők kiszámítása
- svm megvalósítása
	- betanítás folyamata
	- ezen belül a folyamatos konverzió vektorok és mátrixok között
	- klasszifikáció




------
SPEC RENDSZTERV

Specifikáció
Egy olyan program megvalósítás a cél, amely automatikusan képes detektálni és felismerni járművek forgalmára vonatkozó tilalmi jelzőtáblákat gépkocsiba felszerelt videófelvételeken. A program inputja az előre elkészített videók. Ha tilalmi jelzőtábla szerepel a felvételen, arról adjon vizuális jelzést a felhasználónak és írja ki a tábla pontos nevét. Ha a felvételen több jelzőtábla szerepel egyszerre, akkor értelemszerűen az összesről kell információt szolgáltatni a felhasználónak. A táblafelismerő rendszer abban az esetben működik helyesen, ha a felvételen a tábla megjelenésétől az eltűnéséig legalább egyszer sikerült helyesen detektálni és felismerni a táblát, miközben téves detektálás és felismerés nem történt. A videófelvételek feldolgozása a felhasználó kérésére megszakítható.

Rendszerterv
	A program implementálása során felhasználom az OpenCV könyvtárait, az OpenCV elsődleges interfésze a C++, ezért ezt a nyelvet választottam. A feladat megoldását az alábbi négy fő fázisra osztottam fel: adatgyűjtés, videófelvételek feldolgozása, objektumdetektálás, és objektumfelismerés.
	Adatgyűjtés során a videófelvételek elkészítéséhez egy menetrögzítő kamerát használtam fel, mely a 4. fejezetben részletesebben is bemutatásra kerül. Az adatgyűjtés fő célja, hogy a felvételeken található tilalmi jelzőtáblákból felépítsünk egy adatbázist, amelyet a későbbi fázisokban tanuló adatként használunk fel. További - még nem használt - felvételek szükségesek a program teszteléséhez is.
	A videófelvételek betöltését és kezelhető formátummá történő konvertálását az OpenCV beépített függvényeivel végezzük. Ebben a fázisban történik a képkockák kinyerése a korábban rögzített felvételekből. Az előfeldolgozás során ezeket körülvágjuk és átméretezzük a gyorsabb feldolgozás érdekében.
	c
	Detektáláshoz használható a Hough-kör transzformáció és a kontúrdetektálás kombinált változata. Ezeknek a tanítást nem igénylő egyszerű képfeldolgozási műveleteknek az alkalmazása azonban rendkívül sok tévesen detektált régiót generál a legtöbb felvételen, valamint a tábláknak közel azonos szögből, hasonló megvilágításban kell látszódnia, ezért csak igen korlátozott körülmények között működnek helyesen. További lehetőség lehet egy ablak végigfuttatása a képen, ahol egy HOG jellemzőkkel betanított, két osztályból álló SVM határozza meg, hogy az adott képrészlet tábla vagy sem. Ha az SVM-eket detektálásra használjuk a tanításra használt adathalmaz aszimterikus, mert negatív mintából jóval többet kell tartalmaznia. Ahhoz, hogy a rendszer invariáns legyen a skálázásra minden képkeretből egy képpiramist kell építeni, majd ezek minden szintjére ki kell számolni a HOG jellemzőket, ami meglehetősen költséges művelet, és valós idejű alkalmazások esetén nehezen használható. Haar-szerű jellemzők alkalmazása esetén egy kaszkád szerkezet lehetővé teszi a gyors és pontos objektumdetektálást. Ahhoz, hogy ezzel dolgozni tudjunk, először be kell tanítani a rendszert a már korábban felvett adatokkal. Az OpenCV rendelkezik alkalmazásokkal, amelyekkel elvégezhető az adatok előkészítése és betanítása, valamint létezik implentáció, amely a betanított rendszerrel elvégzi a detektálást. A tanulást igénylő módszerek alkalmazása esetén tehát a detektálás további két részfeladatra osztható fel: az egyik a modell betanítása a másik pedig a detektálás elvégzése új adatokon. A tanítás során a bemenet a felvételekből kinyert képkockák, a kimenet pedig a betanított osztályozó, ami általában egy xml fájl. A tényleges detektálás esetén a bemenet egy képkocka, amin szeretnénk detektálni a jelzőtáblákat, és az előző lépésben létrehozott xml fájl, a kimenet pedig a detektált objektumok pozícióját leíró koordináták.
	A táblák felismerésére szintént használható egy HOG jellemzőkkel betanított SVM osztályozó. Elérhetőek más jellemzők is, mint a SIFT vagy a SURF, de ezek a jellemzők túl hosszú vektorokat állítanak elő és csak szigorú feltételek mellett működnek megfelelően. Mind a tanítás, mind pedig a betanított rendszer használata beépített OpenCV osztályok és függvények segítségével történik. A tanuló adatokat külön mappastruktúrába kell rendezni, a képfájlokat betöltjük és kiszámítjuk rájuk a HOG vektorokat, az SVM-et pedig ezekkel az adatokkal tanítjuk be. Az osztályozó használata esetén a bemenet a már detektálás során kinyert objektumok pozíciója, amelyet kivágunk és átméretezünk, valamint a betanított SVM adatait tartalmazó xml fájl. A kimenet az adott részképen szereplő táblának a neve. Ha már rendelkezésre állnak az adott képkockán szereplő tilalmi jelzőtáblák nevére és pozíciójára vonatkozó információk, az adott képkockán megjelenítjük az eredményeket.

----
öf.
	Egy fedélzeti kamera segítségével készítettem olyan felvételeket, amelyek tartalmaznak járművek forgalmára vonatkozó tilalmi jelzőtáblákat. Az így előállított adatokból felépítettem a kis méretű adatbázist, amelyeket a későbbiekben az objektumok detektálására és felismerésére alkalmas tanuló algoritmusokhoz használtam fel. 
	A detektáláshoz egy Haar kaszkád osztályozót használtam fel, melynek betanítását az OpenCV által biztosított alkalmazások segítségével végeztem el. A kaszkád struktúrának köszönhetően a rendszer [gyorsan elveti azokat a képrészeket ahol biztosan nem található táblázat így viszonylag gyorsan működik és jól használható valós idejű alkalmazásoknál is]. [A detektálás során az egyik legnagyobb kihívás a hamis pozitívok számának a lecsökkentése volt.] A sikeresen detektált régiók felismeréséhez egy tartóvektor-gép osztályozó segítségével valósítottam meg, melyet a táblák képeiből számolt HOG vektorokkal tanítottam be. A modell összesen 12 különböző jelzőtábla felismerésére használható. Tapasztalataim alapján pusztán kéfeldolgozási műveletek alkalmazása nem eredményezett elfogadható megoldást, a változó fényviszonyok és egyéb problémák miatt.
	
	Az összeépített rendszer nagy pontossággal képes a gépkocsiból felvett videókon detektálni és felismerni a tilalmi jelzőtáblákat [mert...]
----	
teszt.
	Detektálás tesztelése
	Felismerés tesztelése
	Teljes rendszer tesztelése
		itt miket nézzünk?
		
-----
További ötletek fotók beszúrására:
- helyes detektálás
- téves detektálás
- missed detektálás
- helyes felismerés
- téves felismerés
- missed felismerés


----
KF - Adaboost Training
> Egyszerű Haar szerű jellemzők (fekete terület - fehér tehér terület) -> gyors számítás
> Integrál kép: ii(x,y)=szumma(i(x', y')
	- s(x,y) = s(x,y-1) + i(x,y) (oszlopösszes)
	- ii(x,y) = ii(x-1,y) + s(x,y)
> ADABoost training
	- tréning fázis: pozitív és negatív minták
	- a legkisebb hiba mellett, a hozzá tartozó szakértőt választjuk
	- súlyozás: szakértők helyett hatékonyabb a súlyozás, minden tréning mintához súlyokat rendel (minél nehezebb korrektül 			osztályozni annál nagyobb)
	- ha nem megfelelően lett egy minta osztályozva, akkor a súlyát növeljük meg minden szakértő hibájával arányosan
		ha valamely minta sok lépés után sem került osztályozásra, akkor a súlya nagyon megnő
	- boosting: 1) kezdetben minden tréning minta azonosan van súlyozva; 2) minden boosting futam során keressók meg azt a gyenge osztályozót amely a legkisebb súlyozott tréning hibát szolgáltatja, az aktuális gynege osztályozó által tévesen osztályozott minták súlyát növeljük; 3) a végső osztályozó a gyenge osztályozók lineáris kombinációja 4) a gyenge osztályozók átsúlyozása és kombinálása az adott AdaBoost módszertől függ
	
	Viola-Jones Face Detector
	- kaszkádba szervezés oka: kevésbé pontos, de gyors osztályozó elemet használjunk először a negatív válaasz azonosítására
	
	
	640x320
teljes vektorhossz	 13851        képszélesség/hogszize - 1 szorozva képmagasság/hosize - 1
sor hossza		1539		képszélesség / hogwinsize - 1 * 81(hogsize)

x = i % sorhossz / winSlide
y = i / sorhossz * winSlide
------

instead of using existing feature vectors we could mayhaps use convolutional neural network to determine the feature vectors and train the svm with that.
https://www.mathworks.com/help/vision/examples/image-category-classification-using-deep-learning.html

------
AMD ZEN Family
I.
------------------
Tervezési paradigmák a többmagos processzorok szegmentálásához
-> Monolítikus implementáció [az összes mag ugyanazon a lapkán lesz implementálva - max. 28 mag]
-> Több csipes modul (Multi-Chip-Module - MCM) [a magok több lapkán vannak implementálva, csatlakoztatva vannak egymsához (mindenki mindenkivel) és csomagba vannak ágyazva - 4x8 mag]

Moduláris processzor design előnyei/hátrányai:
+ nagy számú magok gyártása gazdaságosabb, érdemes a nagyobb lapkákat kisebb lapkákra szegmentálni [32 magos proc gyártásának költsége 0.59-szeresére csökken.]
+ a memóriacsatornák száma és az I/O lineárisan skálázva lesz a lapkaszámmal
+ különböző piaci szegmensek számára lehet processzorokat tervezni azáltal különböző számú lapka implementálásával
- magas a késleltetés lapkák között, ami rontja a teljesítményt

Alapvető építő blokkok a Zen alapú AMD procikban: Zen mag, 4 magoc CCX (Core CompleX), 8 magos Zeppelin Module (2x CCX) - Ezek alapján a Zen alapú processzorok:
-> Ryzen Mobile (Mobil): Egy CCX, Vega GPU
-> Ryzen (DT): Zeppelin chip
-> ThreadRipper (HED): (1. gen) 2 Zeppelin chips IF-fel összekapcsolva, MCM-ként implementálva; (2. gen) 4 Zeppelin chip
-> Epyc (1S/2S server): 4 Zeppelin chip IF-fel összekapcsolva, MCM-ként implementálva

IF (cache koherens összeköttetés)
- CCX modult lapkával, MCM-ben lapkát lapkával, két socketet a 2S szerverben köt össze
- HyperTransport összeköttetés fejlesztése: alacsony késleltetés, magas sávszélesség, alacsony energiafogy. és lehetőség van le és felskálázni

ZEN magok:
ZEN (14 nm) -> ZEN+ (12 nm) -> ZEN2 (7 nm) -> ZEN3 (7nm+)

Zen mag:
- elsődleges cél az IPC növelése volt a teljesítménynövelés helyett -> ehhez az AMD kifejlesztette a SenseMI technologia csomagot
- előnyei a korábbi Excavator architektúrával szemben: 14 nm (vs 28 nm), 52%-os teljesítménynövelés egyszálas munkamenet során; 3.7-szeres növekedés teljesítmény/Watt terén;SMT támogatás
(Bulldozer-nél még két integer block volt, itt már egybe van építve + külön FP)

IPC növelés
Innovációk:
- neurális hálózattal javított elágazás becslés
- okos prefetch
- nagyon micro-op cache
Fejlesztések:
- szélesepp op kiküldés (hat FX végrehajtás, 4 FP végrehajtás - utasítás kibocsátás?)
- a mikroarchitektúra továbbfejlesztése

Zen+ mag: 12LP (low power) technológia:
- a frissített Ryzen DT vonalat ugyanakkora lapkaméreten valósította meg ugyanannyi tranzisztorral mint az eredeti design -> ennek következményeképpen a fekete szilikon terület nőtt -> a vastagabb szilikon javítja a lapka termikus viselkedését ***
- körülbelül 11%-kal kevesebb energiát fogyaszt a Ryzen2000 mint a Ryzen 1000 ugyanazon az órafrekvencián -> ugyanazzal az energiafogy.sal pedig 15%-kkal jobb telj.

----
SzA54. A Skylake család I.

(főbb jellemzői, a mikroarchitektúra továbbfejlesztése, a mikroarchitektúra
szélességének növelése, az Intel grafikai családjai, a fejlődés áttekintése)

SzA55. A Skylake család II.

(bevezetett innovációk: az ISP, Memory Side Cache, a Speed Shift technológia
mint az Enhanced SpeedStep technológia továbbfejlesztése)

SzA56. A Kaby Lake család

(bevezetett innovációk: a Speed Shift technológia 2. generációja, az Optane
memória és az M.2 interfész szabvány)
SzA57. A Kaby Lake G-sorozatú processzorok

(felépítésük, MCM megvalósítás, HBM2 memória, az EMIB technika)

SzA58. A Coffe Lake S-sorozat 2. generációjának innovációi

(USB Gen. 2 támogatás, integrált kapcsolat (integrated connectivity),
továbbfejlesztett Optane támogatás)

VII. AMD Zen-alapú processzor családjai
SzA59. A Zen-alapú processzor családok áttekintése

(a processzor lapka szegmentálása, ennek előnyei, hátrányai, a Zen-alapú
processzor családok áttekintése)

SzA60. A CCX mag-blokk és a Zeppelin modul áttekintése

(A CCX mag-blokk összevetése ARM mag cluster-ével, a Zeppelin modul
felépítése)














==================================================================================================================
A robotmanipulátorok modellezésének matematikai alapjai
- megvan -

Homogén transzformációk
- ha adott egy merev test geometriai reprezentációja és ehhez hozzárendelünk egy koordinátarendszert, akkor a test bármely helyzetében kiszámolható a koordináta rendszer pozíciójának és orientációjának a változása -> ennek legtömörebb formája a homogén transzformációk alkalmazása
- rögzítünk egy pontot a B bázis koordinátarendszerben, ennek egységvektorai pedig legyenek x, y, és z. A v helyvektor pedig leírja az L lokális koordináta rendszer OL origóját a bázis origójához képest
[2.4 és 2.5]
- a második derékszögű koordinátarendszer az eredeti kr.-hez viszonyított orientációját egy R mátrixszal határozzuk meg (ha lokális koordinátarendszer bázisvektorait a bázis koordináta rendszer egyes tengelyeire levetítjük)
[2.6]
+ R ortonormált (e1, e2, e3 oszlopvektorok egységnyi hosszúak) -> inverze megegyezik a transzponáltjával
+ R orientációs mátrix
˘[2.7]

Homogén koordináta-transzformációk
- Feladat: adott egy P pont homogén koordinátái lokális koordinátarendszerben
- kérdés: milyen homogén koordináták határozzák meg a pontot a bázis koordinátarendszerben?
- ezek alapján: 2.9 -> 2.10 ez pedsig összevonva tömören, P pont bázis rendszerben meghatározott homogén koordinátás leírása: 2.11
- [2.12] -> eredmény röviden, ahol [2.13] homogén transzformációs mátrix legáltalánosabb formája: R orientációs mátrix, v oszlopvektor a lokális kr. koordináta helyzetét írja le a bázis kr.-hez képest, perspT a majdnem mindig zérüs perspektív vektor, n skálázó faktor, ami általában egység

- elemi forgatások homogén transzformációs mátrixai: [2.14], eltolás mátrix: [2.5]

Összetett homogén transzformációk („aktív” és „passzív” szempont)
Az elemi forgatások és eltolások sorozatát elemi homogén transzformá¬ciós mátrixok összeszorzásával számolhatjuk ki. Mivel a mátrixszorzás nem kommutatív művelet, ezért nem mindegy, hogy az egyes forgatásokat és eltolá¬sokat milyen sorrendben hajtjuk végre.
- Inicializáljuk a T transzformációs mátrixot az egységmátrixszal, amely azt fejezi ki, hogy kezdetben a B bázis és az L lokális ortonormált koordináta-rendszerek egybeesnek
- Ha az L lokális koordináta-rendszert a B bázis valamely egységvektora körül vagy mentén transzformáljuk (aktív szempont vagy mozgás leírás), akkor a T homogén transzformációs mátrixot balról szorozzuk az aktuális elemi forgatási vagy eltolási homogén transzformációs mátrixszal
- Ha az L lokális koordináta-rendszert valamely saját egységvektora körül vagy mentén transzformáljuk (passzív szempont vagy vonatkoztatási rendszer megváltoztatása), akkor a T homogén transzformációs mátrixot jobbról szorozzuk az aktuális elemi forgatási vagy eltolási homogén transz¬formációs mátrixszal
[ha van még hely: 2.16 + 2.17]

Rodrigues transzformáció
- Ha egy B bázishoz képest az L lokális rendszert tetszőlegesen elforgatjuk, akkor ez a transzformáció leírható egy megfelelően választott u egységvektor körül (gamma) szöggel végzett forgatással. Ezt az operációt Rodrigues-transzformációnak szokás nevezni. 
- Legyen B és L két ortonormált koordináta-rendszer és kezdetben essenek egybe. Legyen u egységvektor és tételezzük fel, hogy az L lokális koordináta-rendszert megforgatjuk az u körül egy ismert gamma szöggel. Ekkor a Rodrigues-transzformáció mátrixa, Rodr(u, gamma), amely az L koordinátáit B koordinátáiba képezi le:
- [2.18]
- [2.19]
- [2.28] - [5 transzformációból áll: 2-vel eljutunk az x tengelyig, ott forgatunk, majd 2-vel visszaforgatunk.]
- rf forgatási szöge és az u vektor: 2.30 és 2.31

==================================================================================================================
3Mit tud a grafikus robotszimulációról? Három dimenziós objektumok megjelenítése (3D -> 2D transzformációk). Hogyan írjuk le a munkadarabokat homogén koordinátákkal? Milyen transzformációkat használhatunk a munkadarabok mozgásának leírására? Külső koordináták: Henger, gömb és derékszögű koordinátarendszerek kapcsolata. 

Három dimenziós objektumok megjelenítése
- Képies, perspektivikus, az emberi látáshoz nagymértékben illeszkedő ábrázolásmód. Az ilyen ábra igen szemléletes, de torzításai jelentősek
- A tárgy tényleges méreteiből, arányaiból lehetőleg sokat megtartó ábrázolásmódok. Ide sorolhatók a párhuzamos vetítéssel nyerhető ábrák, az axonometria.

- Vetítés: Vetítésnek nevezzük azokat a dimenzióveszteséggel járó ponttranszformációkat, melyeknél bármelyik képpont és a neki megfelelő összes tárgypont egy egyenesen helyezkedik el
- Az összetartozó tárgy- és képpontokon áthaladó egyenest vetítősugárnak nevezzük. A vetítés eredménye, a vetület, egy térbeli felületen a képfelületen - képződik. Az egyes tárgypontok képe a vetítősugarak döféspontja a képfelületen 

Az a szabály, amely szerint vetítősugarainkat kiválasztjuk, alapvetően befolyásolja a kialakuló kép jellegét. Ennek megfelelően sorolhatjuk csoportokba a következő néhány vetítésfajtát: 
[13. oldal ábra)

- Középpontos vetítés (centrális projekció)
	- a vetítősugarak mindegyike áthalad egy vetítési középponton
	- a létrejövő kép igen közel áll az emberi szem, a fényképezőgép által alkototthoz
	- perspektivikus ábrához jutunk -> a perspektivikus hatás elsősorban a tárgy és a centrumpont távolságától függ. Ha ez a távolság minden határon túl nő, a középpontos vetítés párhuzamos vetítésbe megy át

- Párhuzamos vetítés
	- vetítősugarak egymással párhuzamosak
	- Ha ezen kívül a vetítősugarak még merőlegesek is a képsíkra, a merőleges - ortogonális - vetítés, egyébként a ferde (klinogonális) vetítés elnevezést használjuk

- Axonometria
	- a párhuzamos vetítés segítségével származtattuk
	- a vetületképzés tényleges végrehajtása geometriai megfogalmazás helyett koordináta-transzformációs felfogásban célszerűbb

Hogyan írjuk le a munkadarabokat homogén koordinátákkal? 
- Az objektumok felületét síklapokkal közelítjük.
- Az így kapott csúcsokat valamilyen sorrendben indexszel azonosítjuk, majd meghatározzuk a csúcspontok homogén koordinátás leírásait.
- Ha a modellezés során n számú csúcsot használunk, akkor a leíró vektorokat 4xn méretű mátrixba foglalhatjuk össze. 
[3.4]

- Ha a céltárgyat egy adott pozícióból egy másik helyzetbe kell eljuttatni, akkor a manipuláció leírására homogén transzformációs mátrixot használunk [3.5]

Milyen transzformációkat használhatunk a munkadarabok mozgásának leírására?
LD előző tételek

Küldő koordináták


==================================================================================================================
A Denavit-Hartenberg passzív szemléletű modell. Milyen paraméterek írják le a robotok kinematikai mozgását? Hogyan írjuk le a szomszédos koordinátarendszerek kapcsolatát Denavit-Hartenberg konvenció szerint! Ismertesse a forgó és csúszó típusú izületek Denavit-Hartenberg modelljét! 

1) Denavit-Hartenberg modell
A manipulátorok kinematikai modellezésére alkalmazott, széles körben elter¬jedt egyik módszer a Denavit–Hartenberg-leíráson alapszik

2) Kartag és izületi kinematikai paraméterek
Tételezzünk fel egy egyszerű, nyílt kinematikai láncot n darab izülettel.
Kartagra jellemző két paraméter:
- kartag általánosított hossza - a (az a távolság, amely a két egymást követő izület tengelyeinek a közös normálisa - az az egyenes, amely mindkét egyenesre merőleges)
- kartag csavarási szöge - alfa (a két kitérő tengely hajlásszöge - úgy kaphatjuk meg, ha az egyik izületi tengelyt önmagával párhuzamosan a közös normális mentés a másik izületi tengelyig toljuk)
[3.1. ábra]

Izületeket további két paraméterrel jellemezhetjük:
- izületi távolság - b (a két közös normálisnak az adott izületi tengelyen mért távolsága)
- izületi szög - béta (a két kapcsolódó kartag közös normálisának az egymással bezárt szöge, az izületi tengelyre merőleges síkban mérjük)

-> azu, hogy melyik paraméter változik a mozgás során, az izület típusa határozza meg. ha a kartag elcsavarodhat, vagy billenhet az izületi tengely körül, akkor béta a mozgásra jellemző változó. ha a kartag a tengely mentén egyenes vonaló mozgást végez, akkor a b izületi távolság a változó paraméter
-> a kartagparaméterek értékei konstansok

- A kartag és az izületi paramétereket együttesen Denavit–Hartenberg kinemati¬kai paramétereknek nevezzük. -> Egy n számú mozgástengelyt tartalmazó mani¬pulátor esetében legalább 4n kinematikai paraméter szükséges annak biztosítá¬sára, hogy a manipulátor kinematikai konfigurációja meghatározott legyen.
- ebből három olyan konstans érték, melyet a mechanikai tervezés határoz meg, míg a negyedik érték a mozgás típusától függő változó.

3) Koordináta-rendszer-hozzárendelés a Denavit–Harten¬berg-konvenció szerint
- a kinematikai lánc mozgásának meghatározásához mindegyik izülethez hozzárendelünk egy lokális koordinátarendszert

Koordinátarendszerek meghatározásának az algoritmusa:
1: Az i-edik Li lokális koordináta rendszer az i-edik kartag végén helyezkedik el
2: A zi koordinátatengely mindig a mozgástengely irányába mutat
3: Meghatározzuk a zi-vel azonosított mozgástengelyek közös normálisát, ekkor az xi lokális egységvektor: xi = zi-1 x zi
4: Az Li koordinátarendszer origója a mozgástengely és az i-edik kartag közös normálisának metszéspontja
5: yi = zi x xi (mert jobbsodrású rendszer)
 [3.2 ábra]

- A Denavit–Hartenberg-konvenciókat követve, a manipulátor izületeihez rögzített koordináta-rendszerek közötti kapcsolatot négy elemi transzformáció so¬rozatával biztosíthatjuk.
- Passzív szemléletű módszer szerint minden egyes lépésben az éppen aktuális lokális koordináta-rendszer valamely tengelye mentén vagy akörül hajtunk végre transzformációt.

Az egymást követő elemi transzformációk rendre a következők:
1. Az (i-1)-edik koordináta-rendszert forgassuk el a zi-1 tengely körül az teta)i izületi szöggel. Ennek a transzformációnak az lesz az eredménye, hogy az így transzformált xi-1 tengely a rákövetkező xi tengellyel párhuzamos lesz.
2. Az első lépésben kapott lokális koordináta-rendszer z tengelye mentén toljuk el a lokális koordináta-rendszert az izületi távolsággal. A második transzformá¬ció azt eredményezi, hogy az így kapott lokális koordináta-rendszer xi-1 tenge¬lye és az i-edik koordináta-rendszer xi tengelye egy irányba mutat.
3. Ha a második lépésben kapott lokális koordináta-rendszert az xi-1 egység¬vektora irányában a kartag általánosított hosszával eltoljuk, akkor a transzfor¬máció eredményeként kapott lokális koordináta-rendszer origója és xi-1 tenge¬lye, valamint az i-edik koordináta-rendszer Oi origója és xi tengelye egybe fog esni.
4. Végül forgassuk el a kapott lokális koordináta-rendszert a kartag csavarási szögével az xi-1 egységvektora körül. E transzformáció eredményeként a kapott lokális koordináta-rendszer és az i-edik koordináta-rendszer egybe fog esni.


Írja le a pi, és pi-1 ugyanazt a P pontot az i-edik és (i-1)-edik lokális koordináta-rendszerben:
Ekkor a két vektort a következők szerint feleltethetjük meg egymásnak:
(3.4)
(3.9)
(3.10)
