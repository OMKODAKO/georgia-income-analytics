# Stage 00 — Research & Feasibility Report
**პროექტი:** Georgia Income Analytics — Gender & Income Study
**თარიღი:** 2026-11-27 · ყველა რიცხვი გამოთვლილია ჩამოტვირთული ფაილებიდან (`src/feasibility_lfs.py`, `src/inspect_raw.py`)

## 1. დასკვნა მოკლედ
**დიახ — ინდივიდუალური შემოსავლის სქესის მიხედვით ანალიზი შესაძლებელია**, მაგრამ მხოლოდ
**დაქირავებულთა ხელფასისთვის (net, main job)** და მხოლოდ **ინტერვალების (bands)** სახით.

| წყარო | ინდივიდუალური შემოსავალი? | გამოყენება |
|---|---|---|
| LFS 2017–2025 (microdata) | ✅ net earnings ინტერვალებად, მხოლოდ დაქირავებულები | **მთავარი Dataset** |
| HIES 2017–2025 (microdata) | ❌ შემოსავალი მხოლოდ ოჯახის დონეზე | დამხმარე (ოჯახის შემოსავალი × ოჯახის უფროსის სქესი) |
| Geostat Wages (aggregates) | — | ოფიციალური ბენჩმარკი (GPG, მედიანა, CPI) |

## 2. წყაროები (ყველა ჩამოტვირთულია, SHA-256 → `data/raw/manifest.json`)
- LFS microdata: https://www.geostat.ge/en/modules/categories/130/labour-force-survey-databases (SPSS .sav, 2017–2025, ~3 MB/წელი)
- HIES microdata: https://www.geostat.ge/en/modules/categories/128/ (SPSS .sav, 2009–2025; ჩამოტვირთულია 2017–2025)
- Wages: https://www.geostat.ge/en/modules/categories/39/wages (GPG_Eng.xlsx, earnings by sex, median, ISCO, regions)
- CPI 2010=100: https://www.geostat.ge/en/modules/categories/26/
- Codebooks: LFS variables (2 docx), HIES/IHS variables (pdf), კითხვარები (zip-ებში)

**გამოყენების პირობები / კონფიდენციალურობა:** ფაილები Geostat-ის მიერ საჯაროდ, ანონიმიზებული სახითაა
გამოქვეყნებული (არ შეიცავს სახელებს/მისამართებს; ID-ები მხოლოდ კვარტლის შიგნით უნიკალურია).
საჭიროა წყაროს მითითება (Geostat). ზუსტი თანხები საჯარო ფაილში ინტერვალებადაა გადაყვანილი — სავარაუდოდ
ანონიმიზაციის მიზნით. *Terms of Use* გვერდი (geostat.ge → Services → Terms of Use) Stage 01-ში ხელით უნდა გადაიხედოს.

## 3. LFS — ხელმისაწვდომი ცვლადები (შემოწმებულია ყველა წელზე)
| სასურველი ცვლადი | 2017–2019 | 2020–2025 | შენიშვნა |
|---|---|---|---|
| Gender | `Sex` | `Sex` | 1=ქალი, 2=კაცი |
| Age | `Age` | `Age` | ზუსტი ასაკი |
| Region | `Region` (LFS_ID) | `Region` | 10–11 რეგიონი |
| Urban/Rural | `Urban_Rural` | `Urban_Rural` | |
| Education | `I1_Education` | `Education` | 13 დონე, თანხვედრი |
| Occupation | `B5_Occupation`, `Occupation_converted` | `Occupation`, `Occupation_converted` | ISCO-08 4-ნიშნა / 1-ნიშნა; კოდი 0 ლეიბლი შეიცვალა |
| Industry | `Brunch_2` (NACE 2) | `Brunch` / `Brunch_converted` | |
| Employment_Status | `B7_Status` (ICSE-93) | `Status` (ახალი სქემა) | „Employee"=1 ორივეში |
| Working_Hours | `C2_M_Usually_hours` | `M_Usually_hours` | ჩვეულებრივი კვირის საათები |
| Monthly_Income | `B17_B18_Net_earnings` (9 band) | `B24_B25_Net_earnings` (12 band) | **ინტერვალი, არა თანხა** |
| Household_Size | `Family_Size` (ცალკე ცხრილი) | `Family_Size` | |
| Informal, sector, contract | ✅ | ✅ | დამატებითი კონტროლები |
| Work_Experience | ❌ | ❌ (მხოლოდ „წინა 8 წელს მუშაობდა?") | proxy: potential experience = ასაკი − სწავლის წლები |
| Survey weight | `P_Weights` | `P_Weights` | წლიური შეფასება = weight / 4 |
| Strata / PSU | ❌ | ❌ | ზუსტი design-based SE შეუძლებელია |

## 4. Earnings ცვლადის ხარისხი (დაქირავებულები)
| წელი | n ქალი / კაცი | უარი/გამოტოვება ქ / კ | top band („>2000") ქ / კ | მედიანური band ქ / კ |
|---|---|---|---|---|
| 2017 | 6435 / 6619 | 5.0% / 7.6% | 0.8% / 2.0% | 201–400 / 401–600 |
| 2019 | 7878 / 7986 | 7.7% / 11.8% | 0.8% / 1.8% | 401–600 / 401–600 |
| 2021 | 7520 / 7804 | 15.4% / 16.8% | 1.3% / 2.7% | 401–500 / 601–800 |
| 2023 | 8386 / 8535 | 16.1% / 18.8% | 2.1% / 4.5% | 601–800 / 801–1000 |
| 2025 | 8952 / 9036 | 20.6% / 25.8% | 2.3% / 7.3% | 601–800 / 1001–1500 |
სრული ცხრილი: `reports/stage00_lfs_earnings_coverage.csv`, განაწილება: `stage00_lfs_band_distribution.csv`

**პირველი სიგნალი:** ყველა წელს ქალების მედიანური ინტერვალი კაცებისაზე დაბალია ან თანაბარი (2019) —
ეს ჯერ არ არის სტატისტიკური დასკვნა.

## 5. ოფიციალური ბენჩმარკი (Geostat, GPG_Eng.xlsx, წყარო: LFS)
Adjusted GPG, Total: 2020 — 15.9% საათობრივი / 21.4% თვიური; 2023 — 13.2% / 20.9%; 2024 — 11.4% / 19.3%;
2025 — 13.3% / 20.4%. ჩვენი შედეგები Stage 04-ში ამ მაჩვენებლებთან უნდა შედარდეს.

## 6. შეზღუდვები და რისკები
1. **ინტერვალური შემოსავალი** → საჭიროა interval regression / ordered models; საშუალო მხოლოდ ვარაუდით (midpoint/Pareto top-band).
2. **Band სქემა შეიცვალა 2020-ში** (9→12) → წლების შედარება საერთო საზღვრებზე (100/200/400/600/800/1000/1500/2000).
3. **ნომინალური ზრდა + ღია top band** — 2025-ში კაცების 7.3% „>2000"-შია; CPI დეფლაცია აუცილებელია.
4. **Non-response 5%→26%, კაცებში მეტი** → შესაძლო მიკერძოება; საჭიროა sensitivity/imputation ანალიზი.
5. **2023+ უარი = system-missing** (არა 88); **2017 კოდი „10"** (62 ჩანაწერი) არ არის დოკუმენტირებული → missing.
6. თვითდასაქმებულების შემოსავალი არ არის → შედეგები ეხება მხოლოდ დაქირავებულებს.
7. Panel სტრუქტურა (ოჯახი ≤4 ინტერვიუ) → SE-ები ოჯახზე კლასტერიზებული (`DiaryID` 2020+).
8. ცხრილების/ცვლადების სახელები იცვლება წლებს შორის; 2018-ში `UID` ტექსტად არის → harmonisation map საჭიროა.
9. HIES ინდივიდუალურ შემოსავალს არ იძლევა — **ოჯახის შემოსავალი ინდივიდუალურად არ ჩაითვლება**.

## 7. რეკომენდაცია Stage 01-სთვის
მთავარი Dataset = LFS 2017–2025 დაქირავებულები; HIES — დამხმარე; Wages/CPI — ვალიდაცია და დეფლაცია.
