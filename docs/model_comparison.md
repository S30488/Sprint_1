# Porównanie modeli (Sprint 3)

Wszystkie modele: ten sam split (stratified, test_size 0.2, random_state 42),
2280 rekordów train / 570 test, dane `iranian_churn` (sha256 `90d5fb6b…`).
Metryki jakości liczone dla klasy pozytywnej 1 (klient odchodzi), główna metryka: F1.

Runy W&B: projekt `asi-churn-s3`.

## Jakość

| Model | F1 | Recall | Precision | Accuracy | Wykryci churn (z 89) |
|---|---|---|---|---|---|
| baseline Dummy | 0.000 | 0.000 | 0.000 | 0.844 | 0 |
| baseline LogisticRegression | 0.610 | 0.483 | 0.827 | 0.904 | 43 |
| GradientBoosting | 0.802 | 0.753 | 0.859 | 0.942 | 67 |
| RandomForest | 0.825 | 0.742 | 0.930 | 0.951 | 66 |
| AutoGluon medium_quality (60 s) | 0.864 | 0.854 | 0.874 | 0.958 | 76 |
| AutoGluon best_quality (180 s) | **0.877** | 0.843 | 0.915 | 0.963 | 75 |

## Koszt działania

Pomiar w osobnym procesie, na tym samym komputerze, dla pakietu z `save_model_package`
(`load_model_package` → predykcja). RAM = przyrost pamięci procesu po wczytaniu i pierwszej predykcji.

| Model | Rozmiar pakietu | RAM po wczytaniu | Czas wczytania | Predykcja 1 rekordu | Predykcja 570 rekordów |
|---|---|---|---|---|---|
| baseline LogisticRegression | 0.01 MB | 4 MB | 73 ms | 1.7 ms | 2 ms |
| GradientBoosting | 0.35 MB | 9 MB | 105 ms | 1.8 ms | 3 ms |
| RandomForest | 2.9 MB | 15 MB | 120 ms | 5.2 ms | 9 ms |
| AutoGluon medium_quality | 35 MB | 110 MB | 0.9 s | 1.8 ms | 4 ms |
| AutoGluon best_quality | 113 MB | 217 MB | 1.2 s | 144 ms | 234 ms |

## Wybór: AutoGluon medium_quality (`autogluon_medium`)

- Najlepszy recall (0.854) – wykrywa 76 z 89 odchodzących klientów, o 10 więcej niż RandomForest.
  Dla działu retencji przeoczony klient (FN) to stracony klient, więc recall jest dla nas ważny.
- F1 0.864 – tylko o 0.014 niżej niż best_quality, który wykrywa nawet o 1 klienta mniej.
- best_quality jest ok. 80× wolniejszy na pojedynczym rekordzie (144 ms vs 1.8 ms), 3× większy
  i zajmuje 2× więcej RAM – za minimalny zysk F1.
- Czas predykcji medium_quality jest taki sam jak modeli sklearn (ok. 2 ms na rekord).
- Koszt: pakiet 35 MB i ok. 110 MB RAM oraz zależność `autogluon.tabular` w API.
  To mieści się bez problemu w kontenerze (Cloud Run), a model wczytuje się raz przy starcie API.

Alternatywa, gdyby API musiało być bardzo lekkie: RandomForest (F1 0.825, 2.9 MB, bez AutoGluona).

## Uwagi

- AutoGluon przed zapisem robi `persist()`, więc pakiet (`model.joblib` + `metadata.json`) nie zależy
  od katalogu treningowego i działa po przeniesieniu (np. po pobraniu z W&B).
- W kursowym środowisku AutoGluon trenuje tylko LightGBM, CatBoost i modele sklearn
  (XGBoost i sieci neuronowe nie są instalowane) – ostrzeżenia o pominiętych modelach są oczekiwane.
- `load_peak_ram_kb` w W&B jest mierzone przez `tracemalloc` i zaniża pamięć modeli z kodem natywnym
  (dla AutoGluon 44 MB zamiast ok. 110 MB), dlatego w tabeli wyżej jest pomiar pamięci procesu.
