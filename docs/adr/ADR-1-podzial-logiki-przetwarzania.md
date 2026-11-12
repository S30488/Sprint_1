# ADR-1: Podział logiki przetwarzania danych

- **Status:** przyjęty
- **Sprint:** S2
- **Autor:** Jakub Okurowski
- **Recenzent:** Robert Juchimiuk

## Kontekst

Oddzielamy trening modelu od przygowania danych aby ograniczyć data leakage, powinno zapewnić to powtarzalność

## Rozważone opcje

1. Przetworzenie całości w jednym etapie.
2. rozdzielic pomiedzy: normalizacja, filtrowanie, preprocessing modelu

## Decyzja
Opcja 2, ponieważ ogranicza ryzyko data leakage i ułatwia testowanie na kazdym etapie
data_processing odpwiada za normalizacje i filtrowanie, data_science za preprocesing, trening, ewaluacje

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| Liczba rekordów | 3150 | EDA |
| Liczba duplikatów | 300 | EDA |
| Liczba rekordów po filtrowaniu | 2850 | EDA ||
## Odrzucone alternatywy

Odrzucamy opcje 1 poniewaz utrudnia ona kontrole data leakage i utrudnia testy

## Konsekwencje

Ułatwia testowanie i kontrole nad kodem, zmusze nas jednak do spójnej konfiguracji Kedro przy następnych sprintach

## Historia

| Data | Zmiana |
|---|---|
| 2026-11-12 | Utworzenie ADR-1 |
