> **Cel biznesowy:** Automatyzacja oceny zdolności kredytowej przedsiębiorstw z kalibracją prawdopodobieństw przy zachowaniu pełnej interpretowalności decyzji.

##  O Projekcie (Executive Summary)
Celem projektu było zbudowanie modelu scoringowego, który przewiduje ryzyko upadłości firmy na podstawie jej sprawozdań finansowych.

Projekt rozwiązuje kluczowy problem biznesowy: **jak połączyć wysoką skuteczność "czarnej skrzynki" (AI) z wymogami regulacyjnymi (wyjaśnialność decyzji)?**

---

> **Szczegóły techniczne:** Pełen opis metryk i analizy danych znajdziesz w pliku [MODEL_CARD.md](MODEL_CARD.md).


## (How to run)

### Wymagania
* Python 3.8+
* Biblioteki: numpy, scipy, pandas, scikit-learn, lightgbm, shap, lime (pełna lista w `requirements.txt`)

### Instrukcja
1.  **Sklonuj repozytorium:**
    ```bash
    git clone [https://github.com/twoj-nick/credit-scoring-project.git](https://github.com/twoj-nick/credit-scoring-project.git)
    cd credit-scoring-project
    ```
2.  **Zainstaluj zależności:**
    ```bash
    pip install -r requirements.txt
    ```
    Nie wszystkie z nich są konieczne, jeśli ma się większość podstawowych pakietów ML może to nie być konieczne.
3.  **Uruchom Notebooki:**
    * `EDA&Preprocessing.ipynb` – Analiza danych i inżynieria cech.
    * `BlackBox.ipynb` - Model BlackBox z interpretacją.
    * `LogisticRegression.ipynb` – Regresja logistyczna z interpretacją.

---

##  Struktura Plików
* `zbiór_8.csv` - Dane wejściowe (zanonimizowane sprawozdania finansowe).
* `images/` - Wykresy i wizualizacje do raportu.
* `MODEL_CARD.md` - **Pełna dokumentacja techniczna modelu.**
* `Dane_Opis_zmiennych.pdf` - Opis danych.
* `notebooks` - Wszystkie kody i implementacje.

---
**Autorzy:** Piotr Wysocki, Karol Kacprzak


