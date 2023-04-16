# SSN Lab. 5 RBF

Zapoznaj się z zawartością notatnika Jupyter umieszczonego w repozytorium  i wykonaj zawarte w nim ćwiczenia.

Notatnik: [05_rbf.ipynb](https://github.com/IS-UMK/ssn_23_lab_05/blob/master/05_rbf.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IS-UMK/ssn_23_lab_05/blob/master/05_rbf.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/IS-UMK/ssn_23_lab_05/master?filepath=05_rbf.ipynb)

---

## Zad. 5. Klasyfikacja cyfr za pomocą sieci RBF

Zaimplementuj sieć RBF do zadań klasyfikacji posiadającą następujący algorytm uczenia: 
 
1. Ustalenie centrów funkcji radialnych (metoda ``init_centers(X,y)``) za pomocą algorytmu k-średnich.   
W tym celu można skorzystać z algorytmu [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).

2. Ustalenie rozmyć funkcji radialnych (metoda ``init_sigmas(X, y)``).  
Wartość rozmycia $\sigma_k$ dla każdej funkcji radialnej jest dobierana jako średnia odległość między wektorami treningowymi związanymi z centrum $k$

$$
\sigma_k=\frac{1}{m} \sum_{i=1}^m\left\||\mathbf{x}_i-\boldsymbol{\mu}_k\right\||
$$

Wskazówka: 1) metoda ``predict(X)`` klasy ``KMeans`` zwraca wektor określający przynależność do centrów. 2) funkcja [scipy.spatial.distance.pdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html) oblicza odległości między wierszami macierzy.

3. Uczenie wag warstwy wyjściowej (metoda ``update_weights(X, y)``).  
Wagi i wyrazy wolne ustalone na podstawie rozwiązania układu równań $\mathbf{W}\mathbf{Z} + \mathbf{b}= \mathbf{y}$.

$$ \mathbf{W}' = \left( \mathbf{Z}'^T \mathbf{Z'}\right)^{-1}\mathbf{Z'}\mathbf{y} $$

Zastosuj uzyskany algorytm RBF do klasyfikacji zbioru danych ``digits`` (zob. [sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)).

```python
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
```

Znajdź najskuteczniejszy model klasyfikacji dobierając odpowiednią ilość neuronów ukrytych. Ocenę modeli wykonaj z użyciem  walidacji krzyżowej (np. za pomocą przeszukiwania siatką ``GridSearchCV``). 

Rozwiązanie w postaci notatnika Jupyter (``.ipynb``) lub skrypt w języku Python (``.py``) umieść w Moodle lub prześlij do repozytorium GitHub.

---
## Materiały:

* Chris McCormick, "Radial Basis Function Network (RBFN) Tutorial"  
  http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/  
  MATLAB code: https://chrisjmccormick.wordpress.com/2013/08/16/rbf-network-matlab-code/
* Chris McCormick, "RBFN Tutorial Part II - Function Approximation"  
  http://mccormickml.com/2015/08/26/rbfn-tutorial-part-ii-function-approximation/  
