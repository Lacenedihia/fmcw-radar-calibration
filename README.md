# Calibration extrinsèque d'un radar FMCW à partir des données brutes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lacenedihia/fmcw-radar-calibration/blob/main/fmcw_calibration.ipynb)

Ce projet estime automatiquement la **position (x, y)** et l'**orientation (lacet)** d'un radar FMCW installé sur site. Il part des **données brutes du capteur** et de quelques réflecteurs de référence dont la position est connue. L'objectif est de rendre l'installation d'un radar de surveillance plus simple et plus fiable : l'algorithme n'a besoin **d'aucune mesure préalable de l'installateur** et donne une **incertitude** sur le résultat, ce qui permet de valider l'installation.

Toute la chaîne est simulée puis évaluée par Monte Carlo : signal IF → traitement radar → association → calibration → évaluation.

![Scène et calibration](figures/scene_calibration.png)

## Résultats principaux

| | Valeur |
|---|---|
| Précision de l'estimateur (cible isolée) | **0,8 mm** en distance, **0,03°** en azimut (RMS) |
| Erreur de position (5 réflecteurs, SNR ≥ 22 dB) | **0,4 à 1 cm** (médiane), < 8 cm (90e percentile) |
| Erreur de lacet (5 réflecteurs, SNR ≥ 22 dB) | **0,04 à 0,1°** (médiane), < 0,3° (90e percentile) |
| Taux de succès (SNR ≥ 22 dB) | **99–100 %**, pour une orientation de départ quelconque |
| Cohérence de l'incertitude prédite | 93–98 % des erreurs réelles tombent dans l'ellipsoïde à 99 % |

![Évaluation Monte Carlo](figures/monte_carlo.png)

**Ce qu'on retient :**
- La précision en lacet est limitée par la précision angulaire du réseau (8 antennes). L'erreur de position est fortement couplée à l'erreur de lacet, par effet de bras de levier.
- Ajouter des réflecteurs améliore surtout la **robustesse** : le taux de succès passe de 39 % avec 3 réflecteurs à 99 % avec 8, à SNR modéré.
- Avec une cible isolée, l'erreur est millimétrique. L'erreur résiduelle vient donc des **interférences entre réflecteurs proches** (lobes secondaires). En pratique, il faut espacer les réflecteurs de plusieurs cellules de résolution, en distance et en azimut.

## Méthode

### 1. Simulation des données brutes FMCW
- Radar 24 GHz, bande 250 MHz (résolution 0,6 m), 64 chirps de 100 µs, 256 échantillons IF complexes, réseau virtuel de 8 antennes espacées de λ/2.
- Le cube IF (antennes × chirps × échantillons) contient les réflecteurs de référence (trièdres d'environ 50 m²), du clutter statique, un piéton en mouvement et du bruit thermique.
- L'amplitude des cibles suit l'équation radar (SNR ∝ σ / r⁴).

### 2. Traitement du signal
- Fenêtrage de Hann, puis FFT distance et FFT Doppler, avec intégration non cohérente sur les antennes.
- Détection par **CA-CFAR 2D** (Pfa = 10⁻⁶) et recherche des maxima locaux.
- **Estimation sub-bin** en distance et en angle par maximisation de la DTFT autour du pic (maximum de vraisemblance d'une sinusoïde). Cette méthode supprime le biais de l'interpolation 3 points, qui allait jusqu'à environ 7 cm.

![Cartes distance-Doppler et distance-azimut](figures/range_doppler_azimuth.png)

### 3. Association sans pose initiale (RANSAC)
Les distances entre réflecteurs ne changent pas quand on tourne ou déplace le radar. Pour chaque paire de détections statiques compatible avec une paire de réflecteurs connus, on calcule une pose candidate et on compte les réflecteurs qui retombent sur une détection. L'hypothèse qui en explique le plus est retenue. Les cibles mobiles sont écartées par leur vitesse Doppler, et le clutter par RANSAC.

### 4. Estimation des paramètres extrinsèques
1. **Solution fermée** : alignement rigide 2D par SVD (algorithme de Kabsch).
2. **Raffinement** : moindres carrés non linéaires pondérés (Levenberg-Marquardt) directement dans l'espace de mesure (distance, azimut).
3. **Incertitude** : la covariance (JᵀJ)⁻¹, mise à l'échelle par la variance a posteriori, donne un critère objectif pour valider l'installation.

### 5. Évaluation des performances
Monte Carlo de 150 essais par configuration. À chaque essai, la scène, la pose réelle (orientation quelconque) et le bruit sont nouveaux, et **toute la chaîne** est rejouée. On mesure le taux de succès, la médiane et le 90e percentile des erreurs, et la cohérence de l'incertitude (test NEES / χ²).

## Utilisation

**Google Colab** : cliquer sur le badge en haut de la page, puis *Exécution → Tout exécuter* (environ 1 minute).

**En local :**
```bash
pip install -r requirements.txt
jupyter notebook fmcw_calibration.ipynb
# ou bien
python fmcw_calibration.py
```

## Structure
```
fmcw_calibration.ipynb   # notebook commenté (version principale)
fmcw_calibration.py      # même code en script Python
figures/                 # figures générées
requirements.txt
```

## Limites et perspectives
- Le modèle est en 2D (x, y, lacet). L'étape suivante est de passer à 6 degrés de liberté (hauteur, tangage, roulis) avec un réseau en élévation.
- Les défauts du réseau d'antennes (gain/phase, couplage) et les multitrajets ne sont pas modélisés. Il faudrait les coupler à une calibration intrinsèque.
- Des méthodes **sans cible** restent à comparer : vitesse radiale des cibles mobiles (ego-motion), alignement sur carte (ICP), cibles d'opportunité.
- Une validation sur données réelles reste à faire : campagne de mesure avec réflecteurs en coin relevés au GPS RTK ou à la station totale.

## Auteure
**Dihia LACENE** · M1 Traitement du Signal et des Images (ULCO) · [lacenedihia.netlify.app](https://lacenedihia.netlify.app/)
