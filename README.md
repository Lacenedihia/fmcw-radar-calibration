# fmcw-radar-calibration

Ce que fait le projet
Il simule les données brutes d’un radar FMCW 24 GHz, les traite (FFT, CFAR 2D, angle), puis retrouve la position et l’orientation du radar à partir de réflecteurs connus. L’algorithme n’a pas besoin d’une estimation de départ de l’installateur. Il donne aussi une incertitude sur le résultat. Enfin, il évalue les performances par Monte Carlo, avec 150 essais par configuration.

## Résultats obtenus

Erreur de position : ≈ 1 cm (médiane).
Erreur d’orientation : ≈ 0,1°.
Taux de succès : 99–100 % quand le signal est assez fort.
