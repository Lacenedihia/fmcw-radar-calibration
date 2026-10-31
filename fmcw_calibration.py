# %% [markdown]
# # Calibration extrinsèque d'un radar FMCW à partir des données brutes
#
# **Objectif** : estimer la position $(x_0, y_0)$ et l'orientation (lacet $\psi$) d'un radar FMCW
# installé sur site, à partir de ses **données brutes** (signal IF) et de réflecteurs de référence
# dont la position est connue (relevé GPS / station totale).
#
# **Chaîne complète** :
# 1. Simulation des données brutes FMCW (cube échantillons × chirps × antennes RX) avec bruit, clutter et cible mobile
# 2. Traitement du signal : FFT distance → FFT Doppler → CA-CFAR 2D → FFT angulaire, avec interpolation sub-bin
# 3. Association des détections aux réflecteurs connus (porte de validation)
# 4. Estimation des paramètres extrinsèques : solution fermée (Kabsch/SVD) puis raffinement par moindres carrés non linéaires pondérés
# 5. Évaluation des performances : Monte Carlo en fonction du SNR et du nombre de réflecteurs, incertitude (covariance) vs erreur réelle
#
# Auteure : Dihia LACENE

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.ndimage import uniform_filter, maximum_filter
from scipy.optimize import least_squares, minimize_scalar
import os

C = 3e8
rng = np.random.default_rng(0)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

# %% [markdown]
# ## 1. Paramètres du radar
# Radar 24 GHz, bande 250 MHz (résolution distance 0,6 m), réseau virtuel de 8 antennes RX espacées de λ/2.

# %%
@dataclass
class RadarConfig:
    fc: float = 24e9          # fréquence porteuse (Hz)
    B: float = 250e6          # bande balayée (Hz)
    Tc: float = 100e-6        # durée d'un chirp (s)
    fs: float = 2.56e6        # fréquence d'échantillonnage IF complexe (Hz)
    N: int = 256              # échantillons par chirp
    M: int = 64               # chirps par trame
    L: int = 8                # antennes RX (réseau linéaire uniforme)
    noise_std: float = 1.0    # écart-type du bruit complexe par échantillon

    @property
    def lam(self):  return C / self.fc
    @property
    def slope(self): return self.B / self.Tc
    @property
    def range_res(self): return C / (2 * self.B)
    @property
    def r_max(self): return self.fs * C / (2 * self.slope)
    @property
    def v_max(self): return self.lam / (4 * self.Tc)

cfg = RadarConfig()
print(f"Longueur d'onde      : {cfg.lam*1e3:.1f} mm")
print(f"Résolution distance  : {cfg.range_res:.2f} m")
print(f"Portée max (IF)      : {cfg.r_max:.1f} m")
print(f"Vitesse max non ambiguë : ±{cfg.v_max:.1f} m/s")
print(f"Résolution vitesse   : {cfg.lam/(2*cfg.M*cfg.Tc):.2f} m/s")
print(f"Résolution angulaire (boresight) : {np.degrees(2/cfg.L):.1f}°")

# %% [markdown]
# ## 2. Scène et géométrie
# Repère monde : relevé du site. Repère radar : axe $x$ = axe de visée, $y$ à gauche.
# Un point monde $P$ s'exprime dans le repère radar par $d = R(\psi)^\top (P - p_0)$,
# puis distance $r = \|d\|$ et azimut $\theta = \mathrm{atan2}(d_y, d_x)$.

# %%
def rot(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])

def world_to_radar(P, pose):
    """P : (K,2) points monde ; pose = (x0, y0, psi). Retourne (K,2) en repère radar."""
    x0, y0, psi = pose
    return (np.atleast_2d(P) - np.array([x0, y0])) @ rot(psi)   # = R^T (P - p0) ligne par ligne

def polar(d):
    return np.hypot(d[:, 0], d[:, 1]), np.arctan2(d[:, 1], d[:, 0])

@dataclass
class Scene:
    reflectors: np.ndarray                   # (K,2) positions monde connues des réflecteurs (m)
    refl_rcs: np.ndarray                     # SER des réflecteurs (m²)
    clutter: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    clutter_rcs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    movers: list = field(default_factory=list)   # [(pos monde (2,), vitesse radiale m/s, rcs)]

def make_scene(true_pose, n_refl=5, n_clutter=6, rng=rng):
    """Réflecteurs répartis dans le champ de vue (±50°, 15–110 m) + clutter statique + 1 piéton."""
    x0, y0, psi = true_pose
    az = np.radians(np.linspace(-45, 45, n_refl)) + np.radians(rng.uniform(-4, 4, n_refl))
    rr = rng.uniform(20, 100, n_refl)
    d = np.c_[rr * np.cos(az), rr * np.sin(az)]
    refl = d @ rot(psi).T + np.array([x0, y0])          # radar -> monde
    # clutter (poteaux, arbres...) loin des réflecteurs
    clut = []
    while len(clut) < n_clutter:
        a, r = np.radians(rng.uniform(-60, 60)), rng.uniform(10, 140)
        p = np.array([r * np.cos(a), r * np.sin(a)]) @ rot(psi).T + np.array([x0, y0])
        if np.min(np.linalg.norm(refl - p, axis=1)) > 6:
            clut.append(p)
    mover = (np.array([x0, y0]) + rot(psi) @ np.array([45.0, -8.0]), 1.5, 1.0)
    return Scene(refl, np.full(n_refl, 50.0),   # trièdre ~20 cm d'arête à 24 GHz ≈ 40-50 m²
                 np.array(clut).reshape(-1, 2), rng.uniform(0.3, 3, n_clutter), [mover])

# %% [markdown]
# ## 3. Simulation des données brutes FMCW
# Pour une cible à la distance $r$, vitesse radiale $v$, azimut $\theta$ :
#
# $$s[n,m,l] = A\,\exp\Big(j2\pi\big(f_b+f_d\big)\tfrac{n}{f_s} + j2\pi f_d\, m T_c + j\pi l \sin\theta + j\tfrac{4\pi r}{\lambda}\Big)$$
#
# avec $f_b = 2 S r / c$ (fréquence de battement), $f_d = 2v/\lambda$ (Doppler).
# L'amplitude suit l'équation radar : SNR $\propto \sigma / r^4$.

# %%
def simulate_cube(scene, pose, cfg, snr_ref_db=-20.0, r_ref=50.0, rcs_ref=10.0, rng=rng):
    """Retourne le cube IF complexe (L, M, N). snr_ref_db = SNR par échantillon d'un réflecteur 10 m² à 50 m."""
    n = np.arange(cfg.N) / cfg.fs
    m = np.arange(cfg.M) * cfg.Tc
    l = np.arange(cfg.L)
    targets = []
    for P, rcs in zip(np.vstack([scene.reflectors, scene.clutter]), np.r_[scene.refl_rcs, scene.clutter_rcs]):
        targets.append((P, 0.0, rcs))
    targets += scene.movers
    cube = np.zeros((cfg.L, cfg.M, cfg.N), complex)
    for P, v, rcs in targets:
        r, th = polar(world_to_radar(P, pose))
        r, th = r[0], th[0]
        if abs(th) > np.radians(70) or r > 0.95 * cfg.r_max:
            continue                                     # hors champ de vue
        snr = 10 ** (snr_ref_db / 10) * (rcs / rcs_ref) * (r_ref / r) ** 4
        A = np.sqrt(snr) * cfg.noise_std
        fb, fd = 2 * cfg.slope * r / C, 2 * v / cfg.lam
        s_n = np.exp(2j * np.pi * (fb + fd) * n)
        s_m = np.exp(2j * np.pi * fd * m)
        s_l = np.exp(1j * np.pi * l * np.sin(th))
        phi0 = np.exp(1j * 4 * np.pi * r / cfg.lam)
        cube += A * phi0 * s_l[:, None, None] * s_m[None, :, None] * s_n[None, None, :]
    noise = (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape)) / np.sqrt(2)
    return cube + cfg.noise_std * noise

# %% [markdown]
# ## 4. Traitement du signal
# - Fenêtrage de Hann, FFT distance (axe échantillons) puis FFT Doppler (axe chirps)
# - Intégration non cohérente sur les antennes → carte distance-Doppler
# - **CA-CFAR 2D** (cellules de garde + cellules de référence) + maxima locaux
# - Pour chaque détection : FFT angulaire zéro-paddée sur les 8 antennes
# - **Raffinement sub-bin** en distance et en angle : maximisation de la DTFT autour du pic (estimateur du maximum de
#   vraisemblance d'une sinusoïde), plus précis que l'interpolation 3 points qui présente un biais selon la position dans le bin

# %%
def range_doppler(cube, cfg):
    w_n, w_m = np.hanning(cfg.N), np.hanning(cfg.M)
    rfft = np.fft.fft(cube * w_n[None, None, :], axis=2)
    rd = np.fft.fftshift(np.fft.fft(rfft * w_m[None, :, None], axis=1), axes=1)
    return rd                                             # (L, M, N)

def ca_cfar_2d(power, guard=(2, 2), train=(4, 8), pfa=1e-6):
    """CA-CFAR 2D sur carte de puissance (M, N)."""
    gm, gn = guard; tm, tn = train
    big = np.array([2 * (gm + tm) + 1, 2 * (gn + tn) + 1])
    small = np.array([2 * gm + 1, 2 * gn + 1])
    sum_big = uniform_filter(power, big, mode="wrap") * big.prod()
    sum_small = uniform_filter(power, small, mode="wrap") * small.prod()
    n_train = big.prod() - small.prod()
    noise = (sum_big - sum_small) / n_train
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    return power > alpha * noise

def refine_freq(X, w, k0, n_fft):
    """Fréquence normalisée (en bins de n_fft) maximisant sum |DTFT(X*w)|^2 sur l'axe -1, autour de k0."""
    n = np.arange(X.shape[-1])
    Xw = X * w
    def cost(k):
        return -np.sum(np.abs(Xw @ np.exp(-2j * np.pi * k * n / n_fft)) ** 2)
    return minimize_scalar(cost, bounds=(k0 - 1, k0 + 1), method="bounded", options={"xatol": 1e-4}).x

def detect(cube, cfg, pfa=1e-6, n_angle_fft=256):
    rd = range_doppler(cube, cfg)
    power = np.sum(np.abs(rd) ** 2, axis=0)              # intégration non cohérente sur antennes
    mask = ca_cfar_2d(power, pfa=pfa)
    peaks = mask & (power == maximum_filter(power, size=(3, 3)))
    peaks[:, :2] = False; peaks[:, -2:] = False          # ignore DC et bords (IF complexe : bins 0..N-1 = 0..r_max)
    dop = np.fft.fftshift(np.fft.fft(cube * np.hanning(cfg.M)[None, :, None], axis=1), axes=1)  # Doppler seul (L, M, N)
    w_n, w_l = np.hanning(cfg.N), np.hanning(cfg.L + 2)[1:-1]
    dets = []
    for mi, ni in zip(*np.nonzero(peaks)):
        # distance : raffinement DTFT sur l'axe échantillons (toutes antennes, non cohérent)
        k_r = refine_freq(dop[:, mi, :], w_n, ni, cfg.N)
        r = k_r * cfg.fs / cfg.N * C / (2 * cfg.slope)
        v = (mi - cfg.M // 2) / (cfg.M * cfg.Tc) * cfg.lam / 2
        # angle : FFT zéro-paddée puis raffinement DTFT sur le vecteur d'antennes
        x = rd[:, mi, ni]
        A = np.abs(np.fft.fftshift(np.fft.fft(x * w_l, n_angle_fft))) ** 2
        k = int(np.argmax(A)) - n_angle_fft // 2
        k_a = refine_freq(x[None, :], w_l, k, n_angle_fft)
        th = np.arcsin(np.clip(2 * k_a / n_angle_fft, -1, 1))       # u = sin(theta) = 2k/Nfft (d = lambda/2)
        snr_db = 10 * np.log10(power[mi, ni] / np.median(power))
        dets.append((r, th, v, snr_db))
    return np.array(dets).reshape(-1, 4), power

# %% [markdown]
# **Validation de l'estimateur** : avec une cible unique (pas d'interférence), l'erreur doit être très faible.

# %%
g = np.random.default_rng(2)
errs = []
for _ in range(20):
    r0, a0 = g.uniform(20, 100), np.radians(g.uniform(-45, 45))
    sc1 = Scene(np.array([[r0 * np.cos(a0), r0 * np.sin(a0)]]), np.array([50.0]))
    d1, _ = detect(simulate_cube(sc1, np.zeros(3), cfg, 0.0, rng=g), cfg)
    j = np.argmin(np.abs(d1[:, 0] - r0))
    errs.append((d1[j, 0] - r0, np.degrees(d1[j, 1] - a0)))
errs = np.array(errs)
print(f"Cible unique (SNR élevé) : erreur distance RMS = {np.sqrt(np.mean(errs[:,0]**2))*1000:.2f} mm, "
      f"erreur azimut RMS = {np.sqrt(np.mean(errs[:,1]**2)):.3f}°")

# %% [markdown]
# ## 5. Calibration extrinsèque
# **Association sans pose initiale (RANSAC)** : les distances entre deux réflecteurs sont invariantes par
# rotation/translation. Pour chaque paire de détections statiques dont l'écart correspond à celui d'une paire de
# réflecteurs connus, on calcule une pose candidate, puis on compte les réflecteurs qui retombent sur une détection
# (inliers). La meilleure hypothèse donne l'appariement — **aucune mesure préalable de l'installateur n'est nécessaire**,
# et le clutter / les cibles parasites sont rejetés.
#
# **Solution fermée (Kabsch / SVD)** : les détections converties en cartésien dans le repère radar $d_k$ et les
# positions monde $P_k$ sont liées par $P_k = R(\psi)\,d_k + p_0$ → alignement rigide 2D par SVD.
#
# **Raffinement** : moindres carrés non linéaires pondérés dans l'espace de mesure (distance, azimut) :
#
# $$\hat{\Theta} = \arg\min_{x_0,y_0,\psi} \sum_k \left(\frac{r_k - \hat r_k(\Theta)}{\sigma_r}\right)^2 + \left(\frac{\theta_k - \hat\theta_k(\Theta)}{\sigma_\theta}\right)^2$$
#
# La covariance $(J^\top J)^{-1}$ donne l'incertitude de la calibration — utile pour valider une installation sur le terrain.

# %%
def static_detections_xy(dets, v_max_static=0.3):
    static = dets[np.abs(dets[:, 2]) < v_max_static]
    xy = np.c_[static[:, 0] * np.cos(static[:, 1]), static[:, 0] * np.sin(static[:, 1])]
    return static, xy

def apply_pose(d_radar, pose):
    """repère radar -> repère monde"""
    return d_radar @ rot(pose[2]).T + pose[:2]

def match(det_world, refl, gate):
    """Appariement un-pour-un glouton réflecteur -> détection la plus proche."""
    D = np.linalg.norm(refl[:, None, :] - det_world[None, :, :], axis=2)
    idx_r, idx_d, cost = [], [], 0.0
    for k in np.argsort(D.min(1)):
        j = int(np.argmin(D[k]))
        if D[k, j] < gate and j not in idx_d:
            idx_r.append(int(k)); idx_d.append(j); cost += D[k, j]
            D[:, j] = np.inf
    return np.array(idx_r, int), np.array(idx_d, int), cost

def ransac_associate(dets, refl, tol=1.5, gate=2.0):
    """Retourne (indices réflecteurs, mesures (r, theta)) sans pose initiale."""
    static, xy = static_detections_xy(dets)
    n = len(xy)
    if n < 2:
        return np.array([], int), np.zeros((0, 2))
    Dd = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    Dr = np.linalg.norm(refl[:, None] - refl[None], axis=2)
    best = (0, np.inf, None)
    for i in range(n):
        for j in range(i + 1, n):
            for a, b in zip(*np.nonzero(np.abs(Dr - Dd[i, j]) < tol)):
                if a == b:
                    continue
                pose = kabsch_2d(xy[[i, j]], refl[[a, b]])
                ir, idd, cost = match(apply_pose(xy, pose), refl, gate)
                if len(ir) > best[0] or (len(ir) == best[0] and cost < best[1]):
                    best = (len(ir), cost, pose)
    if best[2] is None:
        return np.array([], int), np.zeros((0, 2))
    ir, idd, _ = match(apply_pose(xy, best[2]), refl, gate)
    return ir, static[idd, :2]

def kabsch_2d(d_radar, P_world):
    """P = R d + t  →  (x0, y0, psi) par SVD."""
    cd, cp = d_radar.mean(0), P_world.mean(0)
    H = (d_radar - cd).T @ (P_world - cp)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    t = cp - R @ cd
    return np.array([t[0], t[1], np.arctan2(R[1, 0], R[0, 0])])

def calibrate(meas, P_world, pose0, sigma_r=0.05, sigma_th=np.radians(0.5)):
    def residuals(pose):
        r_hat, th_hat = polar(world_to_radar(P_world, pose))
        dth = np.angle(np.exp(1j * (meas[:, 1] - th_hat)))
        return np.r_[(meas[:, 0] - r_hat) / sigma_r, dth / sigma_th]
    sol = least_squares(residuals, pose0, method="lm" if 2 * len(meas) >= 3 else "trf")
    J = sol.jac
    dof = max(2 * len(meas) - 3, 1)
    s2 = np.sum(sol.fun ** 2) / dof                       # facteur de variance a posteriori
    cov = np.linalg.pinv(J.T @ J) * s2
    return sol.x, cov, sol

def run_calibration(scene, true_pose, cfg, snr_ref_db=-20.0, rng=rng):
    cube = simulate_cube(scene, true_pose, cfg, snr_ref_db, rng=rng)   # données brutes (inconnues : la pose)
    dets, power = detect(cube, cfg)
    idx, meas = ransac_associate(dets, scene.reflectors)
    if len(idx) < 2:
        return None
    P = scene.reflectors[idx]
    d_radar = np.c_[meas[:, 0] * np.cos(meas[:, 1]), meas[:, 0] * np.sin(meas[:, 1])]
    pose_k = kabsch_2d(d_radar, P)
    pose_ls, cov, _ = calibrate(meas, P, pose_k)
    return dict(cube=cube, dets=dets, power=power, idx=idx, meas=meas,
                pose_kabsch=pose_k, pose=pose_ls, cov=cov)

# %% [markdown]
# ## 6. Démonstration sur un cas
# Pose réelle (inconnue) : radar en (12,40 m ; −3,70 m), lacet 27,5°. Pour comparaison, l'installateur l'estime « à vue » en (11 ; −2,5), 35° — cette valeur **n'est pas utilisée** par l'algorithme.

# %%
true_pose = np.array([12.40, -3.70, np.radians(27.5)])
pose_guess = np.array([11.0, -2.5, np.radians(35.0)])
scene = make_scene(true_pose, n_refl=5, n_clutter=6, rng=np.random.default_rng(3))
res = run_calibration(scene, true_pose, cfg, snr_ref_db=-20.0, rng=np.random.default_rng(4))

def fmt(p): return f"x0={p[0]:7.3f} m  y0={p[1]:7.3f} m  psi={np.degrees(p[2]):7.3f}°"
print("Vraie pose        :", fmt(true_pose))
print("Pose installateur :", fmt(pose_guess))
print("Kabsch (SVD)      :", fmt(res["pose_kabsch"]))
print("Moindres carrés   :", fmt(res["pose"]))
err = res["pose"] - true_pose
std = np.sqrt(np.diag(res["cov"]))
print(f"\nErreur position : {np.hypot(*err[:2])*100:.1f} cm   | erreur lacet : {np.degrees(err[2]):+.3f}°")
print(f"Incertitude 1σ  : x {std[0]*100:.1f} cm, y {std[1]*100:.1f} cm, psi {np.degrees(std[2]):.3f}°")
print(f"{len(res['dets'])} détections CFAR, {len(res['idx'])}/{len(scene.reflectors)} réflecteurs associés")

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
r_axis = np.arange(cfg.N) * cfg.fs / cfg.N * C / (2 * cfg.slope)
v_axis = (np.arange(cfg.M) - cfg.M // 2) / (cfg.M * cfg.Tc) * cfg.lam / 2
P_db = 10 * np.log10(res["power"] / np.median(res["power"]))
im = ax[0].imshow(P_db, aspect="auto", origin="lower", cmap="viridis",
                  extent=[r_axis[0], cfg.r_max, v_axis[0], v_axis[-1]], vmin=0, vmax=60)
ax[0].scatter(res["dets"][:, 0], res["dets"][:, 2], facecolors="none", edgecolors="r", s=60, label="détections CFAR")
ax[0].set(xlabel="Distance (m)", ylabel="Vitesse radiale (m/s)", title="Carte distance-Doppler (dB / bruit)")
ax[0].legend(loc="upper right"); ax[0].grid(False)
plt.colorbar(im, ax=ax[0])

# Carte distance-azimut (FFT angulaire sur le bin Doppler nul)
rd = range_doppler(res["cube"], cfg)
ra = np.abs(np.fft.fftshift(np.fft.fft(rd[:, cfg.M // 2, :] * np.hanning(cfg.L + 2)[1:-1, None], 128, axis=0), axes=0)) ** 2
u = 2 * (np.arange(128) - 64) / 128
th = np.degrees(np.arcsin(u))
R_g, T_g = np.meshgrid(r_axis, np.radians(th))
ra_db = 10 * np.log10(ra / np.median(ra))
ax[1].pcolormesh(R_g * np.sin(T_g), R_g * np.cos(T_g), ra_db, cmap="viridis", vmin=18, vmax=50, shading="gouraud")
d_m = np.c_[res["meas"][:, 0] * np.sin(res["meas"][:, 1]), res["meas"][:, 0] * np.cos(res["meas"][:, 1])]
ax[1].scatter(d_m[:, 0], d_m[:, 1], marker="x", c="r", s=60, label="réflecteurs associés")
ax[1].set(xlabel="Latéral (m)  [gauche = +]", ylabel="Axe de visée (m)", title="Carte distance-azimut (8 antennes : lobe large,\nposition fine obtenue par estimation sub-bin)", aspect="equal", ylim=(0, 110), xlim=(-90, 90))
ax[1].invert_xaxis(); ax[1].legend(loc="upper right"); ax[1].grid(False)
plt.tight_layout(); plt.savefig("figures/range_doppler_azimuth.png", bbox_inches="tight"); plt.show()

# %%
def draw_radar(ax, pose, color, label, L=12):
    x0, y0, psi = pose
    ax.plot(x0, y0, "o", c=color, ms=8, label=label)
    for a in (psi - np.radians(60), psi, psi + np.radians(60)):
        ax.plot([x0, x0 + L * np.cos(a)], [y0, y0 + L * np.sin(a)], c=color, lw=2 if a == psi else 0.8,
                ls="-" if a == psi else "--")

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.scatter(*scene.reflectors.T, marker="^", s=120, c="tab:orange", edgecolors="k", label="réflecteurs (connus)", zorder=3)
ax.scatter(*scene.clutter.T, marker="s", s=30, c="gray", label="clutter")
ax.scatter(*scene.movers[0][0], marker="*", s=150, c="tab:purple", label="piéton (mobile)")
draw_radar(ax, pose_guess, "tab:red", "pose installateur")
draw_radar(ax, true_pose, "k", "pose réelle")
draw_radar(ax, res["pose"], "tab:green", "pose estimée")
P_est = np.c_[res["meas"][:, 0] * np.cos(res["meas"][:, 1]), res["meas"][:, 0] * np.sin(res["meas"][:, 1])] @ rot(res["pose"][2]).T + res["pose"][:2]
ax.scatter(*P_est.T, marker="+", s=150, c="tab:green", label="mesures reprojetées")
ax.set(xlabel="x monde (m)", ylabel="y monde (m)", title="Scène vue de dessus et résultat de calibration", aspect="equal")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig("figures/scene_calibration.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 7. Évaluation des performances (Monte Carlo)
# À chaque essai : nouvelle scène aléatoire, nouvelle pose réelle (orientation quelconque), nouveau bruit — **toute la chaîne** (simulation → CFAR → association → calibration) est rejouée.
#
# On mesure : **taux de succès** (≥ 3 réflecteurs associés et erreur < 2 m), erreurs de position et de lacet
# (médiane et 90e percentile, plus robustes que la RMSE aux quelques cas d'interférence), et la **cohérence** de
# l'incertitude prédite : part des essais où l'erreur réelle tombe dans l'ellipsoïde de confiance à 99 % (attendu ≈ 99 %).

# %%
def monte_carlo(n_trials=100, snr_ref_db=-20.0, n_refl=5, seed=0):
    g = np.random.default_rng(seed)
    pos_err, yaw_err, nees, fails = [], [], [], 0
    for _ in range(n_trials):
        tp = np.array([g.uniform(-20, 20), g.uniform(-20, 20), g.uniform(-np.pi, np.pi)])
        sc = make_scene(tp, n_refl=n_refl, n_clutter=6, rng=g)
        out = run_calibration(sc, tp, cfg, snr_ref_db, rng=g)
        if out is None or len(out["idx"]) < 3:
            fails += 1; continue
        e = out["pose"] - tp
        e[2] = np.angle(np.exp(1j * e[2]))
        if np.hypot(*e[:2]) > 2:                          # mauvaise association
            fails += 1; continue
        pos_err.append(np.hypot(*e[:2])); yaw_err.append(np.degrees(e[2]))
        nees.append(e @ np.linalg.solve(out["cov"], e))
    pos_err, yaw_err, nees = np.array(pos_err), np.abs(np.array(yaw_err)), np.array(nees)
    q = lambda a, p: np.percentile(a, p) if len(a) else np.nan
    return dict(success=1 - fails / n_trials,
                pos_med=q(pos_err, 50), pos_p90=q(pos_err, 90),
                yaw_med=q(yaw_err, 50), yaw_p90=q(yaw_err, 90),
                consist=np.mean(nees < 11.34) if len(nees) else np.nan,   # chi2(3 ddl) à 99 %
                pos_err=pos_err, yaw_err=yaw_err)

snr_list = [-40, -35, -30, -25, -20, -15, -10]
proc_gain = 10 * np.log10(cfg.N * cfg.M)          # gain des FFT distance + Doppler
res_snr = [monte_carlo(150, s, 5, seed=10 + i) for i, s in enumerate(snr_list)]
print(" SNR intégré cible 10 m² @50 m | succès | pos. méd / P90 (cm) | lacet méd / P90 (°) | cohérence 99 %")
for s, r in zip(snr_list, res_snr):
    print(f" {s + proc_gain:>23.0f} dB | {r['success']*100:5.0f}% | {r['pos_med']*100:7.1f} / {r['pos_p90']*100:5.1f}"
          f"     | {r['yaw_med']:7.3f} / {r['yaw_p90']:.3f}    | {r['consist']*100:5.0f}%")

# %%
nref_list = [3, 4, 5, 6, 8]
res_nref = [monte_carlo(150, -25, n, seed=50 + i) for i, n in enumerate(nref_list)]
print(f" Nb réflecteurs (SNR {-25 + proc_gain:.0f} dB) | succès | pos. méd / P90 (cm) | lacet méd / P90 (°)")
for n, r in zip(nref_list, res_nref):
    print(f" {n:>25} | {r['success']*100:5.0f}% | {r['pos_med']*100:7.1f} / {r['pos_p90']*100:5.1f}"
          f"     | {r['yaw_med']:7.3f} / {r['yaw_p90']:.3f}")

# %%
x_snr = np.array(snr_list) + proc_gain
ok = np.array([r["success"] >= 0.2 for r in res_snr])      # erreurs tracées seulement si assez d'essais réussis
pick = lambda key, scale=1: np.array([r[key] * scale for r in res_snr])[ok]
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(x_snr, [r["success"] * 100 for r in res_snr], "o-", c="tab:green")
ax[0, 0].set(xlabel="SNR intégré, cible 10 m² à 50 m (dB)", ylabel="Taux de succès (%)", title="Robustesse vs SNR (5 réflecteurs)", ylim=(-5, 105))
ax[0, 1].plot(x_snr[ok], pick("pos_med", 100), "o-", label="médiane")
ax[0, 1].plot(x_snr[ok], pick("pos_p90", 100), "o--", label="90e percentile")
ax[0, 1].set(xlabel="SNR intégré, cible 10 m² à 50 m (dB)", ylabel="Erreur de position (cm)", title="Précision en position vs SNR", ylim=(0, None)); ax[0, 1].legend()
ax[1, 0].plot(x_snr[ok], pick("yaw_med"), "s-", c="tab:orange", label="médiane")
ax[1, 0].plot(x_snr[ok], pick("yaw_p90"), "s--", c="tab:orange", label="90e percentile")
ax[1, 0].set(xlabel="SNR intégré, cible 10 m² à 50 m (dB)", ylabel="Erreur de lacet (°)", title="Précision en orientation vs SNR", ylim=(0, None)); ax[1, 0].legend()
a2 = ax[1, 1]; a3 = a2.twinx()
a2.bar(nref_list, [r["success"] * 100 for r in res_nref], color="tab:green", alpha=0.35, label="succès (%)")
a3.plot(nref_list, [r["yaw_p90"] for r in res_nref], "s--", c="tab:orange", label="lacet P90 (°)")
a3.plot(nref_list, [r["pos_p90"] for r in res_nref], "o--", c="tab:blue", label="position P90 (m)")
a2.set(xlabel="Nombre de réflecteurs", ylabel="Taux de succès (%)", title=f"Effet du nombre de réflecteurs (SNR {-25 + proc_gain:.0f} dB)", ylim=(0, 105))
a3.set_ylabel("Erreur P90"); a3.set_ylim(0, None); a3.grid(False)
h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels(); a3.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig("figures/monte_carlo.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 8. Conclusions et limites
# - Chaîne complète validée en simulation : sans aucune pose initiale, la calibration atteint une précision
#   **centimétrique** en position et de l'ordre du **dixième de degré** en lacet dès que le SNR est suffisant.
# - L'erreur de lacet est dominée par la précision angulaire du réseau (8 antennes) ; la position par la précision en distance.
# - Plus de réflecteurs améliorent surtout la **robustesse** (taux de succès) ; au-delà de 5–6, le gain en précision est faible.
# - Avec une cible isolée, l'estimateur atteint le mm et le centième de degré : l'erreur résiduelle en scène réelle vient
#   des **interférences entre réflecteurs proches** (lobes secondaires). Recommandation d'installation : espacer les
#   réflecteurs de plusieurs cellules de résolution en distance (> 2–3 m) et en azimut.
# - Le NEES proche de 3 indique que l'incertitude prédite par la covariance est cohérente → critère de validation d'installation.
#
# **Limites / suite** :
# - Modèle 2D (x, y, lacet) : étendre à 6 DDL (hauteur, tangage, roulis) avec un réseau en élévation.
# - Pas de défauts du réseau d'antennes (couplage, déséquilibres de gain/phase) ni de multitrajets → calibration intrinsèque à ajouter.
# - Méthodes **sans cible** à comparer : exploitation de la vitesse radiale de cibles mobiles (ego-motion), alignement sur carte (ICP), cibles d'opportunité.
# - Validation sur données réelles : campagne de mesure avec réflecteurs en coin relevés au GPS RTK / station totale.
