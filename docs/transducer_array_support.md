# Transducer array support in PyMUST

How PyMUST defines transducer arrays (1-D, 2-D and 3-D), and how each algorithm uses the
position of every element.

- **§1 and §6 describe the current code**, which supports arbitrary element positions everywhere.
- **§2–§5 are the analysis as traced at commit `c4399e6`**, before that support was added. Their line
  numbers refer to that commit; the §5.1 "Location" column refers to `2b2a0cd`. §6 lists what changed.
- Paths are relative to the repository root.

---

## 1. Summary

| Array kind | Element positions | How positions are produced | Accessor | Algorithms |
|---|---|---|---|---|
| **Uniform linear** (1-D) | $(x_e, 0, 0)$, equal pitch, centred on $x=0$ | Translated from `Nelements`, `pitch`, `radius=inf` | `getElementPositions()` / `get_element_coordinates()` | all 2-D functions; 3-D functions as $(x_e, 0, 0)$ |
| **Uniform convex** (1-D) | $(x_e, 0, z_e)$ on an arc, tilted by $\theta_e$ | Translated from `Nelements`, `pitch`, `radius<inf` | `getElementPositions()` | all 2-D functions (3-D functions: no, they assume $+z$ normals) |
| **Arbitrary 2-D points** | $(x_e, z_e)$, any layout in the $x$–$z$ plane | `param.elements`: 1 row (x) or 2 rows (x, z) | `getElementPositions()` | `txdelay`, `pfield`, `simus`, `mkmovie`, `dasmtx`, `impolgrid` |
| **Arbitrary 3-D points** | $(x_e, y_e, z_e)$, any layout (a height field over $(x, y)$ for the virtual source) | `param.elements`: 3 rows (x, y, z), or 2 rows (x, y) with $z_e=0$ | `get_element_coordinates()` | `txdelay3`, `pfield3`, `simus3`, `dasmtx3` |

Every model shares these assumptions:

- **Identical elements:** all elements have the same rectangular size (`width` by `height`).
- **Rotation:** element rotation only exists for convex arrays, as an in-plane tilt $\theta_e$ about the
  $y$ axis.
- **Matrix arrays face +z:** every element of a matrix array has its normal along $+z$.
- **No per-element frame:** no array type has a per-element normal or orientation vector.

**What "1-D / 2-D / 3-D" means in PyMUST:**

- **1-D arrays** (linear or convex) are line or arc arrays in the $x$–$z$ plane.
  - Simulated in **2-D space** when `y` is empty or zero.
  - Simulated in **3-D space** when `y` is given. The element height and elevation focus are then
    modelled with a multi-Gaussian beam model (MGBM), not by placing elements along $y$.
- **2-D arrays** are planar matrix arrays in the $z=0$ plane. They are always simulated in 3-D space.
- **Arbitrary element placement** is supported everywhere (§6): any $(x, z)$ points for the 2-D
  functions and any $(x, y, z)$ points for the 3-D functions. Element normals still point along $+z$,
  except for the uniform convex probes.

---

## 2. How arrays are defined

### 2.1 The `Param` container

`Param` (`src/pymust/utils.py:81`) is a `dict` with attribute access (`dotdict`,
`src/pymust/utils.py:10`).

- **Recognised names:** `Param.names` (`src/pymust/utils.py:83-88`) lists the field names used to
  normalise their case: `pitch`, `kerf`, `width`, `height`, `focus`, `radius`, `Nelements`, `RXdelay`,
  `TXapodization`, and others.
- **`elements` is not in that list.** It is accepted only because `Param` is an open dict.
- **Missing fields return `None`**, because `__getattr__ = dict.get`. A field that was never set does
  not raise `AttributeError`.

Geometry fields:

| Field | Meaning | Used by |
|---|---|---|
| `Nelements` | number of elements E | 1-D arrays (positions); everywhere as a size check |
| `pitch` | centre-to-centre spacing (m) | 1-D arrays |
| `width` / `kerf` | element width along the array; `pitch = width + kerf` | all (directivity, sub-element splitting, f-number) |
| `height` | element size in elevation ($y$) | `pfield` (MGBM), `pfield3`/`dasmtx3` (y-directivity, f-number) |
| `focus` | elevation focus of a 1-D array (lens) | `pfield` (MGBM) |
| `radius` | radius of curvature; `inf` = linear | 1-D arrays |
| `elements` | explicit coordinates, 2×E $(x, y)$ or 3×E $(x, y, z)$ | 2-D/3-D functions |
| `TXapodization` | per-element TX weight (1×E) | `pfield`, `pfield3` |
| `RXdelay` | per-element receive delay (1×E) | `simus`, `simus3` |
| `TXdelay` | per-element TX delays (T×E, T delay series), written by `txdelay*` | `dasmtx*` |

### 2.2 Predefined probes: `getparam` (1-D only)

`src/pymust/getparam.py:5`, `getparam(probe)`, returns a `Param` describing a **uniform linear or
convex array**. No predefined matrix probe exists.

| Probe | Line | Type | E | pitch (µm) | width (µm) | radius | height / focus |
|---|---|---|---|---|---|---|---|
| `L11-5v` | 81 | linear | 128 | 300 | 270 | inf | 5 mm / 18 mm |
| `L12-3v` | 92 | linear | 192 | 200 | 170 | inf | 5 mm / 20 mm |
| `C5-2v` | 103 | **convex** | 128 | 508 | 460 | 49.57 mm | 13.5 mm / 60 mm |
| `P4-2v` | 114 | linear (phased) | 64 | 300 | 250 | inf | 14 mm / 60 mm |
| `PA4-2/20` | 126 | linear | 64 | 300 | (kerf 50) | – | 14 mm / – |
| `L9-4/38` | 134 | linear | 128 | 304.8 | (kerf 35) | – | 6 mm / – |
| `LA530` | 142 | linear | 192 | 245 | (kerf 30) | – | – |
| `L14-5/38` | 150 | linear | 128 | 304.8 | (kerf 25) | – | – |
| `L14-5W/60` | 158 | linear | 128 | 472 | (kerf 25) | – | – |
| `P6-3` | 166 | linear | 64 | 218 | (kerf 25) | – | – |

Probes without `radius` default to linear: each consumer sets `param.radius = np.inf` (for example
`src/pymust/pfield.py:302-303`).

### 2.3 1-D arrays: `Param.getElementPositions()`

`src/pymust/utils.py:90-113` returns four `1×E` arrays: `xe, ze, THe, h`.

**Linear** (`utils.py:97-102`):

$$x_e = \left(e - \tfrac{E-1}{2}\right) p,\qquad z_e = 0,\qquad \theta_e = 0,\qquad h = 0,\qquad e = 0..E-1$$

**Convex** (`utils.py:103-112`): the elements sit on an arc of radius $R$, and the chord between
the end elements is placed on $z=0$:

$$\text{chord} = 2R\sin\!\big((E-1)\arcsin(p/2R)\big),\quad h=\sqrt{R^2-\text{chord}^2/4}\ (\text{apothem})$$
$$\theta_e = \text{linspace}\big(\operatorname{atan2}(-\text{chord}/2,h),\ \operatorname{atan2}(\text{chord}/2,h),\ E\big)$$
$$x_e = R\sin\theta_e,\qquad z_e = R\cos\theta_e - h$$

- $\theta_e$ is the angle between element $e$'s normal and the $z$ axis.
- The arc centre is at $(0, -h)$.
- The end elements lie on $z=0$, and the middle elements bulge to $z>0$ (e.g. C5-2v: $z_{64}\approx 10.1$ mm).

Example declarations (checked against the code):

```python
import numpy as np
import pymust
from pymust.utils import Param

# Predefined linear probe: 128 elements, xe from -19.05 mm to +19.05 mm
param = pymust.getparam("L11-5v")
xe, ze, the, h = param.getElementPositions()      # each (1, 128); ze = the = h = 0

# Predefined convex probe: theta_e in [-0.651, +0.651] rad, apothem h = 39.4 mm
param = pymust.getparam("C5-2v")

# Custom uniform linear array. 'radius' must be set explicitly, otherwise
# getElementPositions() raises TypeError: np.isinf(None)
param = Param(fc=5e6, pitch=300e-6, width=250e-6, Nelements=32, bandwidth=70, radius=np.inf)

# 1-D array with elevation focusing (used by pfield/simus when y is given)
param.height = 5e-3   # element height along y
param.focus = 20e-3   # elevation lens focus
```

### 2.4 Planar 2-D arrays: `param.elements` (2×E)

Positions are given directly as a 2-row matrix: row 0 holds $x_e$, row 1 holds $y_e$, and $z_e=0$
is implied. The layout is free, but `txdelay3` "Diverging Wave" requires a full rectangular grid
(`src/pymust/txdelay3.py:208-212`).

Where each 3-D function reads the elements:

| Function | Lines | Accepted shape |
|---|---|---|
| `txdelay3` | `src/pymust/txdelay3.py:175-181` | exactly 2 rows |
| `pfield3` | `src/pymust/pfield3.py:264-270` | exactly 2 rows. A comment notes "ze = 0 in this PFIELD3 version". |
| `simus3` | `src/pymust/simus3.py:264` (E), `:303-304` (positions) | 2 rows |
| `dasmtx3` | `src/pymust/dasmtx3.py:243-254` | 2 or 3 rows (also 2/3 columns, which it transposes) |

`width` and `height` are required and apply to every element (`pfield3.py:273-281`,
`dasmtx3.py:205-214`).

Example declarations (adapted from `examples/pfield3.ipynb`):

```python
# 32 x 32 matrix array, 300 µm pitch, 3 MHz
param = Param(fc=3e6, bandwidth=70, width=250e-6, height=250e-6)
pitch = 300e-6
xe, ye = np.meshgrid((np.arange(1, 33) - 16.5) * pitch, (np.arange(1, 33) - 16.5) * pitch, indexing="ij")
param.elements = np.array([xe.flatten(order="F"), ye.flatten(order="F")])   # shape (2, 1024)

# No 1-D fields (radius, pitch, Nelements) are needed since §5.1 #2 was fixed;
# examples/pfield3.ipynb still sets them, which is harmless
delays = pymust.txdelay3(0, -2e-3, 30e-3, param)   # focused, shape (1, 1024)

# A 1-D linear array expressed as a "2-D" array (64 x 1)
xe = (np.arange(1, 65) - 32.5) * pitch
param3 = Param(fc=3e6, bandwidth=70, width=250e-6, height=14e-3,
               elements=np.array([xe, np.zeros_like(xe)]))
```

### 2.5 Non-planar 3-D positions: `param.elements` (3×E), `dasmtx3` only

```python
param.elements = np.stack([xe, ye, ze])   # (3, E): accepted by dasmtx3 (dasmtx3.py:251-252)
pymust.dasmtx3(1j*np.array([nl, E]), x, y, z, delays, param)   # works
pymust.pfield3(x, y, z, delays, param)                         # AssertionError (pfield3.py:267)
pymust.txdelay3(x0, y0, z0, param)                             # AssertionError (txdelay3.py:176)
```

---

## 3. Notation used in the matrix descriptions

| Symbol | Size | Meaning |
|---|---|---|
| $P$ | | number of field points / pixels / scatterers (`numel(x)`) |
| $E$ | | number of elements |
| $S$ | | sub-elements per element ($M$ for 1-D, $M\cdot N$ for 2-D) |
| $K$ | | number of frequency samples kept above `dBThresh` |
| $T$ | | number of TX delay series (rows of `delaysTX`, e.g. MLT/MPT) |
| $\mathbf{x}_p$ | $P\times 3$ | point coordinates |
| $\mathbf{c}_e$ | $E\times 3$ | element centres |
| $\mathbf{o}_{e,s}$ | $E\times S\times 3$ | sub-element offsets relative to $\mathbf{c}_e$ |
| $\tau_{t,e}$ | $T\times E$ | TX delays |
| $a_e$ | $E$ | TX apodization |
| $\omega_k=2\pi f_k$, $k_k=\omega_k/c$ | $K$ | angular frequency and wavenumber |

The core tensor in every simulator is the **point-to-sub-element distance**
$r_{p,e,s} = \lVert \mathbf{x}_p - \mathbf{c}_e - \mathbf{o}_{e,s}\rVert$, stored as a dense
`[P, E, S]` array.

---

## 4. How each algorithm uses element positions

### 4.1 `txdelay`: TX delay laws for 1-D arrays (`src/pymust/txdelay.py:13`)

Element positions come from `getElementPositions()` (`txdelay.py:173`), and the result is a $T\times E$
delay matrix. Every law ends with $\tau \leftarrow \tau - \min_e \tau$ (`txdelay.py:218`), and the
result is stored in `param.TXdelay` (`:220`).

| Law | Lines | Formula per element $e$ |
|---|---|---|
| Plane wave, linear | 175-179 | $\tau_e = x_e\sin(\text{tilt})/c$ |
| Plane wave, convex | 180-194 | distance from $(x_e,z_e)$ to the line tangent to the arc at angle tilt: $\tau_e=-d_e/c$ |
| Focused / diverging (`Origo`) | 196-204 | $\tau_e = \mp\lVert(x_e-x_0,\ z_e-z_0)\rVert/c$; sign from `sign(z0)` (linear) or inside/outside the arc (convex) |
| Circular wave (linear only) | 206-216 | virtual source from `(tilt, width)` via `angles2origo` (`:223`), then as `Origo` |

`tilt`, `x0` and `z0` may be column vectors, which produces several delay series ($T>1$).

### 4.2 `txdelay3`: TX delay laws for planar arrays (`src/pymust/txdelay3.py:15`)

$x_e, y_e$ come from `param.elements` (`:177-181`), with $z_e = 0$.

| Law | Lines | Formula |
|---|---|---|
| Plane wave | 189-195 | $\tau_e = (x_e\sin\text{tilt}_y - y_e\sin\text{tilt}_x)/c$ |
| Diverging wave | 197-243 | requires a plaid grid (`:208-212`). The array footprint $l=\max x_e-\min x_e+w$, $b=\max y_e-\min y_e+h$ sets the virtual-source distance for the requested solid angle (`solidAngle`, `:267`), then focuses as `Origo`. |
| Focus point | 245-251 | $d_e=\lVert(x_e-x_0,\ y_e-y_0,\ -z_0)\rVert$, $\tau_e=-\operatorname{sign}(z_0)\,d_e/c$ |
| Focus line | 252-256 | $d_e$ = distance from $(x_e,y_e,0)$ to the line through the two points, via a cross product |

### 4.3 `pfield`: 1-D array radiation pattern, 2-D or 3-D space (`src/pymust/pfield.py:32`)

**Geometry tensors** (all `[P, E, M]`):

1. **Element centres** (`:486`): `getElementPositions()` gives $\mathbf{c}_e=(x_e, 0, z_e)$.
2. **Sub-element splitting** (`:414-415`, `:494-497`): each element is split into
   $M=\lceil w/\lambda_{\min}\rceil$ segments, laid along the element surface (tangent to the arc
   for convex arrays):
   $$u_s = -\tfrac{w}{2}+\tfrac{w}{M}(s+\tfrac12),\qquad \mathbf{o}_{e,s} = u_s\,(\cos\theta_e,\ 0,\ -\sin\theta_e)$$
3. **Distances** (`:511-521`):
   $$\Delta x = x_p - x_e - o^x_{e,s},\quad d^2 = \Delta x^2 + (z_p - z_e - o^z_{e,s})^2,\quad r = \sqrt{d^2 + y_p^2},\quad r\leftarrow\max(r,\lambda_c/2)$$
   The element always sits at $y=0$. $y_p$ only enters through $r$ and the MGBM.
4. **Angle to the element normal** (`:524-525`), measured in the $x$–$z$ plane:
   $$\Theta_{p,e,s} = \arcsin(\Delta x/\sqrt{d^2}) - \theta_e$$
5. **Out-of-field mask** (`:500-502`): $z_p<0$, or inside the convex probe's circle
   ($x_p^2+(z_p+h)^2\le R^2$).

**Per-element frequency-domain kernel** (`:633-701`):

$$\mathcal{E}^{(0)}_{p,e,s} = e^{(-\alpha_0 + i k_0) r}\cdot \underbrace{O(\Theta)}_{\text{obliquity}}\cdot \underbrace{\operatorname{sinc}\!\big(k_c \tfrac{w}{2M}\sin\Theta\big)}_{\text{segment directivity}} \cdot r^{-\gamma}$$

- Obliquity (`:636-645`): $O(\Theta)=\cos\Theta$ for a soft baffle, $\cos\Theta/(\cos\Theta+\alpha_b)$
  for a baffle impedance ratio $\alpha_b$, or 1 for a rigid baffle.
- Spreading exponent: $\gamma=\tfrac12$ in 2-D (cylindrical, `:671`) and $\gamma=1$ in 3-D (spherical, `:668`).
- With `FullFrequencyDirectivity`, the directivity uses $k_k$ inside the loop (`:787-789`).
- The kernel is advanced in frequency by a recursive product, $\mathcal{E}^{(k)}=\mathcal{E}^{(k-1)}\odot e^{(-\Delta\alpha+i\Delta k)r}$
  (`:663`, `:779`).

**Elevation (3-D space only, `y` given)** (`:666-669`, `:707-719`, `:757-769`):

- With the mean distance $\bar r_{p,e}=\operatorname{mean}_s r_{p,e,s}$, define
  $\alpha=\tfrac{i}{2}(1/R_f-1/\bar r)$, $\gamma=\tfrac{i y^2}{2\bar r}$ and $\beta^2=-(y/\bar r)^2$.
- The elevation factor is then
  $$\text{MGBM}_{p,e}(k)=\sum_{n=1}^{4}A_n\sqrt{\tfrac{\pi}{k\alpha+B_n/H^2}}\exp\!\Big(\tfrac{k^2\beta^2/4+k\gamma\,(k\alpha+B_n/H^2)}{k\alpha+B_n/H^2}\Big)$$
  with coefficients from `MGBMcoeff(4)` (`:897`), $H$ = `height` and $R_f$ = `focus`.
- It is recomputed only at $\max(3, K/20)$ frequencies (`:718-719`).

**Frequency loop: the matrix form of the array sum** (numpy loop `:752-855`, JAX branch `:730-751`; JAX kernel in
`src/pymust/jax_backend.py:22-87`):

$$\mathbf{R}_k = \operatorname{mean}_s\big(\mathcal{E}^{(k)}\big)\ [\odot\ \text{MGBM}(k)] \quad\in\mathbb{C}^{P\times E}\qquad(\text{RPmono},\ :796\text{-}813)$$
$$\mathbf{w}_k = \Big(\sum_{t=1}^{T} e^{i\omega_k\tau_{t,e}}\Big)_e \odot \mathbf{a}\quad\in\mathbb{C}^{E}\qquad(\text{DELAPOD},\ :820)$$
$$\mathbf{p}_k = S_{\text{pulse}}(\omega_k)\,S_{\text{probe}}(\omega_k)\ \mathbf{R}_k\,\mathbf{w}_k\quad\in\mathbb{C}^{P},\qquad \mathbf{p}_k[\text{isOUT}]=0\qquad(:823\text{-}830)$$
$$\text{RMS pressure: } \mathrm{RP}_p=\sqrt{\Delta f\,[w]\sum_k |p_{k,p}|^2}\qquad(:850,\ :877\text{-}891)$$

**Where element positions enter:**

- $\mathbf{R}_k$ holds every element's one-way propagation and directivity to every point. Element
  position and orientation enter only through $r_{p,e,s}$ and $\Theta_{p,e,s}$.
- $\mathbf{w}_k$ holds only the electronic delays and apodization, so steering and focusing are not
  geometric inside `pfield`.

### 4.4 `simus`: RF simulation with 1-D arrays (`src/pymust/simus.py`)

- **Element positions** (`:347-353`): `getElementPositions()` gives the maximum element–scatterer
  distance $\max_{p,e}\lVert(x_p-x_e, z_p-z_e)\rVert$, plus the pulse length. This sets the
  frequency step $\Delta f$ that avoids time aliasing (`:357`).
- **Spectra:** `pfield` is then called in SIMUS mode (`:385`, or in a pool `:366-381`). In that mode
  the same $\mathbf{R}_k$ matrix is reused for reception, by reciprocity (`pfield.py:840-848`):

$$\mathbf{s}_k = S_{\text{probe}}(\omega_k)\ \big[(\mathbf{p}_k\odot \mathbf{RC})^{\!\top}\mathbf{R}_k\big]\odot e^{i\omega_k\boldsymbol{\tau}^{RX}}\quad\in\mathbb{C}^{E}$$

- $\mathbf{p}_k\in\mathbb{C}^{P}$ is the transmitted field at the scatterers.
- $\mathbf{R}_k^\top$ maps the scatterer echoes back onto each element.
- $\boldsymbol{\tau}^{RX}$ = `param.RXdelay` (`simus.py:310-315`).
- **Time domain:** $\mathrm{RF} = \operatorname{irfft}(\overline{\mathbf{S}})$ over $k$ (`simus.py:390`), giving one column per element.

### 4.5 `mkmovie`: wave-propagation movie, 1-D arrays (`src/pymust/mkmovie.py`)

- **Image grid:** the movie grid is a centred rectangle whose size comes from `param.movie`
  (`:290-295`).
- **Array extent:** only the aperture length $L=p(E-1)$ (`:249-250`) is used, to bound the travel
  distance and $\Delta f$ (`:298-299`). This assumes a uniform linear array, even for convex probes.
- **Field:** `pfield` runs in MKMOVIE mode. The scatterer coordinates are appended to the points
  (`pfield.py:471-474`), and a dense scatterer→pixel propagator is built (`pfield.py:683-690`):

$$\mathbf{G}^{RC}_{q,j} = e^{(-\alpha+ik)\,\lVert\mathbf{x}_q-\mathbf{x}^{\text{scat}}_j\rVert},\qquad \mathbf{F}_k = \mathbf{p}_k[\text{grid}] + \mathbf{G}^{RC}_k\big(\mathbf{p}_k[\text{scat}]\odot\mathbf{RC}\big)$$

The element positions enter only through $\mathbf{R}_k$, as in §4.3.

### 4.6 `pfield3`: planar 2-D array radiation pattern, 3-D space (`src/pymust/pfield3.py:28`)

**Geometry** (all `[P, E, M·N]`):

1. **Element centres** (`:264-270`): $\mathbf{c}_e=(x_e, y_e, 0)$ from `param.elements`.
2. **Sub-element splitting** (`:386-394`, `:455-461`): a rectangular grid of $M\times N$ sub-elements,
   with $M=\lceil w/\lambda_{\min}\rceil$ and $N=\lceil H/\lambda_{\min}\rceil$ (or
   `options.ElementSplitting=[M, N]`):
   $$\mathbf{o}_{m,n} = \Big(-\tfrac{w}{2}+\tfrac{w}{M}(m+\tfrac12),\ -\tfrac{H}{2}+\tfrac{H}{N}(n+\tfrac12),\ 0\Big)$$
   The offsets are **identical for all elements**: there is no rotation, and every element faces $+z$.
3. **Local spherical angles** (`:477-492`), with $\Delta x = x_p - x_e - o^x$,
   $\Delta y = y_p - y_e - o^y$, $\rho=\sqrt{\Delta x^2+\Delta y^2}$ and $r=\sqrt{\rho^2+z_p^2}$:
   $$\cos T = z_p/r,\quad \sin T=\rho/r,\quad \cos\Phi=\Delta x/\rho,\quad \sin\Phi=\Delta y/\rho$$
   The polar axis is the global $z$ axis. Using $z_p$ rather than $z_p - z_e$ hard-codes $z_e=0$.
4. **Out-of-field mask** (`:465`): $z_p<0$.

**Kernel** (`:572-617`):

$$\mathcal{E}^{(0)} = e^{(-\alpha_0+ik_0)r}\cdot O(\cos T)\cdot\operatorname{sinc}\!\big(k_c\tfrac{w}{2M}\cos\Phi\sin T\big)\operatorname{sinc}\!\big(k_c\tfrac{H}{2N}\sin\Phi\sin T\big)\cdot r^{-1}$$

- Obliquity uses $\cos T$ (`:577-581`).
- The two sinc factors are the separable directivity of a rectangle (`:615-617`; per-frequency
  version at `:657-659`).

**Frequency loop** (JAX branch `:628-641`, numpy loop `:642-707`): identical to §4.3, with no MGBM. Elevation is already resolved by
the $N$ sub-elements along $y$.

$$\mathbf{R}_k=\operatorname{mean}_{m,n}\mathcal{E}^{(k)},\quad \mathbf{p}_k = S_{\text{pulse}}S_{\text{probe}}\,\mathbf{R}_k\mathbf{w}_k$$

### 4.7 `simus3`: RF simulation with planar arrays (`src/pymust/simus3.py`)

- **Element count:** $E$ comes from `param.elements.shape[1]` (`:264`).
- **Maximum distance** (`:303-310`): $\max\lVert(x_p-x_e,\ y_p-y_e,\ z_p)\rVert$. This again assumes $z_e=0$.
- **Spectra:** `pfield3` runs in SIMUS3 mode, with the same reception formula as §4.4
  (`pfield3.py:694-702`), then $\operatorname{irfft}$ (`simus3.py:345`).

### 4.8 `dasmtx`: DAS beamforming matrix, 1-D arrays (`src/pymust/dasmtx.py:8`)

The output is a sparse matrix $\mathbf{M}\in\mathbb{C}^{P\times(n_l\cdot n_c)}$ (COO,
`:636-645`), applied as $\mathbf{b}=\mathbf{M}\,\operatorname{vec}_F(\text{SIG})$ with
`utils.applyDasMTX` (`src/pymust/utils.py:282`). Here $n_l$ is the number of fast-time samples and
$n_c=E$ is the number of channels.

**Geometry:**

1. **Element centres and normal angles** (`:392-399`): $(x_e, z_e, \theta_e)$ from `getElementPositions()`,
   or $(x_e, z_e)$ from `param.elements` (`:348-367`) with $\theta_e=0$, i.e. normals along $+z$.
2. **Transmit wavefront** (`:412-430`, `:456-468`). The transmitting elements are those with
   non-NaN delays.
   - Their positions and delays are cubic-interpolated onto $4\,n_{TX}$ points
     $(\tilde x_i,\tilde z_i,\tilde\tau_i)$, a finer sampling of the emitting aperture.
   - The TX distance is then the first arrival of the wavefront:
   $$d^{TX}_p=\min_i\big(c\,\tilde\tau_i + \lVert(x_p-\tilde x_i,\ z_p-\tilde z_i)\rVert\big)$$
   - With `useVirtualSource` (`:470-511`), a virtual transducer is derived from the delays
     (`vxdcr`, `:680`, which uses the numerical derivative `diff2`, `:659`).
3. **Receive distance and time of flight** (`:518-527`):
   $$d^{RX}_{p,e}=\lVert(x_p-x_e,\ z_p-z_e)\rVert,\qquad \tau_{p,e}=\frac{d^{TX}_p+d^{RX}_{p,e}}{c},\qquad t_{p,e}=(\tau_{p,e}-t_{0})\,f_s$$
4. **Receive aperture** (f-number $F$, `:547-567`). $F$ is found automatically from the element
   directivity if not given (`:434-443`).
   - Linear array: $|x_p-x_e|\le z_p/(2F)$.
   - Convex array: $|\arcsin(\Delta x/d^{RX})-\theta_e|\le\arctan(1/2F)$. This uses each element's
     normal $\theta_e$.
   - With a receive angle (vector Doppler): $|x_p-x_e-z_p\tan\phi|\le z_p/(2F\cos\phi)$.
5. **Matrix entries** (`:569-645`): for every valid $(p,e)$ and interpolation tap $j$:
   $$\mathbf{M}\big[p,\ e\,n_l+\lfloor t_{p,e}\rfloor + j\big] = h_j\big(t_{p,e}-\lfloor t_{p,e}\rfloor\big)\cdot\big[e^{i\omega_c\tau_{p,e}}\big]_{\text{I/Q only}}$$
   $h_j$ is the kernel of the selected interpolation method: nearest, linear, quadratic,
   lanczos3, 5points or lanczos5.

Element geometry enters through $d^{RX}$ (receive time of flight), $d^{TX}$ (transmit time of
flight) and the aperture mask.

### 4.9 `dasmtx3`: DAS beamforming matrix, 2-D/3-D arrays (`src/pymust/dasmtx3.py:6`)

1. **Element centres** (`:243-254`): $(x_e, y_e, z_e)$, with $z_e=0$ when only two rows are given.
   This is the only function that accepts element $z$.
2. **Receive distance** (`:345-346`): $d^{RX}_{p,e}=\lVert \mathbf{x}_p-\mathbf{c}_e\rVert$, in full 3-D.
3. **Transmit distance** (`:348-351`): $d^{TX}_p=\min_e(c\,\tau_e + d^{RX}_{p,e})$, at the element
   centres with no aperture interpolation (unlike `dasmtx`).
   - The virtual-source branch (`:352-393`) uses `vxdcr3` (`:523`) and `trigrad` (`:550`).
     `vxdcr3` places each virtual point at distance $cT_e$ behind its element:
     $x_V = x_e - \tfrac{c^2}{2}\partial_x T^2$, $y_V = y_e - \tfrac{c^2}{2}\partial_y T^2$
     (`:539-546`). Each pixel then takes the virtual point whose normal line passes closest to it
     (`:375-391`), and $d^{TX}_p=\lVert\mathbf{x}_p-\mathbf{V}\rVert$ (`:393`).
4. **Aperture** (`:424-427`), with two f-numbers (from `width` and `height`, `:312-325`):
   $$|x_p-x_e|\le\frac{|z_p-z_e|}{2F_x}\ \wedge\ |y_p-y_e|\le\frac{|z_p-z_e|}{2F_y}$$
   This implicitly assumes every element normal points along $+z$.
5. **Matrix entries** (`:398-510`): built exactly as in §4.8.

Convex matrix arrays are explicitly rejected (`:221`), and so is `RXangle` (`:227`).

### 4.10 `impolgrid`: imaging grid derived from the array (`src/pymust/impolgrid.py`)

Builds a polar imaging grid from the array extent (`:93-139`). It uses `pitch`, `Nelements` and
`radius`, not the element coordinates:

- Linear: $L=p(E-1)$, origin $z_0=0$.
- Convex: chord $L$, apothem $d$, origin $z_0=-d$ (the arc centre).

### 4.11 Functions that do not use element positions

These functions work on channel data or beamformed images. Array geometry reaches them only through
the DAS matrix of §4.8–4.9.

- `rf2iq`
- `bmode`
- `tgc`
- `iq2doppler`
- `wfilt`
- `sptrack`
- `smoothn`
- `genscat`
- `getpulse` (spectrum only)

### 4.12 JAX backend

`src/pymust/jax_backend.py` reproduces the frequency loops of §4.3 to §4.7 with the same matrices:

- **TX/RX weights on the host:** `compute_frequency_weights` (`:157-164`) builds the
  $\mathbf{w}_k$ and $\mathbf{s}_k$ weight matrices for all $k$ at once.
- **Frequency loop:** `sum_over_spectrum` (`:22-87`) carries $\mathcal{E}^{(k)}$ through
  `lax.scan`. It computes $\mathbf{R}_k$ (`:61-65`) and $\mathbf{R}_k\mathbf{w}_k$ (`:67`).
- **Inputs from the numpy code:** the geometry tensors are passed in already built. They are
  $r$, the directivity arguments ($\tfrac{w}{2M}\sin\Theta$ for `pfield`;
  $\tfrac{w}{2M}\cos\Phi\sin T$ and $\tfrac{H}{2N}\sin\Phi\sin T$ for `pfield3`), and the MGBM
  arguments.
- **No new geometry:** the backend adds no geometric assumptions of its own.

---

## 5. Defects and limits found while tracing

### 5.1 Bugs (all fixed)

All six are fixed, and each has a regression test in `tests/test_array_geometry.py`.

| # | Location (before the fix) | Problem | Effect | Fix |
|---|---|---|---|---|
| 1 | `src/pymust/dasmtx.py:347-367` vs `:393` | `param.elements` is parsed into `xe, ze`, then unconditionally overwritten by `getElementPositions()`. `isPARAMelements` is never used. | Custom 2-D element positions are silently ignored: a matrix built with shifted `param.elements` was identical to the default one. | Positions come from `param.elements` when it is given, with normals along $+z$ ($\theta_e=0$). The TX-aperture interpolation keeps the given $z_e$. |
| 2 | `src/pymust/txdelay3.py:191` | Calls the 1-D `getElementPositions()` and never uses the result. | Every `txdelay3` law failed on a plain matrix-array `Param` (`TypeError: np.isinf(None)`) unless `radius`, `pitch` and `Nelements` were also set. | Call removed. |
| 3 | `src/pymust/utils.py:94-97` | `getElementPositions()` does not default `radius`. | Failed on a hand-built `Param` without `radius`. | Missing `radius` means linear array (`inf`). |
| 4 | `src/pymust/dasmtx3.py:525` | `xV = xe - c**22*dT2dx/2` (should be `c**2`). | Virtual points at ~1e60 m, so `useVirtualSource` produced garbage. | `c**2`, with the derivation in a comment. Virtual points now lie within ~1 µm of the true wavefront sphere for a known diverging wave. |
| 5 | `src/pymust/dasmtx3.py:360` | Single-element virtual source uses `hypot(xe-x, ze-z)`, which ignores $y$. | Wrong TX distance for single-element transmits with 2-D arrays. | Full 3-D distance. |
| 6 | `src/pymust/dasmtx3.py:375` | The pixel-to-normal-line distance `Dn` paired the $x$-slope with $y$ (and the $y$-slope with $z$) and divided by $1+z_x^2+z_y^2$ without a square root. | Even with #4 fixed, the wrong virtual point was picked: TX distance errors up to 1.8 mm. | Geometric distance to the normal line, $\lVert\mathbf{P}-\mathbf{V}\rVert^2-((\mathbf{P}-\mathbf{V})\cdot\hat{\mathbf{n}})^2$, documented in a comment. TX distance error ~5 µm, the same as the element-based path. |

### 5.2 Geometric limits (as traced; see §6.5 for what remains)

These were the limits at commit `c4399e6`:
- **Elements fixed to $z_e=0$** in `pfield3`, `simus3` and `txdelay3`.
- **Array extent from `pitch`/`Nelements`** in `mkmovie`, `impolgrid` and `txdelay`.
- **No per-element orientation or size.**

§6 removes the first two.

### 5.3 What an arbitrary array definition had to touch

§6 implements this for positions. Per-element orientation, a normal $\mathbf{n}_e$ with in-plane axes
$\mathbf{u}_e,\mathbf{v}_e$, remains future work (§6.5).

---

## 6. Arbitrary element positions (current code)

Every algorithm now reads the element positions from one of two accessors and uses the full position
vector. No algorithm recomputes positions from `pitch`, drops or zeroes a coordinate, or writes back
to `param.elements`. Pre-made probes (`getparam`, or `pitch`/`Nelements`/`radius`) are translated
into the same position vector by the accessors.

The acceptance test is **translation invariance**: shift every element and every field point,
scatterer, focus or pixel by the same vector, and every output stays unchanged. This is checked for
every function by `tests/test_arbitrary_elements.py`.

### 6.1 Defining an array

| `param.elements` | 2-D functions (`txdelay`, `pfield`, `simus`, `mkmovie`, `dasmtx`, `impolgrid`) | 3-D functions (`txdelay3`, `pfield3`, `simus3`, `dasmtx3`) |
|---|---|---|
| absent | uniform linear/convex array from `pitch`, `Nelements`, `radius` | uniform linear array as $(x_e, 0, 0)$; convex is rejected |
| 1 row | $x$, with $z=0$ | $x$, with $y=z=0$ |
| 2 rows | $(x, z)$ | $(x, y)$, with $z=0$ |
| 3 rows | $(x, y, z)$ with $y=0$ required | $(x, y, z)$ |

- **Orientation:** E×k arrays are also accepted. When `elements` is given, `radius` must be `inf` or
  absent.
- **`Nelements`:** if set, it must match the number of elements.
- **`pitch`:** no longer required; `width` is required instead.

```python
import numpy as np
import pymust
from pymust.utils import Param

# Arbitrary 2-D points: 64 elements on a shallow parabola in the x-z plane
x = (np.arange(64) - 31.5) * 300e-6
param = Param(fc=5e6, bandwidth=70, width=270e-6, elements=np.stack([x, 2e-3 * (x / x.max())**2]))
delays = pymust.txdelay(0, 30e-3, param)          # focused; pfield/simus/dasmtx use the same positions

# Arbitrary 3-D points: 32x32 matrix on a spherical cap of radius 20 mm
g = (np.arange(32) - 15.5) * 300e-6
xe, ye = (a.ravel() for a in np.meshgrid(g, g))
ze = 20e-3 - np.sqrt(20e-3**2 - xe**2 - ye**2)
param3 = Param(fc=3e6, bandwidth=70, width=270e-6, height=270e-6, elements=np.stack([xe, ye, ze]))
delays3 = pymust.txdelay3(param3, 0.1, 0.2)       # plane wave tilted about both x and y
fig = pymust.utils.plot_transducer_array(param3)   # interactive 3-D boxes in Jupyter
```

### 6.2 Accessors (`src/pymust/utils.py`)

- **`Param.read_elements()`:** returns `param.elements` as a (rows, E) copy and validates it.
- **`Param.getElementPositions()`:** used by the 2-D functions; returns `xe, ze, THe, h` (1×E).
  - With `elements` given, positions are taken as given and $\theta_e = 0$, so normals point along $+z$.
  - Otherwise it applies the uniform linear/convex formulas of §2.3.
- **`Param.get_element_coordinates()`:** used by the 3-D functions; returns `xe, ye, ze` (length E).

### 6.3 What changed in each function

| Function | Change |
|---|---|
| `dasmtx` | Local `elements` parsing removed. The TX aperture is interpolated with the real $z_e$. The receive aperture uses $z_p-z_e$, so $\lvert x_p-x_e\rvert\le (z_p-z_e)/2F$, including the RX-angle option. |
| `pfield` | `pitch` is only needed without `elements`. Out-of-field means $z_p<\min_e z_e$. |
| `simus` | The element count comes from the positions. |
| `txdelay` | Plane wave: $\tau_e=(x_e\sin\text{tilt}+z_e\cos\text{tilt})/c$. Circular wave: aperture $L=\operatorname{ptp}(x_e)$, origin relative to the array centre, distance $\lVert(x_e-x_0,\ z_e-z_0)\rVert$. Focused: the side is decided relative to the array centre. |
| `mkmovie`, `impolgrid` | Array width $L=\operatorname{ptp}(x_e)$. `impolgrid` centres its grid on the array, and its pitch defaults to the mean element spacing. |
| `pfield3` | $r$ and $\cos T$ use $z_p-z_e$. Out-of-field means $z_p<\min_e z_e$. Elements no longer radiate backwards (obliquity clamped where $\cos T\le0$, as in `pfield`). |
| `simus3` | Positions come from the accessor, and the maximum distance uses $z_p-z_e$. |
| `txdelay3` | Plane wave: $\tau_e=\hat k\cdot\mathbf c_e/c$ with $\hat k=(\sin t_y\cos t_x,\ -\sin t_x,\ \cos t_y\cos t_x)$, the rotation used by the diverging option. **This changes the delays when both tilts are non-zero:** the earlier x-term lacked $\cos t_x$. Focus point and focus line use $z_e$. The diverging-wave virtual source is placed relative to the array centre. |
| `dasmtx3` | Positions come from the accessor, so `param.elements` is no longer transposed in place. `vxdcr3` handles non-planar arrays (§6.4). `trigrad` triangulates relative, rounded positions, so its gradients do not depend on where the array is. |

`examples/txdelay3_tilted_delays.ipynb` plots the `txdelay3` delays as 3-D bars:
- tilts in both directions;
- the change from the earlier formula, which is up to 28 ns for a 12×12, 300 µm array at 0.3/0.3 rad;
- a curved array, whose $z_e$ now shapes the delays.

### 6.4 Virtual source on a non-planar array (`vxdcr3`)

**Envelope condition.** The virtual point $V_e$ lies at distance $cT_e$ from element $e$. It is where
that element's sphere touches the envelope of all the spheres. Differentiating
$(x_e-x_V)^2+(y_e-y_V)^2+(z_e-z_V)^2=c^2T^2$ along the array surface $z_e(x, y)$, at fixed $V$, gives

$$a+h\,g_x=p_x,\qquad b+h\,g_y=p_y$$

where:
- $a = x_e-x_V$, $b = y_e-y_V$ and $h = z_e-z_V \ge 0$;
- $g = \nabla z_e$ is the slope of the array surface;
- $p = \tfrac{c^2}{2}\nabla T^2$.

This is the same condition the 2-D `vxdcr` solves with `fsolve`.

**Closed form.** Substituting $a$ and $b$ into the sphere equation gives a quadratic in $h$, whose
non-negative root is

$$h=\frac{p\cdot g+\sqrt{(p\cdot g)^2-(1+|g|^2)(|p|^2-c^2T^2)}}{1+|g|^2}$$

It reduces to the planar formula when $g=0$.

**Accuracy.** For a 16×16 array on a 20 mm spherical cap, with a known diverging source, the virtual
points lie on the true wavefront sphere to within 5 µm.

**Limit.** The elements must form a height field over $(x, y)$.

### 6.5 Verification pending: tilt equations and delay profiles

The delay laws changed in this work are tested for internal consistency only. The tests check
translation invariance, the analytic formulas themselves, and a known diverging source for `vxdcr3`.
They have **not yet been verified against an independent reference**.

**What still needs checking:**
- **`txdelay3` plane wave:** the rotation convention, TILTx about x and then TILTy about y, which
  gives $\hat k=(\sin t_y\cos t_x,\ -\sin t_x,\ \cos t_y\cos t_x)$.
  - It must be confirmed against MATLAB MUST, whose plane-wave formula
    $(x_e\sin t_y - y_e\sin t_x)/c$ this replaces.
  - It must also be confirmed against the actual steering direction of the `pfield3` main lobe,
    especially when both tilts are non-zero.
- **`txdelay` plane wave with element depths:** the $z_e\cos(\text{tilt})$ term for non-flat 2-D
  arrays. Check that the simulated wavefront (`pfield`) is planar.
- **Origins relative to the array centre:**
  - `txdelay` circular wave: its origin is now placed relative to the array's bounding-box centre.
  - `txdelay3` diverging wave: its virtual source is likewise relative to that centre.
  - Focused laws in both functions: the side is decided relative to the same centre.

  All three are exact for centred uniform arrays, but need checking for off-centre and irregular arrays.
- **Delay profiles:** check them, for example with the bar plots of
  `examples/txdelay3_tilted_delays.ipynb`, against beam directions and focal positions measured on
  simulated fields.

Until this is done, treat delays from these laws with both tilts non-zero, or on non-centred or
non-planar arrays, as provisional.

### 6.6 Remaining limits

- **Orientation:** custom elements face $+z$; only the uniform convex probes carry a tilt $\theta_e$.
  Per-element orientation would need a normal $\mathbf n_e$ and in-plane axes
  $\mathbf u_e,\mathbf v_e$, used in the directivity, the obliquity, the sub-element offsets and the
  apertures.
- **Uniform element size:** a single scalar `width` and `height` for the whole array.
- **Plaid-grid requirement:** `txdelay3` "Diverging Wave" still needs the elements on a plaid
  $(x, y)$ grid.
- **Virtual source:** `vxdcr3` needs a height-field array.
- **Off-plane elements in 2-D functions:** `pfield` and `dasmtx` reject them ($y_e \ne 0$); use the
  3-D functions instead.
