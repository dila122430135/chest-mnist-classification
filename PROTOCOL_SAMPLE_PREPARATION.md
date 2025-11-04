# PROTOCOL: Sample Preparation for AS7265X Glucose Detection

## 📋 TABLE OF CONTENTS
1. [Preparation of Synthetic Urine Samples](#preparation-of-synthetic-urine-samples)
2. [Reagent-Free Method (Direct Spectroscopy)](#method-1-reagent-free)
3. [GOD-POD Enzymatic Method](#method-2-god-pod-enzymatic)
4. [Quality Control](#quality-control)
5. [Troubleshooting](#troubleshooting)

---

## PREPARATION OF SYNTHETIC URINE SAMPLES 🧪

### Overview
Protokol ini untuk membuat 5 sampel urin sintetis dengan konsentrasi glukosa: **0, 50, 100, 250, 500 mg/dL**. Sampel ini digunakan untuk kalibrasi sensor, validasi model ML, atau pelatihan dataset.

---

### MATERIALS REQUIRED

#### A. Base Urine Matrix (Urin Sintetis) - 500 mL

**Bahan Utama:**
```
┌────────────────────────────────┬──────────┬──────────────┐
│ Bahan                          │ Jumlah   │ Fungsi       │
├────────────────────────────────┼──────────┼──────────────┤
│ Urea (CH₄N₂O)                  │ 12.5 g   │ N-metabolit  │
│ Sodium Chloride (NaCl)         │ 4.6 g    │ Elektrolit   │
│ Potassium Chloride (KCl)       │ 1.1 g    │ Elektrolit   │
│ Creatinine (C₄H₇N₃O)           │ 0.5 g    │ Biomarker    │
│ Ammonium Chloride (NH₄Cl)      │ 1.0 g    │ pH buffer    │
│ Sodium Phosphate (Na₂HPO₄)     │ 1.4 g    │ pH buffer    │
│ Potassium Phosphate (KH₂PO₄)   │ 0.7 g    │ pH buffer    │
│ Calcium Chloride (CaCl₂)       │ 0.3 g    │ Mineral      │
│ Magnesium Sulfate (MgSO₄)      │ 0.3 g    │ Mineral      │
│ Sodium Sulfate (Na₂SO₄)        │ 1.8 g    │ Sulfat ion   │
│ Distilled Water                │ 500 mL   │ Pelarut      │
└────────────────────────────────┴──────────┴──────────────┘

Optional (untuk warna/bau realistis):
- Urobilin atau Riboflavin: 5 mg (warna kuning)
- Uric acid: 0.15 g (komponen minor)
```

**Cara Pembuatan Base Urine:**
```
1. Siapkan beaker glass 1000 mL
2. Tambahkan 400 mL distilled water
3. Larutkan bahan dalam urutan:
   a. Urea → aduk hingga larut
   b. NaCl + KCl → aduk 2 menit
   c. Creatinine → aduk hingga larut sempurna
   d. NH₄Cl + Na₂HPO₄ + KH₂PO₄ → aduk 5 menit
   e. CaCl₂ + MgSO₄ + Na₂SO₄ → aduk 3 menit
   f. (Optional) Riboflavin untuk warna
4. Cek pH dengan pH meter → target pH 6.0 ± 0.2
   - Jika pH < 5.8: tambah Na₂HPO₄ (0.1 g)
   - Jika pH > 6.2: tambah NH₄Cl (0.1 g)
5. Tambahkan distilled water hingga volume 500 mL
6. Aduk dengan magnetic stirrer 10 menit
7. Saring dengan filter paper 0.45 μm
8. Transfer ke botol amber (lindungi dari cahaya)
9. Label: "Synthetic Urine Base (0 mg/dL glucose)"
10. Simpan pada 4°C (tahan 1 bulan)
```

#### B. Glucose Stock Solution (5000 mg/dL) - 100 mL

**Bahan:**
```
- D-Glucose anhydrous (dried): 5.000 g
- Sodium benzoate: 0.100 g (pengawet)
- Distilled water: 100 mL
```

**Cara Pembuatan:**
```
1. Keringkan glucose powder di oven 60°C selama 2 jam
2. Dinginkan dalam desikator 30 menit
3. Timbang tepat 5.000 g (timbangan analitik)
4. Larutkan dalam 80 mL distilled water (aduk perlahan)
5. Tambahkan sodium benzoate
6. Transfer ke labu ukur 100 mL
7. Tambahkan air hingga tanda batas
8. Kocok hingga homogen
9. Simpan di botol amber pada 4°C
10. Label: "Glucose Stock 5000 mg/dL", tanggal, exp: +3 bulan
```

---

### COMPOSITION OF 5 SAMPLES (100 mL Each)

#### Sample 1: **0 mg/dL** (Negative Control)
```
Komposisi:
- Synthetic Urine Base: 100 mL
- Glucose Stock: 0 mL
- Distilled Water: 0 mL

Total Volume: 100 mL
Konsentrasi Glukosa: 0 mg/dL

Cara Pembuatan:
1. Pipet 100 mL base urine ke dalam botol 100 mL
2. Label: "Sample 1 - 0 mg/dL"
3. Simpan pada 4°C
```

#### Sample 2: **50 mg/dL** (Low - Normal)
```
Komposisi:
- Synthetic Urine Base: 99 mL
- Glucose Stock (5000 mg/dL): 1.0 mL
- Distilled Water: 0 mL

Total Volume: 100 mL
Konsentrasi Glukosa: 50 mg/dL

Perhitungan:
C₁V₁ = C₂V₂
5000 × V₁ = 50 × 100
V₁ = 1.0 mL

Cara Pembuatan:
1. Pipet 99 mL base urine ke labu ukur 100 mL
2. Tambahkan 1.0 mL glucose stock (pipet akurat!)
3. Kocok perlahan 20× (hindari gelembung)
4. Transfer ke botol amber
5. Label: "Sample 2 - 50 mg/dL"
6. Simpan pada 4°C
```

#### Sample 3: **100 mg/dL** (Normal - Borderline)
```
Komposisi:
- Synthetic Urine Base: 98 mL
- Glucose Stock (5000 mg/dL): 2.0 mL
- Distilled Water: 0 mL

Total Volume: 100 mL
Konsentrasi Glukosa: 100 mg/dL

Perhitungan:
C₁V₁ = C₂V₂
5000 × V₁ = 100 × 100
V₁ = 2.0 mL

Cara Pembuatan:
1. Pipet 98 mL base urine ke labu ukur 100 mL
2. Tambahkan 2.0 mL glucose stock
3. Kocok perlahan 20×
4. Transfer ke botol amber
5. Label: "Sample 3 - 100 mg/dL"
6. Simpan pada 4°C
```

#### Sample 4: **250 mg/dL** (High - Diabetic Range)
```
Komposisi:
- Synthetic Urine Base: 95 mL
- Glucose Stock (5000 mg/dL): 5.0 mL
- Distilled Water: 0 mL

Total Volume: 100 mL
Konsentrasi Glukosa: 250 mg/dL

Perhitungan:
C₁V₁ = C₂V₂
5000 × V₁ = 250 × 100
V₁ = 5.0 mL

Cara Pembuatan:
1. Pipet 95 mL base urine ke labu ukur 100 mL
2. Tambahkan 5.0 mL glucose stock
3. Kocok perlahan 20×
4. Transfer ke botol amber
5. Label: "Sample 4 - 250 mg/dL"
6. Simpan pada 4°C
```

#### Sample 5: **500 mg/dL** (Very High - Severe Diabetes)
```
Komposisi:
- Synthetic Urine Base: 90 mL
- Glucose Stock (5000 mg/dL): 10.0 mL
- Distilled Water: 0 mL

Total Volume: 100 mL
Konsentrasi Glukosa: 500 mg/dL

Perhitungan:
C₁V₁ = C₂V₂
5000 × V₁ = 500 × 100
V₁ = 10.0 mL

Cara Pembuatan:
1. Pipet 90 mL base urine ke labu ukur 100 mL
2. Tambahkan 10.0 mL glucose stock
3. Kocok perlahan 20×
4. Transfer ke botol amber
5. Label: "Sample 5 - 500 mg/dL"
6. Simpan pada 4°C
```

---

### SUMMARY TABLE: Complete Composition

```
┌─────────┬──────────┬─────────────┬───────────┬────────────┐
│ Sample  │ Glucose  │ Base Urine  │ Glucose   │ DI Water   │
│   #     │ (mg/dL)  │   (mL)      │ Stock(mL) │   (mL)     │
├─────────┼──────────┼─────────────┼───────────┼────────────┤
│    1    │    0     │   100.0     │    0.0    │    0.0     │
│    2    │   50     │    99.0     │    1.0    │    0.0     │
│    3    │  100     │    98.0     │    2.0    │    0.0     │
│    4    │  250     │    95.0     │    5.0    │    0.0     │
│    5    │  500     │    90.0     │   10.0    │    0.0     │
└─────────┴──────────┴─────────────┴───────────┴────────────┘

Total Materials Needed:
- Synthetic Urine Base: 482 mL
- Glucose Stock 5000 mg/dL: 18 mL
- Total Volume: 500 mL (5 samples × 100 mL)
```

---

### EQUIPMENT CHECKLIST

```
[ ] Analytical balance (0.001 g precision)
[ ] pH meter
[ ] Magnetic stirrer + stir bar
[ ] Beaker glass: 1000 mL (×1), 250 mL (×2)
[ ] Volumetric flask: 100 mL (×6)
[ ] Micropipettes: 1000 μL, 5000 μL
[ ] Filter paper 0.45 μm + funnel
[ ] Amber bottles: 100 mL (×6), 500 mL (×1)
[ ] Graduated cylinder: 100 mL
[ ] Disposable gloves, safety goggles
[ ] Labels and permanent marker
[ ] Parafilm for sealing
[ ] Refrigerator (4°C storage)
```

---

### VALIDATION PROCEDURE

#### Step 1: Verify Glucose Concentration (GOD-POD Method)

**Test each sample immediately after preparation:**

```
1. Run GOD-POD assay (see Method 2 below)
2. Measure absorbance at 510 nm with AS7265X
3. Calculate actual glucose concentration
4. Acceptance criteria:
   ✓ Measured value within ±5% of target
   ✓ Example: Sample 2 (target 50 mg/dL)
     → Accepted range: 47.5 - 52.5 mg/dL

5. If outside range:
   - Check pipetting accuracy
   - Verify glucose stock concentration
   - Remake sample if needed
```

#### Step 2: AS7265X Spectral Measurement

**Measure all 18 channels for each sample:**

```python
import serial
import time
import pandas as pd

# Measure all 5 samples
samples = [0, 50, 100, 250, 500]  # mg/dL
data = []

for i, glucose_level in enumerate(samples, 1):
    print(f"\n=== Sample {i}: {glucose_level} mg/dL ===")
    
    # Take 10 readings
    readings = []
    for j in range(10):
        sensor.takeMeasurements()
        spectrum = {
            'A_410': sensor.getCalibratedA(),
            'B_435': sensor.getCalibratedB(),
            'C_460': sensor.getCalibratedC(),
            'D_485': sensor.getCalibratedD(),
            'E_510': sensor.getCalibratedE(),
            'F_535': sensor.getCalibratedF(),
            # ... all 18 channels
            'S_940': sensor.getCalibratedS()
        }
        readings.append(spectrum)
        time.sleep(1)
    
    # Average 10 readings
    avg_spectrum = pd.DataFrame(readings).mean()
    avg_spectrum['glucose_mgdl'] = glucose_level
    avg_spectrum['sample_id'] = f'Sample_{i}'
    
    data.append(avg_spectrum)

# Save calibration data
df = pd.DataFrame(data)
df.to_csv('AS7265X_calibration_5samples.csv', index=False)
print("\n✓ Calibration data saved!")
```

#### Step 3: Check Linearity

**Verify linear response across concentration range:**

```python
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Plot glucose vs intensity for key wavelength (510 nm)
glucose = [0, 50, 100, 250, 500]
intensity_510 = df['E_510'].values

plt.figure(figsize=(8, 6))
plt.scatter(glucose, intensity_510, s=100, alpha=0.7)
plt.plot(glucose, intensity_510, 'r--')
plt.xlabel('Glucose (mg/dL)', fontsize=12)
plt.ylabel('Intensity @ 510 nm', fontsize=12)
plt.title('Linearity Check: AS7265X Sensor')
plt.grid(True, alpha=0.3)

# Calculate R²
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(glucose, intensity_510)
r_squared = r_value**2

plt.text(250, max(intensity_510)*0.9, f'R² = {r_squared:.4f}', fontsize=14)
plt.savefig('calibration_linearity.png', dpi=300)
print(f"Linearity R² = {r_squared:.4f}")

# Acceptance: R² > 0.995 for good calibration
if r_squared > 0.995:
    print("✓ PASS: Excellent linearity")
else:
    print("✗ FAIL: Check sample preparation or sensor")
```

---

### STORAGE & STABILITY

```
Storage Conditions:
- Temperature: 4°C (refrigerator)
- Container: Amber glass bottles (protect from light)
- Seal: Tight cap + parafilm
- Location: Away from direct light

Stability:
- Synthetic urine base (0 mg/dL): 30 days at 4°C
- Glucose-spiked samples: 14 days at 4°C
- Glucose stock solution: 90 days at 4°C

Before Use:
1. Warm to room temperature (25°C ± 2°C)
2. Mix gently by inverting 10×
3. Check for precipitation/cloudiness
4. If cloudy → discard and prepare fresh
5. Measure pH (should be 6.0 ± 0.3)
```

---

### COST CALCULATION (Estimate)

```
┌─────────────────────────────┬──────────┬──────────┬──────────┐
│ Bahan                       │ Jumlah   │ Harga/g  │ Total    │
├─────────────────────────────┼──────────┼──────────┼──────────┤
│ Urea                        │  12.5 g  │ Rp 500   │ Rp 6,250 │
│ NaCl                        │   4.6 g  │ Rp 100   │ Rp   460 │
│ KCl                         │   1.1 g  │ Rp 300   │ Rp   330 │
│ Creatinine                  │   0.5 g  │ Rp 8,000 │ Rp 4,000 │
│ D-Glucose                   │   5.0 g  │ Rp 1,000 │ Rp 5,000 │
│ Phosphate buffers (mix)     │   2.1 g  │ Rp 500   │ Rp 1,050 │
│ Other salts (Ca, Mg, NH₄)   │   2.6 g  │ Rp 400   │ Rp 1,040 │
│ Sodium benzoate             │   0.2 g  │ Rp 200   │ Rp    40 │
│ Filter paper 0.45 μm        │  1 pack  │ Rp 5,000 │ Rp 5,000 │
├─────────────────────────────┴──────────┴──────────┼──────────┤
│ TOTAL COST                                        │ Rp 23,170│
└───────────────────────────────────────────────────┴──────────┘

Cost per sample (100 mL): Rp 4,634
Very economical compared to commercial QC materials!
```

---

### TROUBLESHOOTING SAMPLE PREPARATION

**Problem 1: Sample cloudy/precipitate**
```
Cause: Minerals (Ca²⁺, Mg²⁺) precipitation at high pH
Solution:
- Add phosphates BEFORE adding CaCl₂
- Keep pH < 6.5
- Warm solution to 30°C while stirring
- Filter through 0.45 μm if needed
```

**Problem 2: pH drift over time**
```
Cause: Urea hydrolysis → ammonia → pH increase
Solution:
- Add more NH₄Cl (pH stabilizer)
- Store at 4°C (slow down hydrolysis)
- Use within 2 weeks
- Add 0.02% sodium azide (inhibit bacteria)
```

**Problem 3: Glucose concentration decreases**
```
Cause: Bacterial degradation or non-enzymatic glycation
Solution:
- Add sodium benzoate 0.1% (preservative)
- Store at 4°C
- Use sterile technique
- Verify concentration weekly with GOD-POD
- Discard if >10% deviation from target
```

**Problem 4: Color too yellow or brown**
```
Cause: Excessive riboflavin or oxidation
Solution:
- Use only 5 mg riboflavin per 500 mL
- Store in amber bottles (block UV light)
- Add 0.1% ascorbic acid (antioxidant)
- Fresh base urine should be pale yellow
```

---

## METHOD 1: REAGENT-FREE (Direct NIR Spectroscopy) ⭐ RECOMMENDED

### Materials Required
- [ ] Fresh urine sample (2-3 mL)
- [ ] Optical glass cuvette (1 cm path length)
- [ ] Filter paper 0.45 μm (optional)
- [ ] Disposable pipettes
- [ ] AS7265X sensor + Arduino
- [ ] Temperature probe
- [ ] pH meter

### Procedure

#### Step 1: Sample Collection
```
1. Collect fresh urine in sterile container
2. Record collection time
3. Keep at room temperature (avoid refrigeration before measurement)
4. Measure within 2 hours of collection
```

#### Step 2: Sample Preparation (5 minutes)
```
1. Mix urine sample gently (avoid bubbles)
2. Measure temperature: Target 25°C ± 2°C
   - If cold: warm to room temp
   - If warm: cool to room temp
3. Measure pH: Should be 5.5-7.5
   - If pH < 5.5 or > 7.5: Note as interference risk
4. [OPTIONAL] Filter through 0.45 μm filter
   - Only if sample is turbid/cloudy
   - Reduces light scattering artifacts
5. Transfer 2 mL to optical cuvette
```

#### Step 3: AS7265X Measurement (2 minutes)
```
1. Place cuvette in sensor holder
2. Wait 30 seconds for temperature stabilization
3. Take 10 consecutive readings (1 second apart)
4. Average the readings to reduce noise
5. Record all 18 channel intensities (410-940 nm)
6. Clean cuvette with DI water between samples
```

#### Step 4: Data Processing
```python
# Python code for prediction
import joblib
import numpy as np

# Load trained model
model = joblib.load('models/glucose_model_advanced.pkl')
scaler = joblib.load('models/glucose_scaler_advanced.pkl')

# Raw data from sensor (18 channels)
raw_spectrum = np.array([
    ch410, ch435, ch460, ch485, ch510, ch535,
    ch560, ch585, ch610, ch645, ch680, ch705,
    ch730, ch760, ch810, ch860, ch900, ch940
])

# Feature engineering (43 features)
features = engineer_features(raw_spectrum, temperature, ph)

# Normalize
features_scaled = scaler.transform([features])

# Predict
glucose_mgdl = model.predict(features_scaled)[0]

print(f"Glucose Concentration: {glucose_mgdl:.1f} mg/dL")
```

### Expected Results
- **Measurement time:** 2-3 minutes
- **Accuracy:** R² = 96.76% (MAE ± 42 mg/dL)
- **Range:** 0-500 mg/dL
- **Precision:** CV% < 5%

---

## METHOD 2: GOD-POD ENZYMATIC (For Validation)

### Materials Required

#### Reagents
- [ ] GOD-POD working reagent (prepared as below)
- [ ] Glucose standard 100 mg/dL
- [ ] Phosphate buffer pH 7.0
- [ ] Distilled water

#### Equipment
- [ ] Water bath 37°C
- [ ] Micropipettes (10 μL, 1000 μL)
- [ ] Test tubes
- [ ] Timer
- [ ] AS7265X sensor OR spectrophotometer 505 nm

### Reagent Preparation

#### A. Phosphate Buffer (pH 7.0) - 1000 mL
```
Ingredients:
- Na₂HPO₄ (Disodium phosphate): 7.09 g
- KH₂PO₄ (Potassium dihydrogen phosphate): 2.72 g
- Distilled water: 1000 mL

Procedure:
1. Dissolve Na₂HPO₄ in 500 mL warm water
2. Dissolve KH₂PO₄ in 500 mL water (separate beaker)
3. Mix both solutions
4. Check pH with pH meter → adjust to 7.0 ± 0.05
5. Store at 4°C (stable 1 month)
```

#### B. GOD-POD Working Reagent - 100 mL
```
⚠️ SAFETY: Wear gloves, lab coat, goggles. Work in fume hood.

Ingredients:
- Glucose Oxidase (1000 U/mg): 15 mg
- Peroxidase (1000 U/mg): 1 mg
- 4-Aminoantipyrine (4-AAP): 30 mg
- Phenol (⚠️ TOXIC): 110 mg
- Phosphate buffer pH 7.0: 100 mL
- Sodium azide (preservative): 95 mg

Procedure:
1. Add 80 mL phosphate buffer to beaker
2. Add 4-AAP, stir until dissolved (clear solution)
3. Add Phenol slowly (CAUTION: corrosive!)
4. Add Glucose Oxidase powder, stir gently (enzyme fragile)
5. Add Peroxidase powder, stir gently
6. Add sodium azide
7. Adjust to 100 mL with buffer
8. Stir 10 min with magnetic stirrer (low speed)
9. Filter through 0.22 μm filter
10. Aliquot into amber bottles (protect from light)
11. Label: "GOD-POD Reagent", Date, Exp: +6 months
12. Store 2-8°C, DO NOT FREEZE

Quality Check:
- Color: Pale yellow/clear
- pH: 7.0 ± 0.1
- Test with 100 mg/dL standard → pink color after 10 min
```

#### C. Glucose Standard Solution (1000 mg/dL)
```
Ingredients:
- D-Glucose anhydrous (dried): 1.000 g
- Sodium benzoate: 0.200 g
- Distilled water: 100 mL

Procedure:
1. Dry glucose powder in oven 60°C for 2 hours
2. Cool in desiccator 30 minutes
3. Weigh exactly 1.000 g (analytical balance)
4. Dissolve in 80 mL water
5. Add sodium benzoate (preservative)
6. Transfer to 100 mL volumetric flask
7. Add water to mark
8. Mix well, store at 4°C (stable 3 months)

Prepare working standards by dilution:
- 500 mg/dL: 50 mL stock + 50 mL water
- 250 mg/dL: 25 mL stock + 75 mL water
- 100 mg/dL: 10 mL stock + 90 mL water
- 50 mg/dL: 5 mL stock + 95 mL water
```

### Measurement Procedure

#### Setup
```
Prepare 3 tubes:
┌─────────────┬──────────┬───────────┬─────────┐
│   Tube      │  Reagent │  Sample   │  Water  │
├─────────────┼──────────┼───────────┼─────────┤
│ Blank       │ 1000 μL  │     -     │  10 μL  │
│ Standard    │ 1000 μL  │ 10 μL std │    -    │
│ Sample      │ 1000 μL  │ 10 μL urin│    -    │
└─────────────┴──────────┴───────────┴─────────┘
```

#### Step-by-Step
```
1. Pipette 1000 μL GOD-POD reagent into each tube

2. Add samples:
   - Blank: 10 μL distilled water
   - Standard: 10 μL glucose std (100 mg/dL)
   - Sample: 10 μL urine sample

3. Mix by inverting 5 times (gently)

4. Incubate in water bath 37°C for 10 minutes
   - Set timer
   - Cover tubes with parafilm

5. Reaction occurs:
   Glucose + O₂ --[GOD]--> Gluconic acid + H₂O₂
   H₂O₂ + 4-AAP + Phenol --[POD]--> Quinoneimine (PINK)

6. After 10 min, color should be stable:
   - Blank: Colorless/pale yellow
   - Standard: Pink (consistent intensity)
   - Sample: Pink (intensity proportional to glucose)

7. Measure within 30 minutes (color stable)
```

#### Measurement with AS7265X
```python
# Measure quinoneimine peak @ 505 nm
# Use AS7265X channels 510 nm and 535 nm

sensor.takeMeasurements()
intensity_510 = sensor.getChannel510nm()
intensity_535 = sensor.getChannel535nm()

# Calculate absorbance (relative to blank)
abs_sample = -log10(intensity_510_sample / intensity_510_blank)
abs_standard = -log10(intensity_510_standard / intensity_510_blank)

# Calculate glucose concentration
glucose_mgdl = (abs_sample / abs_standard) * 100  # standard is 100 mg/dL

print(f"Glucose: {glucose_mgdl:.1f} mg/dL")
```

### Expected Results
- **Linearity:** 0-500 mg/dL (R² > 0.999)
- **Sensitivity:** LOD < 5 mg/dL
- **Precision:** CV% < 2%
- **Specificity:** >99% (enzyme specific for glucose)

---

## QUALITY CONTROL

### Daily QC
```
Run 3 levels of control every day:
- Low: 50 mg/dL (target ± 5)
- Normal: 100 mg/dL (target ± 10)
- High: 300 mg/dL (target ± 15)

Acceptance criteria:
✓ All 3 within ±10% of target
✓ CV% < 5%
✗ If fail: Recalibrate sensor, check reagent expiry
```

### Weekly QC
```
1. Linearity check (6 standards: 0, 50, 100, 200, 350, 500 mg/dL)
   - Plot calibration curve
   - R² should be > 0.995
   
2. Interference testing
   - Add 1 g/L albumin → should not affect ±5%
   - Add 0.5 mM bilirubin → should not affect ±5%
   - Add 1 mM ascorbic acid → GOD method resistant
   
3. Temperature stability
   - Measure same sample at 20°C, 25°C, 30°C
   - Difference should be < 10 mg/dL
```

### Monthly QC
```
1. Compare with reference method (Hexokinase or lab analyzer)
   - Test 20 patient samples
   - Calculate correlation (r > 0.95)
   - Bias should be < 10 mg/dL
   
2. Bland-Altman plot
   - Mean difference ± 2SD within clinical limits
   
3. Recalibrate ML model if systematic bias detected
```

---

## TROUBLESHOOTING

### Problem 1: Inconsistent Readings (CV% > 10%)

**Possible Causes:**
- Temperature fluctuation
- Bubbles in cuvette
- Dirty optical surface
- Sample degradation

**Solutions:**
```
✓ Stabilize room temperature 25°C ± 2°C
✓ Degas sample (centrifuge 1000 rpm, 2 min)
✓ Clean cuvette with ethanol, rinse with DI water
✓ Use fresh sample (< 2 hours old)
✓ Increase number of readings (10 → 20)
```

### Problem 2: Low Sensitivity (Can't detect < 50 mg/dL)

**Solutions:**
```
✓ Switch to GOD-POD method (LOD: 5 mg/dL)
✓ Increase integration time (AS7265X: 50 → 100 cycles)
✓ Use longer path length cuvette (1 cm → 2 cm)
✓ Retrain model with more low-concentration samples
```

### Problem 3: Interference from Other Substances

**Interferents:**
- Ascorbic acid (Vitamin C): Up to 0.5 mM OK
- Protein (Albumin): Up to 3 g/L OK
- Ketones: May cause +10-20% bias
- Hemoglobin (blood): Major interference!

**Solutions:**
```
✓ GOD-POD method: Resistant to most interferents
✓ Add ascorbate oxidase to eliminate Vitamin C
✓ Clarify turbid samples by centrifugation
✓ If hematuria: reject sample, request new collection
✓ Use machine learning model trained on interference data
```

### Problem 4: GOD-POD No Color Development

**Possible Causes:**
- Expired reagent
- Inactive enzyme (improper storage)
- Wrong pH
- Insufficient incubation time

**Troubleshooting:**
```
1. Check reagent expiry date
2. Test enzyme activity:
   - Add reagent to 500 mg/dL glucose
   - Should turn pink within 10 min
   - If no color → reagent failed
3. Check pH of buffer (should be 7.0 ± 0.1)
4. Ensure water bath exactly 37°C
5. Extend incubation to 15 min
6. Prepare fresh reagent
```

### Problem 5: AS7265X Baseline Drift

**Symptoms:**
- Readings increase/decrease over time
- Blank reading not zero

**Solutions:**
```
✓ Warm up sensor for 30 min before use
✓ Run dark calibration (sensor.takeMeasurementsWithBulb(0))
✓ Use temperature compensation algorithm
✓ Clean sensor window with isopropanol
✓ Shield sensor from ambient light (black housing)
✓ Recalibrate weekly with standards
```

---

## VALIDATION PROTOCOL

### Clinical Validation (20 Patient Samples)

```
1. Collect paired samples:
   - Morning first void urine
   - Measure with AS7265X method
   - Measure with hospital lab analyzer (reference)

2. Record data:
   ┌──────────┬─────────────┬──────────────┬───────┐
   │ Sample # │ AS7265X     │ Reference    │ Bias  │
   ├──────────┼─────────────┼──────────────┼───────┤
   │    1     │  95 mg/dL   │  100 mg/dL   │  -5   │
   │    2     │ 245 mg/dL   │  250 mg/dL   │  -5   │
   │   ...    │    ...      │     ...      │  ...  │
   └──────────┴─────────────┴──────────────┴───────┘

3. Statistical analysis:
   - Correlation coefficient (r) > 0.95
   - Mean bias < 10 mg/dL
   - 95% limits of agreement within ±30 mg/dL

4. If validation passes → Model ready for clinical use
   If validation fails → Retrain model, check calibration
```

---

## REFERENCES

1. Trinder, P. (1969). Determination of glucose in blood using glucose oxidase 
   with an alternative oxygen acceptor. Annals of Clinical Biochemistry, 6(1), 24-27.

2. Barham, D., & Trinder, P. (1972). An improved colour reagent for the 
   determination of blood glucose by the oxidase system. Analyst, 97(1151), 142-145.

3. Clinical and Laboratory Standards Institute (CLSI). (2013). 
   Urinalysis; Approved Guideline—Third Edition. GP16-A3.

4. AS7265X Datasheet. AMS AG. https://ams.com/as7265x

---

**Document Version:** 1.0  
**Last Updated:** 2024-10-28  
**Author:** Nafiz Ahmadin Harily (122430051)  
**Institution:** Universitas Tadulako - Metodologi Penelitian
