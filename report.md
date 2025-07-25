# Demographic Predictive Modeling from MRIs
##### Predicting age/gender from structural MRI (T1) using CNNs
### Pre-processed Data
---
| participant_id   |     Age | AgeGroup   | Child_Adult   | Gender   | Handedness   |   ToM Booklet-Matched |   ToM Booklet-Matched-NOFB |   FB_Composite | FB_Group   |   WPPSI BD raw |   WPPSI BD scaled |   KBIT_raw |   KBIT_standard |   DCCS Summary | Scanlog: Scanner   | Scanlog: Coil   | Scanlog: Voxel slize   |   Scanlog: Slice Gap |
|:-----------------|--------:|:-----------|:--------------|:---------|:-------------|----------------------:|---------------------------:|---------------:|:-----------|---------------:|------------------:|-----------:|----------------:|---------------:|:-------------------|:----------------|:-----------------------|---------------------:|
| sub-pixar001     | 4.77481 | 4yo        | child         | M        | R            |                  0.8  |                   0.736842 |              6 | pass       |             22 |                13 |        nan |             nan |              3 | 3T1                | 7-8yo 32ch      | 3mm iso                |                  0.1 |
| sub-pixar002     | 4.85695 | 4yo        | child         | F        | R            |                  0.72 |                   0.736842 |              4 | inc        |             18 |                 9 |        nan |             nan |              2 | 3T1                | 7-8yo 32ch      | 3mm iso                |                  0.1 |
| sub-pixar003     | 4.15332 | 4yo        | child         | F        | R            |                  0.44 |                   0.421053 |              3 | inc        |             15 |                 9 |        nan |             nan |              3 | 3T1                | 7-8yo 32ch      | 3mm iso                |                  0.1 |
| sub-pixar004     | 4.47365 | 4yo        | child         | F        | R            |                  0.64 |                   0.736842 |              2 | fail       |             17 |                10 |        nan |             nan |              3 | 3T1                | 7-8yo 32ch      | 3mm iso                |                  0.2 |
| sub-pixar005     | 4.83778 | 4yo        | child         | F        | R            |                  0.6  |                   0.578947 |              4 | inc        |             13 |                 5 |        nan |             nan |              2 | 3T1                | 7-8yo 32ch      | 3mm iso                |                  0.2 |
#### Class Prioirs
---
##### Classification 1: AgeGroup
- $P(Y=3yo) = 10.97\%$
- $P(Y=4yo) = 9.03\%$
- $P(Y=5yo) = 21.94\%$
- $P(Y=7yo) = 14.84\%$
- $P(Y=8-12yo) = 21.94\%$
- $P(Y=Adult) = 21.29\%$
##### Classification 2: Child_Adult
- $P(Y=adult) = 21.29\%$
- $P(Y=child) = 78.71\%$
##### Classification 3: Gender
- $P(Y=F) = 54.19\%$
- $P(Y=M) = 45.81\%$
##### n = 155
#### Class Priors After Resampling
##### Classification 1: AgeGroup
- $P(Y=3yo) = 16.67\%$
- $P(Y=4yo) = 16.67\%$
- $P(Y=5yo) = 16.67\%$
- $P(Y=7yo) = 16.67\%$
- $P(Y=8-12yo) = 16.67\%$
- $P(Y=Adult) = 16.67\%$
##### Classification 2: Child_Adult
- $P(Y=adult) = 16.67\%$
- $P(Y=child) = 83.33\%$
##### Classification 3: Gender
- $P(Y=F) = 50.00\%$
- $P(Y=M) = 50.00\%$
##### n = 84
[fetch_atlas_harvard_oxford] Dataset found in C:\Users\danie\nilearn_data\fsl
