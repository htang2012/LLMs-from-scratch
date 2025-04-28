# Calculating FLOPS, TFLOPS, and PFLOPS

**Formula:**

- **FLOPS (Floating Point Operations Per Second)** = Total Floating Point Operations ÷ Total Execution Time (in seconds)
- **TFLOPS (TeraFLOPS)** = FLOPS ÷ 1e12
- **PFLOPS (PetaFLOPS)** = FLOPS ÷ 1e15

**Steps:**

1. **Determine the Total Floating Point Operations (FLO):**
   - This is the total number of floating-point calculations your model performs. It depends on the architecture and the operations used.

2. **Measure the Total Execution Time:**
   - The total time taken to perform all the floating-point operations (in seconds).

3. **Calculate FLOPS:**
   - Use the formula above to compute FLOPS.

4. **Convert FLOPS to TFLOPS or PFLOPS:**
   - Divide FLOPS by 1e12 for TFLOPS.
   - Divide FLOPS by 1e15 for PFLOPS.

**Example Calculation:**

Suppose you have:

- Total Floating Point Operations (FLO) = **9 × 10¹⁴**
- Total Execution Time = **300 seconds**

**Compute FLOPS:**

- FLOPS = FLO ÷ Time
- FLOPS = (9 × 10¹⁴) ÷ 300
- FLOPS = **3 × 10¹² FLOPS**

**Convert to TFLOPS:**

- TFLOPS = FLOPS ÷ 1e12
- TFLOPS = (3 × 10¹²) ÷ 1e12
- TFLOPS = **3 TFLOPS**

**Convert to PFLOPS:**

- PFLOPS = FLOPS ÷ 1e15
- PFLOPS = (3 × 10¹²) ÷ 1e15
- PFLOPS = **0.003 PFLOPS**

**Summary:**

- **FLOPS:** 3 × 10¹² FLOPS
- **TFLOPS:** 3 TFLOPS
- **PFLOPS:** 0.003 PFLOPS

**Note on Prefixes:**

- **Kilo (K)** = 1e3
- **Mega (M)** = 1e6
- **Giga (G)** = 1e9
- **Tera (T)** = 1e12
- **Peta (P)** = 1e15

| Prefix       | Symbol | Power of 10 | Decimal Multiplier                | Power of 2 | Binary Multiplier                 | Binary Prefix | Binary Symbol |
|--------------|--------|-------------|-----------------------------------|------------|-----------------------------------|---------------|---------------|
| Kilo         | k      | 10^3        | 1,000                             | 2^10       | 1,024                             | Kibi          | Ki            |
| Mega         | M      | 10^6        | 1,000,000                         | 2^20       | 1,048,576                         | Mebi          | Mi            |
| Giga         | G      | 10^9        | 1,000,000,000                     | 2^30       | 1,073,741,824                     | Gibi          | Gi            |
| Tera         | T      | 10^12       | 1,000,000,000,000                 | 2^40       | 1,099,511,627,776                 | Tebi          | Ti            |
| Peta         | P      | 10^15       | 1,000,000,000,000,000             | 2^50       | 1,125,899,906,842,624             | Pebi          | Pi            |
| Exa          | E      | 10^18       | 1,000,000,000,000,000,000         | 2^60       | 1,152,921,504,606,846,976         | Exbi          | Ei            |
| Zetta        | Z      | 10^21       | 1,000,000,000,000,000,000,000     | 2^70       | 1,180,591,620,717,411,303,424     | Zebi          | Zi            |
| Yotta        | Y      | 10^24       | 1,000,000,000,000,000,000,000,000 | 2^80       | 1,208,925,819,614,629,174,706,176 | Yobi          | Yi            |

### Key Notes
- **Power of 10**: Used in SI units (e.g., 1 km = 1,000 meters) and decimal-based storage (e.g., hard drives, where 1 GB = 10^9 bytes).
- **Power of 2**: Used in computing for binary storage/memory (e.g., 1 GiB = 2^30 bytes). Binary prefixes (kibi, mebi, etc.) were introduced by the IEC to distinguish from decimal.
- **Formula**:
  - Decimal: `Value = Base × 10^n` (e.g., 1k = 1 × 10^3 = 1,000).
  - Binary: `Value = Base × 2^n` (e.g., 1Ki = 1 × 2^10 = 1,024).
- In practice, "k", "M", etc., may be used for binary values (e.g., 1 MB = 2^20 bytes in some contexts), which can cause confusion. Always check the context (e.g., storage vs. RAM).
