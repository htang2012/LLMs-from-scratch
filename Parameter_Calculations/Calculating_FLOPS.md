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
