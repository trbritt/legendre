//! Counter-based random numbers for stochastic PDE terms.
//!
//! **Design decision:** noise must be deterministic *and independent of
//! scheduling order*, so there is no RNG stream to advance. Instead every
//! random number is a pure function of a key — `(seed, step, block, field,
//! cell)` — built by chaining a `SplitMix64` finalizer. Any worker, in any
//! order, on any thread count, computes the identical increment for a given
//! cell at a given step; reproducing a run needs only the seed.

/// `SplitMix64` finalizer: a high-quality 64-bit mixing function.
#[inline(always)]
#[must_use]
pub const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Chain-mix a sequence of key components into one 64-bit state.
#[inline(always)]
#[must_use]
pub fn mix_key(seed: u64, parts: &[u64]) -> u64 {
    let mut s = splitmix64(seed);
    for &p in parts {
        s = splitmix64(s ^ p);
    }
    s
}

/// Map 64 random bits to a uniform deviate in `(0, 1]` (53 mantissa bits;
/// never zero, so `ln()` is safe).
#[inline(always)]
#[must_use]
pub fn unit_open(bits: u64) -> f64 {
    ((bits >> 11) + 1) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Standard normal deviate as a pure function of `key` (Box–Muller, cosine
/// branch, from two decorrelated uniforms derived from the key).
#[inline(always)]
#[must_use]
pub fn standard_normal(key: u64) -> f64 {
    let a = splitmix64(key);
    let b = splitmix64(a ^ 0xD1B5_4A32_D192_ED03);
    let u = unit_open(a);
    let v = unit_open(b);
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Exact equality is the property under test: the generator is a pure
    // function of its key.
    #[allow(clippy::float_cmp)]
    fn deterministic_and_key_sensitive() {
        let k = mix_key(42, &[7, 3, 1, 999]);
        assert_eq!(standard_normal(k), standard_normal(k));
        assert_ne!(standard_normal(k), standard_normal(k ^ 1));
    }

    #[test]
    fn moments_are_standard_normal() {
        let n = 200_000;
        let (mut sum, mut sum2) = (0.0, 0.0);
        for i in 0..n {
            let x = standard_normal(mix_key(1234, &[i]));
            sum += x;
            sum2 += x * x;
        }
        let mean = sum / n as f64;
        let var = sum2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }
}
