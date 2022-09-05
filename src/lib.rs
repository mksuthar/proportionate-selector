//! Proportionate selection from discrete distribution.
//!
//! `proportionate_selector` allows sampling from empirical discrete distribution,
//! at rumtime. Each sample is generated independently, and has no coupling to previously
//! generated or future samples. This allows for quick, and reliable sample generation from
//! some known discrete distribution.
//!
//! ## Use cases
//!
//! * Multivariant a/b tests
//! * Simple lootbox generation in games
//! * Use in evolutionary algorithms
//! * Help content promotion
//! * Coupon code generation
//! * and more...
//!
//! ## Example
//!
//! Suppose we want to build _very simple_ lootbox reward collectables, based on
//! some rarity associated with the reward collectables. And we want to be able to
//! modify _rarity_ of such collectables (thousands of possible items) are runtime.
//!
//! For example,
//!
//! | Reward Item | Rarity | Probability of Occurance (1/Rarity) |
//! | ----------- | :----: | :---------------------------------: |
//! | Reward A    |   50   |     (1/50) = 0.02                   |
//! | Reward B    |   10   |     (1/10) = 0.10                   |
//! | Reward C    |   10   |     (1/10) = 0.10                   |
//! | Reward D    |   2    |     (1/2) = 0.5                     |
//! | No Reward   | 3.5714 |     (1/3.5714) = 0.28               |
//!
//! Note: `proportionate_selector` requires that sum of probabilities equals to 1.
//! For some reason, you are using different ranking methoddologies, you can
//! normalize probabilities prior to using `proportionate_selector`. In most cases,
//! you should be doing this anyways.
//!
//! ```rust
//! use proportionate_selector::*;
//!
//! #[derive(PartialEq, Debug)]
//! pub struct LootboxItem {
//!     pub id: i32,
//!     /// Likelihood of recieve item from lootbox.
//!     /// Rarity represents inverse lilihood of recieveing
//!     /// this item.
//!     ///
//!     /// e.g. rairity of 1, means lootbox item will be more
//!     /// frequently generated as opposed to rarity of 100.
//!     pub rarity: f64,
//! }
//!
//! impl Probability for LootboxItem {
//!     fn prob(&self) -> f64 {
//!         // rarity is modeled as 1 out of X occurance, so
//!         // rarity of 20 has probability of 1/20.
//!         1.0 / self.rarity
//!     }
//! }
//!
//! let endOfLevel1Box = vec![
//!     LootboxItem {id: 0, rarity: 50.0},   // 2%
//!     LootboxItem {id: 1, rarity: 10.0},   // 10%
//!     LootboxItem {id: 2, rarity: 10.0},   // 10%
//!     LootboxItem {id: 3, rarity: 2.0},    // 50%
//!     LootboxItem {id: 4, rarity: 3.5714}, // 28%
//! ];
//!
//! // create discrete distribution for sampling
//! let epdf = DiscreteDistribution::new(&endOfLevel1Box, SamplingMethod::Linear).unwrap();
//! let s = epdf.sample();
//!
//! println!("{:?}", epdf.sample());
//! ```
//!
//! ## Benchmarks (+/- 5%)
//!
//! | Sampling     |  Time  | Number of Items |
//! | ------------ | :----: | --------------- |
//! | Linear       | 30 ns  | 100             |
//! | Linear       | 6 us   | 10,000          |
//! | Linear       | 486 us | 1,000,000       |
//! | Cdf          | 31 ns  | 100             |
//! | Cdf          | 41 ns  | 10,000          |
//! | Cdf          | 62 ns  | 1,000,000       |
//! | Stochastic   | 315 ns | 100             |
//! | Stochastic   | 30 us  | 10,000          |
//! | Stochastic   | 40 us  | 1,000,000       |
//!
//! Beanchmark ran on:
//! ```text
//!   Model Name: Mac mini
//!   Model Identifier: Macmini9,1
//!   Chip: Apple M1
//!   Total Number of Cores: 8 (4 performance and 4 efficiency)
//!   Memory: 16 GB
//! ```
//!
pub mod util;

use rand::distributions::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

use util::*;

/// Sampling method to use when, sampling from discrete distribution.
#[derive(Debug, Clone, Copy)]
pub enum SamplingMethod {
    /// Performs linear scan on probabilities.
    ///
    /// Worst case is O(n).
    Linear,
    /// Performs selection by creating cumulative distribution function (CDF),
    /// and performing selection.
    ///
    /// Worst case is O(ln n).
    CumulativeDistributionFunction,
    /// Performs selection using stochastic acceptance. Average case is O(1), but
    /// may require at most N call to random number generator function.
    ///
    /// Worst case is O(n).
    /// Average case is O(1).
    ///
    /// Reference: <https://arxiv.org/abs/1109.3627>
    StochasticAcceptance,
}

pub trait Probability {
    /// Returns non-negative probability of occurance.
    ///
    /// Probability must be within range of 0 to 1.
    fn prob(&self) -> f64;
}

enum DistributionStore<'a, T: Probability> {
    Frequency {
        freq: Vec<f64>,
        total: f64,
        items: &'a Vec<T>,
    },
    Cdf {
        cdf: Vec<f64>,
        items: &'a Vec<T>,
    },
    MaxFrequency {
        max_freq: f64,
        items: &'a Vec<T>,
    },
}

#[derive(Debug)]
pub enum ProportionalSelectionErr {
    ProbabilitiesGreatherThanOne,
    ProbabilitiesLessThanOne,
    InsufficientProbabilitiesProvided,
}

/// Represents empirical discrete distribution.
pub struct DiscreteDistribution<'a, T: Probability> {
    /// Stores distribution attributes for quick sample generation.
    store: DistributionStore<'a, T>,
}

impl<'_a, T: Probability> DiscreteDistribution<'_a, T> {
    /// Returns DiscreteDistribution based on the selection method.
    ///
    /// ```rust
    /// use proportionate_selector::*;
    ///
    /// #[derive(PartialEq)]
    /// pub struct MultiVariantMarketingWebsiteItem {
    ///     pub id: i32,
    ///     /// Likelihood of recieve marketing website version.
    ///     /// Rarity represents inverse lilihood of recieveing
    ///     /// this item.
    ///     ///
    ///     /// e.g. rairity of 1, means website item item will be more
    ///     /// frequently generated as opposed to rarity of 100.
    ///     pub rarity: f64,
    /// }
    ///
    /// impl Probability for MultiVariantMarketingWebsiteItem {
    ///     fn prob(&self) -> f64 {
    ///         // rarity is modeled as 1 out of X occurance, so
    ///         // rarity of 20 has probability of 1/20.
    ///         1.0 / self.rarity
    ///     }
    /// }
    ///
    /// let summer2020Launch = vec![
    ///     MultiVariantMarketingWebsiteItem {id: 0, rarity: 5.0}, // 20%
    ///     MultiVariantMarketingWebsiteItem {id: 1, rarity: 5.0}, // 20%
    ///     MultiVariantMarketingWebsiteItem {id: 2, rarity: 10.0}, // 10%
    ///     MultiVariantMarketingWebsiteItem {id: 3, rarity: 2.5}, // 40%
    ///     MultiVariantMarketingWebsiteItem {id: 4, rarity: 10.0}, // 40%
    /// ];
    ///
    /// // create discrete distribution for sampling
    /// let epdf = DiscreteDistribution::new(&summer2020Launch, SamplingMethod::Linear);
    ///
    /// assert_eq!(epdf.is_ok(), true);
    /// ```
    pub fn new(
        items: &'_a Vec<T>,
        method: SamplingMethod,
    ) -> Result<Self, ProportionalSelectionErr> {
        use ProportionalSelectionErr::*;

        // Apply some tolerance to probability bounds.
        const EPSILON: f64 = 0.001;
        let total_p: f64 = items.iter().map(|i| i.prob()).sum();

        // Impossible to perform sampling per occurance probabilitity, if
        // occurance of all possible scenario is greater than 1.
        if total_p > 1.0 + EPSILON {
            print!("{:?}", total_p);
            return Err(ProbabilitiesGreatherThanOne);
        }
        // Impossible to perform sampling per occurance probabilitity, if
        // occurance of all possible scenario is less than 1.
        if 1.0 - EPSILON > total_p {
            return Err(ProbabilitiesLessThanOne);
        }

        let n = items.len();
        if n <= 1 {
            return Err(InsufficientProbabilitiesProvided);
        }

        match method {
            SamplingMethod::Linear => {
                let mut freq: Vec<f64> = vec![0.0; n];
                let mut total = 0.0;

                for (i, item) in items.iter().enumerate() {
                    freq[i] = item.prob() * 100.0;
                    total += item.prob() * 100.0;
                }

                Ok(Self {
                    store: DistributionStore::Frequency { freq, total, items },
                })
            }

            SamplingMethod::CumulativeDistributionFunction => {
                let mut cdf: Vec<f64> = vec![0.0; n];
                let mut acc = 0.0;

                for (i, item) in items.iter().enumerate() {
                    acc += item.prob();
                    cdf[i] = acc;
                }

                Ok(Self {
                    store: DistributionStore::Cdf { cdf, items },
                })
            }

            SamplingMethod::StochasticAcceptance => {
                let max_freq = items
                    .iter()
                    .map(|i| i.prob())
                    .max_by(|lhs, rhs| lhs.total_cmp(rhs))
                    .unwrap_or(0.0);

                Ok(Self {
                    store: DistributionStore::MaxFrequency { max_freq, items },
                })
            }
        }
    }

    /// Returns a sample based on discrete distribution.
    ///
    /// As invocation of sample() reaches large number (e.g. +infinity), the
    /// difference between population (defined discrete distribution), and
    /// distribution from generated sample diminishes.
    ///
    /// ```rust
    /// use proportionate_selector::*;
    ///
    /// #[derive(PartialEq)]
    /// pub struct LootboxItem {
    ///     pub id: i32,
    ///     /// Likelihood of recieve item from lootbox.
    ///     /// Rarity represents inverse lilihood of recieveing
    ///     /// this item.
    ///     ///
    ///     /// e.g. rairity of 1, means lootbox item will be more
    ///     /// frequently generated as opposed to rarity of 100.
    ///     pub rarity: f64,
    /// }
    ///
    /// impl Probability for LootboxItem {
    ///     fn prob(&self) -> f64 {
    ///         // rarity is modeled as 1 out of X occurance, so
    ///         // rarity of 20 has probability of 1/20.
    ///         1.0 / self.rarity
    ///     }
    /// }
    ///
    /// let endOfLevel1Box = vec![
    ///     LootboxItem {id: 0, rarity: 5.0}, // 20%
    ///     LootboxItem {id: 1, rarity: 5.0}, // 20%
    ///     LootboxItem {id: 2, rarity: 10.0}, // 10%
    ///     LootboxItem {id: 3, rarity: 2.5}, // 40%
    ///     LootboxItem {id: 4, rarity: 10.0}, // 40%
    /// ];
    ///
    /// // create discrete distribution for sampling
    /// let epdf = DiscreteDistribution::new(&endOfLevel1Box, SamplingMethod::Linear).unwrap();
    /// let s = epdf.sample();
    ///
    /// assert_eq!(s.is_none(), false);
    /// ```
    ///
    pub fn sample(&'_a self) -> Option<&T> {
        match &self.store {
            DistributionStore::Cdf { cdf, items } => sample_cdf(cdf, items),
            DistributionStore::Frequency { freq, total, items } => {
                sample_linear(freq, total, items)
            }
            DistributionStore::MaxFrequency { max_freq, items } => {
                sample_stochastic(max_freq, items)
            }
        }
    }
}

fn sample_linear<'a, T: Probability>(
    freq: &'a [f64],
    total: &'a f64,
    items: &'a [T],
) -> Option<&'a T> {
    let mut rng = rand::thread_rng();
    let total_n = convert(*total + 1.0);
    let terminal = f64::from(Uniform::from(0..total_n).sample(&mut rng));
    let mut acc: f64 = 0.0;

    for (i, f) in freq.iter().enumerate() {
        acc += *f;
        if acc > terminal {
            return items.get(i);
        }
    }

    None
}

fn sample_cdf<'a, T: Probability>(cdf: &'a [f64], items: &'a [T]) -> Option<&'a T> {
    let mut rng = rand::thread_rng();
    let random = rng.gen();
    items.get(bisect_left(cdf, &random))
}

fn sample_stochastic<'a, T: Probability>(max_freq: &'a f64, items: &'a [T]) -> Option<&'a T> {
    let n = items.len();
    let mut small_rng = SmallRng::from_entropy();
    loop {
        let i = small_rng.gen_range(0..n);
        match items.get(i) {
            None => return None,
            Some(x) => {
                let rand: f64 = small_rng.gen();
                if rand < (x.prob() / max_freq) {
                    return Some(x);
                }
                continue;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use kolmogorov_smirnov::test_f64;
    use std::collections::HashMap;

    use crate::*;
    use SamplingMethod::*;

    /// Some probability of english alphabet on occurance.
    /// Refernce: https://www3.nd.edu/~busiforc/handouts/cryptography/letterfrequencies.html
    const FIXTURE_ALPHABETS_PROBS: [(char, f64); 26] = [
        ('E', 0.111607),
        ('M', 0.030129),
        ('A', 0.084966),
        ('H', 0.030034),
        ('R', 0.075809),
        ('G', 0.024705),
        ('I', 0.075448),
        ('B', 0.020720),
        ('O', 0.071635),
        ('F', 0.018121),
        ('T', 0.069509),
        ('Y', 0.017779),
        ('N', 0.066544),
        ('W', 0.012899),
        ('S', 0.057351),
        ('K', 0.011016),
        ('L', 0.054893),
        ('V', 0.010074),
        ('C', 0.045388),
        ('X', 0.002902),
        ('U', 0.036308),
        ('Z', 0.002722),
        ('D', 0.033844),
        ('J', 0.001965),
        ('P', 0.031671),
        ('Q', 0.001962),
    ];

    impl Probability for (char, f64) {
        fn prob(&self) -> f64 {
            self.1
        }
    }

    /// Basic Monte Carlo Simulation
    fn monte_carlo(store: DiscreteDistribution<(char, f64)>, n: i64) -> HashMap<char, f64> {
        let mut counter: HashMap<char, f64> = HashMap::new();
        let mut iter = 0;
        loop {
            if iter > n {
                break;
            }
            iter += 1;

            match store.sample() {
                Some(p) => *counter.entry(p.0).or_insert(0.0) += 1.0,
                None => continue,
            }
        }
        counter
    }

    /// Asserts if generated distribution from sample() matches that of
    /// provided distribution using Kolmogorov-Smirnov test.
    ///
    /// Reference: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    fn matches_expected_distribution(n: i64, method: SamplingMethod) {
        // Setup
        let mut abc_probs = FIXTURE_ALPHABETS_PROBS.to_vec();
        abc_probs.sort_by(|a, b| a.0.cmp(&b.0));
        let store = DiscreteDistribution::new(&abc_probs, method.clone()).unwrap();

        // Act
        let mut obs: Vec<(char, f64)> =
            monte_carlo(store, n).iter().map(|r| (*r.0, *r.1)).collect();

        obs.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let expected_pdf: Vec<f64> = abc_probs.iter().map(|e| e.1).collect();
        let obs_pdf: Vec<f64> = obs.iter().map(|o| o.1 / (n as f64)).collect();

        // Assert
        let ks_result = test_f64(&expected_pdf, &obs_pdf, 0.99);
        assert!(
            !ks_result.is_rejected,
            "Generated samples do not belong to expected distribution per KS test, for n={}, sampling={:#?} confidence={}",
            n,
            method,
            ks_result.confidence
        )
    }

    #[test]
    fn linear_sampling() {
        matches_expected_distribution(1000, Linear);
        matches_expected_distribution(100_00, Linear);
        matches_expected_distribution(100_00_00, Linear);
    }

    #[test]
    fn cfd_sampling() {
        matches_expected_distribution(1000, CumulativeDistributionFunction);
        matches_expected_distribution(100_00, CumulativeDistributionFunction);
        matches_expected_distribution(100_00_00, CumulativeDistributionFunction);
    }

    #[test]
    fn stochastic_sampling() {
        matches_expected_distribution(1000, StochasticAcceptance);
        matches_expected_distribution(100_00, StochasticAcceptance);
        matches_expected_distribution(100_00_00, StochasticAcceptance);
    }
}
