# Proportionate selection

`proportionate_selector` allows sampling from empirical discrete distribution,
at rumtime. Each sample is generated independently, and has no coupling to previously
generated or future samples. This allows for quick, and reliable sample generation from
some known discrete distribution.

## Use cases

- Multivariant a/b tests
- Simple lootbox generation in games
- Use in evolutionary algorithms
- Help content promotion
- Coupon code generation
- and more...

## Example

Suppose we want to build _very simple_ lootbox reward collectables, based on
some rarity associated with the reward collectables. And we want to be able to
modify _rarity_ of such collectables (thousands of possible items) are runtime.

For example,

| Reward Item | Rarity | Probability of Occurance (1/Rarity) |
| ----------- | :----: | :---------------------------------: |
| Reward A    |   50   |            (1/50) = 0.02            |
| Reward B    |   10   |            (1/10) = 0.10            |
| Reward C    |   10   |            (1/10) = 0.10            |
| Reward D    |   2    |             (1/2) = 0.5             |
| No Reward   | 3.5714 |          (1/3.5714) = 0.28          |

Note: `proportionate_selector` requires that sum of probabilities equals to 1.
For some reason, you are using different ranking methoddologies, you can
normalize probabilities prior to using `proportionate_selector`. In most cases,
you should be doing this anyways.

```rust
use proportionate_selector::*;

#[derive(PartialEq, Debug)]
pub struct LootboxItem {
    pub id: i32,
    /// Likelihood of recieve item from lootbox.
    /// Rarity represents inverse lilihood of recieveing
    /// this item.
    ///
    /// e.g. rairity of 1, means lootbox item will be more
    /// frequently generated as opposed to rarity of 100.
    pub rarity: f64,
}

impl Probability for LootboxItem {
    fn prob(&self) -> f64 {
        // rarity is modeled as 1 out of X occurance, so
        // rarity of 20 has probability of 1/20.
        1.0 / self.rarity
    }
}

let endOfLevel1Box = vec![
    LootboxItem {id: 0, rarity: 50.0},   // 2%
    LootboxItem {id: 1, rarity: 10.0},   // 10%
    LootboxItem {id: 2, rarity: 10.0},   // 10%
    LootboxItem {id: 3, rarity: 2.0},    // 50%
    LootboxItem {id: 4, rarity: 3.5714}, // 28%
];

// create discrete distribution for sampling
let epdf = DiscreteDistribution::new(&endOfLevel1Box, SamplingMethod::Linear).unwrap();
let s = epdf.sample();

println!("{:?}", epdf.sample());
```

## Benchmarks (+/- 5%)

| Sampling   |  Time  | Number of Items |
| ---------- | :----: | --------------- |
| Linear     | 30 ns  | 100             |
| Linear     |  6 us  | 10,000          |
| Linear     | 486 us | 1,000,000       |
| Cdf        | 31 ns  | 100             |
| Cdf        | 41 ns  | 10,000          |
| Cdf        | 62 ns  | 1,000,000       |
| Stochastic | 315 ns | 100             |
| Stochastic | 30 us  | 10,000          |
| Stochastic | 40 us  | 1,000,000       |

Beanchmark ran on:

```text
  Model Name: Mac mini
  Model Identifier: Macmini9,1
  Chip: Apple M1
  Total Number of Cores: 8 (4 performance and 4 efficiency)
  Memory: 16 GB
```

## Development

```bash
cargo build     # build
cargo test      # run tests
cargo doc       # generate docs
cargo criterion # benchmarks
cargo clippy    # linter
```
