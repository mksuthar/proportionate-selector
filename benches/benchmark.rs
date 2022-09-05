use criterion::*;
use proportionate_selector::*;
use rand::Rng;

#[derive(Debug)]
pub struct Item {
    pub occurance_p: f64,
}

impl Probability for Item {
    fn prob(&self) -> f64 {
        self.occurance_p
    }
}

fn mk_seed(n: i32) -> Vec<Item> {
    let mut rng = rand::thread_rng();
    let mut items = Vec::new();

    for _ in 0..n {
        let random = f64::from(rng.gen_range(0..10000));
        items.push(Item {
            occurance_p: 1.0 / (random + 0.001),
        })
    }

    // normalize probabilities
    let total_raw_p: f64 = items.iter().map(|i| i.prob()).sum();
    items.iter_mut().for_each(|i| i.occurance_p /= total_raw_p);
    items
}

fn criterion_benchmark(c: &mut Criterion) {
    use SamplingMethod::*;

    let mut group = c.benchmark_group("Sampling");

    let ns: [i32; 6] = [10, 100, 1000, 10_000, 100_000, 1_000_000];
    let samplings: [SamplingMethod; 3] =
        [Linear, CumulativeDistributionFunction, StochasticAcceptance];

    for sampling in samplings {
        for n in ns {
            let seed = mk_seed(n);
            let store = DiscreteDistribution::new(&seed, sampling).unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", sampling), n),
                &n,
                |b, _n| b.iter(|| store.sample()),
            );
        }
    }
    group.finish()
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
