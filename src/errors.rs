use thiserror::Error;

#[derive(Error, Debug)]
pub enum ProportionalSelectionErr {
    #[error("Exepected sum of all probabilities to equal 1, but sum is {actual:?})")]
    SumOfAllProbabilitiesDoesNotEqualToOne { actual: f64 },
    #[error(
        "Insufficient amount of probabilities ({actual:?}) provided, must provide more than 1."
    )]
    InsufficientProbabilitiesProvided { actual: usize },
}
