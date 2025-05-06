use datafusion::{error::DataFusionError, execution::FunctionRegistry};


pub mod bernoulli;
pub mod beta;
pub mod binomial;
pub mod cauchy;
pub mod chi;
pub mod chi_squared;
pub mod dirac;
pub mod erlang;
pub mod exp;
pub mod fisher_snedecor;
pub mod gamma;
pub mod geometric;
pub mod gumbel;
pub mod hypergeometric;
pub mod inverse_gamma;
pub mod laplace;
pub mod log_normal;
pub mod negative_binomial;
pub mod normal;
pub mod pareto;
pub mod poisson;
pub mod students_t;
pub mod triangular;
pub mod uniform;
pub mod weibull;

pub fn register(registry: &mut dyn FunctionRegistry) -> Result<(), DataFusionError> {
    bernoulli::register(registry)?;
    beta::register(registry)?;
    binomial::register(registry)?;
    cauchy::register(registry)?;
    chi::register(registry)?;
    chi_squared::register(registry)?;
    dirac::register(registry)?;
    erlang::register(registry)?;
    exp::register(registry)?;
    fisher_snedecor::register(registry)?;
    gamma::register(registry)?;
    geometric::register(registry)?;
    gumbel::register(registry)?;
    hypergeometric::register(registry)?;
    inverse_gamma::register(registry)?;
    laplace::register(registry)?;
    log_normal::register(registry)?;
    negative_binomial::register(registry)?;
    normal::register(registry)?;
    pareto::register(registry)?;
    poisson::register(registry)?;
    students_t::register(registry)?;
    triangular::register(registry)?;
    uniform::register(registry)?;
    weibull::register(registry)?;
    Ok(())
}