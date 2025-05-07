use datafusion::{error::DataFusionError, execution::FunctionRegistry};

/// Bernoulli Distribution
pub mod bernoulli;
/// Beta Distribution
pub mod beta;
/// Binomial Distribution
pub mod binomial;
/// Cauchy Distribution
pub mod cauchy;
/// Chi Distribution
pub mod chi;
/// ChiSquared Distribution
pub mod chi_squared;
/// Dirac Distribution
pub mod dirac;
/// Erlang Distribution
pub mod erlang;
/// Exponential Distribution
pub mod exp;
/// Fisher-Snedcor (aka F) Distribution
pub mod fisher_snedecor;
/// Gamma Distribution
pub mod gamma;
/// Geometric Distribution
pub mod geometric;
/// Gumbel Distribution
pub mod gumbel;
/// Hypergeometric Distribution
pub mod hypergeometric;
/// Inverse Gamma Distribution
pub mod inverse_gamma;
/// Laplace Distribution
pub mod laplace;
/// LogNormal Distribution
pub mod log_normal;
/// Negative Binomial Distribution
pub mod negative_binomial;
/// Normal (aka Gaussian) Distribution
pub mod normal;
/// Pareto Distribution
pub mod pareto;
/// Poisson Distribution
pub mod poisson;
/// Student's T Distribution
pub mod students_t;
/// Triangular Distribution
pub mod triangular;
/// Uniform Distribution
pub mod uniform;
/// Weibull Distribution
pub mod weibull;


/// Register the functions for all the supported distributions.
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