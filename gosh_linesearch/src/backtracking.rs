// [[file:../linesearch.note::*imports][imports:1]]
use crate::*;
// imports:1 ends here

// [[file:../linesearch.note::*base][base:1]]
#[derive(Clone, Debug)]
pub struct BackTracking {
    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 1e-4. This parameter should be greater
    /// than zero and smaller than 0.5.
    ftol: f64,

    /// A parameter to control the accuracy of the line search routine.
    ///
    /// The default value is 0.9. If the function and gradient evaluations are
    /// inexpensive with respect to the cost of the iteration (which is
    /// sometimes the case when solving very large problems) it may be
    /// advantageous to set this parameter to a small value. A typical small
    /// value is 0.1. This parameter shuold be greater than the ftol parameter
    /// (1e-4) and smaller than 1.0.
    gtol: f64,

    /// The factor to increase step size
    fdec: f64,

    /// The factor to decrease step size
    finc: f64,

    /// The minimum step of the line search routine.
    ///
    /// The default value is 1e-20. This value need not be modified unless the
    /// exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    min_step: f64,

    /// The maximum step of the line search.
    ///
    /// The default value is 1e+20. This value need not be modified unless the
    /// exponents are too large for the machine being used, or unless the
    /// problem is extremely badly scaled (in which case the exponents should be
    /// increased).
    max_step: f64,

    /// Inexact line search condition
    pub condition: LineSearchCondition,
}

impl Default for BackTracking {
    fn default() -> Self {
        Self {
            ftol: 1e-4,
            gtol: 0.9,
            fdec: 0.5,
            finc: 2.1,

            min_step: 1e-20,
            max_step: 1e20,

            condition: LineSearchCondition::StrongWolfe,
        }
    }
}

impl BackTracking {
    pub(crate) fn set_condition(mut self, c: LineSearchCondition) -> Self {
        self.condition = c;
        self
    }
}
// base:1 ends here

// [[file:../linesearch.note::*core][core:1]]
/// # Parameters
///
/// * phi: callback function for evaluating value and gradient along searching direction
///
/// # Return
///
/// * Return true if an optimal step has been found.
///
fn find_next<E>(vars: &BackTracking, stp: &mut f64, mut phi: E) -> Result<bool>
where
    E: FnMut(f64) -> Result<(f64, f64)>,
{
    use self::LineSearchCondition::*;

    let (phi0, dginit) = phi(0.0)?;
    let dgtest = vars.ftol * dginit;

    // Evaluate the function and gradient values along search direction.
    let (phi_k, dg) = phi(*stp)?;

    let width = if phi_k > phi0 + *stp * dgtest {
        vars.fdec
    } else if vars.condition == Armijo {
        // The sufficient decrease condition.
        // Exit with the Armijo condition.
        return Ok(true);
    } else {
        // Check the Wolfe condition.
        if dg < vars.gtol * dginit {
            vars.finc
        } else if vars.condition == Wolfe {
            // Exit with the regular Wolfe condition.
            return Ok(true);
        } else if dg > -vars.gtol * dginit {
            vars.fdec
        } else {
            return Ok(true);
        }
    };

    *stp *= width;

    // The step is the minimum value.
    if *stp < vars.min_step {
        bail!("The line-search step became smaller than LineSearch::min_step.");
    }
    // The step is the maximum value.
    if *stp > vars.max_step {
        bail!("The line-search step became larger than LineSearch::max_step.");
    }

    Ok(false)
}
// core:1 ends here

// [[file:../linesearch.note::*api][api:1]]
impl crate::LineSearchFindNext for BackTracking {
    fn find_next<E>(&self, stp: &mut f64, phi: E) -> Result<bool>
    where
        E: FnMut(f64) -> Result<(f64, f64)>,
    {
        find_next(&self, stp, phi)
    }
}
// api:1 ends here
