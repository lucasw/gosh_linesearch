// [[file:../linesearch.note::*header][header:1]]
//! Line search, also called one-dimensional search, refers to an optimization
//! procedure for univariable functions.
//! 
//! # Available algorithms
//! 
//! * MoreThuente
//! * BackTracking
//! * BackTrackingArmijo
//! * BackTrackingWolfe
//! * BackTrackingStrongWolfe
//! 
//! # References
//! 
//! * Sun, W.; Yuan, Y. Optimization Theory and Methods: Nonlinear Programming, 1st
//!   ed.; Springer, 2006.
//! * Nocedal, J.; Wright, S. Numerical Optimization; Springer Science & Business
//!   Media, 2006.
//!
//! # Examples
//!
//! ```ignore
//! use line::linesearch;
//! 
//! let mut step = 1.0;
//! let count = linesearch()
//!     .with_initial_step(1.5) // the default is 1.0
//!     .with_algorithm("BackTracking") // the default is MoreThuente
//!     .find(5, |a: f64, out: &mut Output| {
//!         // restore position
//!         x.veccpy(&x_k);
//!         // update position with step along d
//!         x.vecadd(&d_k, a);
//!         // update value and gradient
//!         out.fx = f(x, &mut gx)?;
//!         // update line search gradient
//!         out.gx = gx.vecdot(d);
//!         // update optimal step size
//!         step = a;
//!         // return any user defined data
//!         Ok(())
//!     })?;
//! 
//! let ls = linesearch()
//!     .with_max_iterations(5) // the default is 10
//!     .with_initial_step(1.5) // the default is 1.0
//!     .with_algorithm("BackTracking") // the default is MoreThuente
//!     .find_iter(|a: f64, out: &mut Output| {
//!         // restore position
//!         x.veccpy(&x_k);
//!         // update position with step along d
//!         x.vecadd(&d_k, a);
//!         // update value and gradient
//!         out.fx = f(x, &mut gx)?;
//!         // update line search gradient
//!         out.gx = gx.vecdot(d);
//!         // update optimal step size
//!         step = a;
//!         // return any user defined data
//!         Ok(())
//!     })?;
//! 
//! for success in ls {
//!     if success {
//!         //
//!     } else {
//!         //
//!     }
//! }
//!```

use crate::common::*;
// header:1 ends here

// [[file:../linesearch.note::*mods][mods:1]]
mod backtracking;
mod morethuente;
// mods:1 ends here

// [[file:../linesearch.note::*common][common:1]]
pub(crate) mod common {
    pub use gut::prelude::*;
}
// common:1 ends here

// [[file:../linesearch.note::*algorithm][algorithm:1]]
/// Line search algorithms.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LineSearchAlgorithm {
    /// MoreThuente method proposd by More and Thuente. This is the default for
    /// regular LBFGS.
    MoreThuente,

    ///
    /// BackTracking method with the Armijo condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// the sufficient decrease (Armijo) condition,
    ///   - f(x + a * d) <= f(x) + ftol * a * g(x)^T d,
    ///
    /// where x is the current point, d is the current search direction, and
    /// a is the step length.
    ///
    BackTrackingArmijo,

    /// BackTracking method with strong Wolfe condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (BacktrackingArmijo)
    /// and the following condition,
    ///   - |g(x + a * d)^T d| <= gtol * |g(x)^T d|,
    ///
    /// where x is the current point, d is the current search direction, and
    /// a is the step length.
    ///
    BackTrackingStrongWolfe,

    ///
    /// BackTracking method with regular Wolfe condition.
    ///
    /// The backtracking method finds the step length such that it satisfies
    /// both the Armijo condition (BacktrackingArmijo)
    /// and the curvature condition,
    ///   - g(x + a * d)^T d >= gtol * g(x)^T d,
    ///
    /// where x is the current point, d is the current search direction, and a
    /// is the step length.
    ///
    BackTrackingWolfe,
}

impl Default for LineSearchAlgorithm {
    /// The default algorithm (MoreThuente method).
    fn default() -> Self {
        LineSearchAlgorithm::MoreThuente
    }
}
// algorithm:1 ends here

// [[file:../linesearch.note::*base][base:1]]
#[derive(Clone, Debug, PartialEq)]
pub enum LineSearchCondition {
    /// The sufficient decrease condition.
    Armijo,
    Wolfe,
    StrongWolfe,
}

/// A trait for performing line search
pub(crate) trait LineSearchFind<E>
where
    E: Fn(f64) -> (f64, f64),
{
    /// Given initial step size and phi function, returns an satisfactory step
    /// size.
    ///
    /// `step` is a positive scalar representing the step size along search
    /// direction. phi is an univariable function of `step` for evaluating the
    /// value and the gradient projected onto search direction.
    fn find(&mut self, step: &mut f64, phi: E) -> Result<usize>;
}
// base:1 ends here

// [[file:../linesearch.note::*builder][builder:1]]
/// A unified interface to line search methods.
///
/// # Examples
///
/// ```ignore
/// use line::linesearch;
/// 
/// let mut step = 1.0;
/// let count = linesearch()
///     .with_initial_step(1.5) // the default is 1.0
///     .with_algorithm("BackTracking") // the default is MoreThuente
///     .find(5, |a: f64, out: &mut Output| {
///         // restore position
///         x.veccpy(&x_k);
///         // update position with step along d
///         x.vecadd(&d_k, a);
///         // update value and gradient
///         out.fx = f(x, &mut gx)?;
///         // update line search gradient
///         out.gx = gx.vecdot(d);
///         // update optimal step size
///         step = a;
///         // return any user defined data
///         Ok(())
///     })?;
/// 
/// let ls = linesearch()
///     .with_max_iterations(5) // the default is 10
///     .with_initial_step(1.5) // the default is 1.0
///     .with_algorithm("BackTracking") // the default is MoreThuente
///     .find_iter(|a: f64, out: &mut Output| {
///         // restore position
///         x.veccpy(&x_k);
///         // update position with step along d
///         x.vecadd(&d_k, a);
///         // update value and gradient
///         out.fx = f(x, &mut gx)?;
///         // update line search gradient
///         out.gx = gx.vecdot(d);
///         // update optimal step size
///         step = a;
///         // return any user defined data
///         Ok(())
///     })?;
/// 
/// for success in ls {
///     if success {
///         //
///     } else {
///         //
///     }
/// }
///```
pub fn linesearch() -> LineSearch {
    LineSearch::default()
}

pub struct LineSearch {
    algorithm: LineSearchAlgorithm,
    initial_step: f64,
}

impl Default for LineSearch {
    fn default() -> Self {
        LineSearch {
            algorithm: LineSearchAlgorithm::default(),
            initial_step: 1.0,
        }
    }
}

impl LineSearch {
    /// Set initial step size when performing line search. The default is 1.0.
    pub fn with_initial_step(mut self, stp: f64) -> Self {
        assert!(
            stp.is_sign_positive(),
            "line search initial step should be a positive float!"
        );

        self.initial_step = stp;
        self
    }

    /// Set line search algorithm. The default is MoreThuente algorithm.
    pub fn with_algorithm(mut self, s: &str) -> Self {
        self.algorithm = match s {
            "MoreThuente" => LineSearchAlgorithm::MoreThuente,
            "BackTracking" | "BackTrackingWolfe" => LineSearchAlgorithm::BackTrackingWolfe,
            "BackTrackingStrongWolfe" => LineSearchAlgorithm::BackTrackingWolfe,
            "BackTrackingArmijo" => LineSearchAlgorithm::BackTrackingArmijo,
            _ => unimplemented!(),
        };

        self
    }
}
// builder:1 ends here

// [[file:../linesearch.note::57376052][57376052]]
use std::fmt::Debug;

// 定义线搜索函数计算核心
pub trait LineSearchFindNext {
    // 执行单步line search. 通过返回值可判断当前位置是否满足线搜索条件
    fn find_next<E>(&self, stp: &mut f64, phi: E) -> Result<bool>
    where
        E: FnMut(f64) -> Result<(f64, f64)>;
}

// Input is initial step for performing line search
pub type Input = f64;

// 需要搜索步长对应的函数值及梯度
pub struct Output {
    /// The value of function at step `x`
    pub fx: f64,
    /// The gradient of function at step `x`
    pub gx: f64,
}

// 给定NAN数据, 避免未处理output可能的副作用
impl Default for Output {
    fn default() -> Self {
        use std::f64::NAN;
        Self { fx: NAN, gx: NAN }
    }
}

#[derive(Clone, Debug)]
pub struct Progress<T>
where
    T: Debug + Clone,
{
    /// current step
    pub step: f64,
    /// Indicates line search done or not
    pub done: bool,
    /// The data returned from user defined closure for function evaluation
    pub data: T,
}

/// T is user defined data
pub struct LineSearchEval<E, T>
where
    E: FnMut(Input, &mut Output) -> Result<T>,
    T: Debug + Clone,
{
    eval_fn: E,
    user_data: Option<T>,
}

impl<E, T> LineSearchEval<E, T>
where
    E: FnMut(Input, &mut Output) -> Result<T>,
    T: Debug + Clone,
{
    pub fn new(f: E) -> Self {
        Self {
            eval_fn: f,
            user_data: None,
        }
    }

    /// 调用回调函数, 同时保留用户自定义进度数据
    pub fn call(&mut self, x: f64) -> Result<(f64, f64)> {
        let mut out = Output::default();
        self.user_data = (self.eval_fn)(x, &mut out)?.into();
        Ok((out.fx, out.gx))
    }
}

pub struct LineSearchIter<A, E, T>
where
    A: LineSearchFindNext,
    E: FnMut(Input, &mut Output) -> Result<T>,
    T: Debug + Clone,
{
    step: f64,
    eval: LineSearchEval<E, T>,
    algo: Option<A>,
}

impl<A, E, T> Iterator for LineSearchIter<A, E, T>
where
    A: LineSearchFindNext,
    E: FnMut(Input, &mut Output) -> Result<T>,
    T: Debug + Clone,
{
    type Item = Progress<T>;

    /// Iterate over current line search step along searching direction. Return
    /// user defined progress data.
    fn next(&mut self) -> Option<Self::Item> {
        let mut step = self.step;
        let mut algo = self.algo.take();
        let done = algo
            .as_mut()
            .unwrap()
            .find_next(&mut step, |stp| {
                let out = self.eval.call(stp)?;
                Ok(out)
            })
            .ok()?;
        self.step = step;
        self.algo = algo;

        Progress {
            // 关键数据1: 当前步长
            step,
            // 关键数据2: 完成与否
            done,
            // 用户数据: 用户定义的重要数据
            data: self.eval.user_data.take().expect("no user data"),
        }
        .into()
    }
}
// 57376052 ends here

// [[file:../linesearch.note::*api][api:1]]
impl LineSearch {
    /// Perform line search with a callback function `phi` to evaluate function
    /// value and gradient projected onto search direction. This is the iterator
    /// version of `find` method.
    fn find_iter<E, T>(&self, phi: E) -> impl Iterator<Item = Progress<T>>
    where
        E: FnMut(Input, &mut Output) -> Result<T>,
        T: Debug + Clone,
    {
        use self::LineSearchAlgorithm as lsa;

        let mut bt_iter = None;
        let mut mt_iter = None;
        match self.algorithm {
            lsa::MoreThuente => {
                let iter = LineSearchIter {
                    step: self.initial_step,
                    eval: crate::LineSearchEval::new(phi),
                    algo: crate::morethuente::MoreThuente::default().into(),
                };
                mt_iter = iter.into();
            }
            other => {
                let condition = match other {
                    lsa::BackTrackingWolfe => LineSearchCondition::Wolfe,
                    lsa::BackTrackingStrongWolfe => LineSearchCondition::StrongWolfe,
                    lsa::BackTrackingArmijo => LineSearchCondition::Armijo,
                    _ => todo!(),
                };
                let mut ls = LineSearchIter {
                    step: self.initial_step,
                    eval: crate::LineSearchEval::new(phi),
                    algo: crate::backtracking::BackTracking::default()
                        .set_condition(condition)
                        .into(),
                };
                bt_iter = ls.into();
            }
        }
        bt_iter.into_iter().flatten().chain(mt_iter.into_iter().flatten())
    }

    /// Perform line search with a callback function `phi` to evaluate function
    /// value and gradient projected onto search direction within `m` iterations
    ///
    /// # Return
    ///
    /// Return success or not within line search iteration.
    ///
    pub fn find<E, T>(&self, m: usize, phi: E) -> bool
    where
        E: FnMut(Input, &mut Output) -> Result<T>,
        T: Debug + Clone,
    {
        for x in self.find_iter(phi).take(m) {
            if x.done {
                return true;
            }
        }
        warn!("ls: optimal step not found!");
        false
    }
}
// api:1 ends here

// [[file:../linesearch.note::*test][test:1]]
#[test]
fn test_ls_iter() -> Result<()> {
    let mut step = 1.0;
    let ls = linesearch()
        .with_initial_step(1.5) // the default is 1.0
        .with_algorithm("BackTracking") // the default is MoreThuente
        .find_iter(|a: f64, out: &mut Output| {
            // restore position
            // x.veccpy(&x_k);
            // update position with step along d
            // x.vecadd(&d_k, a);
            // update value and gradient
            // out.fx = f(x, &mut gx)?;
            out.fx = 0.1;
            // update line search gradient
            // out.gx = gx.vecdot(d);
            out.gx = 0.1;
            // update optimal step size
            // step = a;
            Ok(())
        });

    for x in ls.take(5) {
        dbg!(x);
    }

    Ok(())
}
// test:1 ends here
