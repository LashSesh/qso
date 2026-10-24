//! Nelder–Mead simplex method with the dimension-adaptive coefficients of
//! Gao & Han (Comput. Optim. Appl. 51, 259–277, 2012).

use super::{Counted, OptimizeResult, Options};

pub(super) fn run(obj: &Counted<'_>, x0: &[f64], step: f64, options: &Options) -> OptimizeResult {
    let n = x0.len();
    let nf = n.max(1) as f64;
    let (alpha, beta, gamma, delta) =
        (1.0, 1.0 + 2.0 / nf, 0.75 - 1.0 / (2.0 * nf), 1.0 - 1.0 / nf);
    let delta = if n == 1 { 0.5 } else { delta };

    let mut simplex: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n + 1);
    simplex.push((obj.value(x0), x0.to_vec()));
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += if x[i] != 0.0 {
            step * x[i].abs().max(1.0)
        } else {
            step
        };
        simplex.push((obj.value(&x), x));
    }
    let mut history = Vec::new();
    let mut converged = false;
    let mut message = String::from("maximum number of iterations reached");
    let mut iterations = 0;
    let x_tol: f64 = 1e-10;

    for it in 0..options.max_iterations {
        simplex.sort_by(|a, b| a.0.total_cmp(&b.0));
        if options.record_history {
            history.push(simplex[0].0);
        }
        iterations = it;
        let f_spread = (simplex[n].0 - simplex[0].0).abs();
        let x_spread = simplex[1..]
            .iter()
            .flat_map(|(_, x)| x.iter().zip(&simplex[0].1).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f64::max);
        if f_spread <= options.f_tol.max(1e-15) * simplex[0].0.abs().max(1.0)
            && x_spread <= x_tol.max(options.f_tol.sqrt())
        {
            converged = true;
            message = "simplex collapsed below tolerance".into();
            break;
        }
        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex[..n].iter().map(|(_, x)| x[j]).sum::<f64>() / nf)
            .collect();
        let towards = |coef: f64| -> Vec<f64> {
            centroid
                .iter()
                .zip(&simplex[n].1)
                .map(|(c, w)| c + coef * (c - w))
                .collect()
        };
        let xr = towards(alpha);
        let fr = obj.value(&xr);
        if fr < simplex[0].0 {
            let xe = towards(alpha * beta);
            let fe = obj.value(&xe);
            simplex[n] = if fe < fr { (fe, xe) } else { (fr, xr) };
        } else if fr < simplex[n - 1].0 {
            simplex[n] = (fr, xr);
        } else {
            let (xc, fc) = if fr < simplex[n].0 {
                let xc = towards(alpha * gamma);
                let fc = obj.value(&xc);
                (xc, fc)
            } else {
                let xc = towards(-gamma);
                let fc = obj.value(&xc);
                (xc, fc)
            };
            if fc < simplex[n].0.min(fr) {
                simplex[n] = (fc, xc);
            } else {
                // Shrink towards the best vertex.
                let best = simplex[0].1.clone();
                for vertex in simplex.iter_mut().skip(1) {
                    let x: Vec<f64> = best
                        .iter()
                        .zip(&vertex.1)
                        .map(|(b, v)| b + delta * (v - b))
                        .collect();
                    *vertex = (obj.value(&x), x);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (value, x) = simplex.swap_remove(0);
    OptimizeResult {
        x,
        value,
        iterations,
        evaluations: 0,
        converged,
        message,
        history,
    }
}
