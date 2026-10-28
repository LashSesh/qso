use super::*;
use std::sync::atomic::AtomicU64;

fn tasks() -> usize {
    if cfg!(miri) { 40 } else { 10_000 }
}

#[test]
fn every_task_runs_once() {
    let n = tasks();
    let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    run(n, |i| {
        hits[i].fetch_add(1, SeqCst);
    });
    assert!(hits.iter().all(|h| h.load(SeqCst) == 1));
    // Again after the workers have parked (longer than they spin).
    std::thread::sleep(spin_time() + Duration::from_millis(5));
    run(n, |i| {
        hits[i].fetch_add(1, SeqCst);
    });
    assert!(hits.iter().all(|h| h.load(SeqCst) == 2));
}

#[test]
fn nested_and_concurrent_calls_finish() {
    let n = if cfg!(miri) { 6 } else { 200 };
    let total = AtomicU64::new(0);
    // A task that runs a job itself: the inner job runs serially.
    run(n, |i| {
        run(n, |j| {
            total.fetch_add((i * n + j) as u64, SeqCst);
        });
    });
    let m = (n * n) as u64;
    assert_eq!(total.load(SeqCst), m * (m - 1) / 2);
    // Several threads at once: one gets the pool, the others run serially.
    let sums: Vec<u64> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..3)
            .map(|_| {
                scope.spawn(|| {
                    let sum = AtomicU64::new(0);
                    run(n, |i| {
                        sum.fetch_add(i as u64, SeqCst);
                    });
                    sum.load(SeqCst)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let expected = (n * (n - 1) / 2) as u64;
    assert!(sums.iter().all(|&s| s == expected));
}

#[test]
fn a_panic_reaches_the_caller_after_the_others_finished() {
    let n = if cfg!(miri) { 8 } else { 1000 };
    let done = AtomicU64::new(0);
    let result = catch_unwind(AssertUnwindSafe(|| {
        run(n, |i| {
            if i == 3 {
                panic!("task 3");
            }
            done.fetch_add(1, SeqCst);
        })
    }));
    assert!(result.is_err());
    assert_eq!(done.load(SeqCst), n as u64 - 1);
    // The pool still works.
    let count = AtomicU64::new(0);
    run(n, |_| {
        count.fetch_add(1, SeqCst);
    });
    assert_eq!(count.load(SeqCst), n as u64);
}
