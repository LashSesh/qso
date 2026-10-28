//! A worker pool for the state-vector kernels.
//!
//! Rayon's workers go to sleep as soon as they run out of work, and waking
//! them costs tens to hundreds of microseconds on virtual machines: the
//! first gate after a pause (an optimiser step, Python code) pays it. The
//! workers of this pool spin for a while ([`spin_time`]) after each job
//! before they park, so a simulation that resumes within that time finds
//! them awake.
//!
//! One job runs at a time. [`run`] hands out task indices `0..n` through an
//! atomic counter; the calling thread works on the job too and returns when
//! every task has finished and no worker still looks at the job. A call
//! made while another job runs (from another thread, or from inside a task)
//! runs serially instead of waiting, so nesting cannot deadlock and
//! parallel callers (rayon tasks) do not oversubscribe the machine.
//!
//! Which tasks run on which thread varies; the kernels write disjoint data
//! per task, so results do not depend on it.

use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A job: tasks `0..n` of `f`.
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    n: usize,
    next: AtomicUsize,
    finished: AtomicUsize,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Job<'_> {
    /// Run tasks until none is left.
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= self.n {
                return;
            }
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| (self.f)(i))) {
                let mut slot = self.panic.lock().unwrap_or_else(|e| e.into_inner());
                slot.get_or_insert(payload);
            }
            self.finished.fetch_add(1, SeqCst);
        }
    }
}

struct Pool {
    workers: usize,
    /// The running job (type-erased lifetime: valid while `active` workers
    /// may use it, see [`run`]).
    current: AtomicPtr<Job<'static>>,
    /// Incremented for every job.
    epoch: AtomicU64,
    /// Workers between reading `current` and leaving the job.
    active: AtomicUsize,
    /// Held by the thread whose job runs.
    submit: Mutex<()>,
    /// Parked workers wait here.
    sleep: Mutex<()>,
    wake: Condvar,
    sleeping: AtomicUsize,
    spin: Duration,
}

thread_local! {
    /// Whether this thread is a pool worker or runs a job (nested calls run
    /// serially).
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

/// How long workers spin after a job before parking: `QSO_POOL_SPIN_US`
/// microseconds, 200 µs by default (0 under Miri).
pub(crate) fn spin_time() -> Duration {
    if cfg!(miri) {
        return Duration::ZERO;
    }
    let us = std::env::var("QSO_POOL_SPIN_US")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(200);
    Duration::from_micros(us.min(1_000_000))
}

/// Worker threads: `QSO_NUM_THREADS` (or `RAYON_NUM_THREADS`) minus one, or
/// the available parallelism minus one; the caller is the remaining one.
fn worker_count() -> usize {
    let configured = ["QSO_NUM_THREADS", "RAYON_NUM_THREADS"]
        .iter()
        .find_map(|k| std::env::var(k).ok()?.parse::<usize>().ok())
        .filter(|&t| t > 0);
    let threads = configured.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    if cfg!(miri) {
        // Two workers: interleavings worth checking, at Miri's speed.
        return 2;
    }
    threads.saturating_sub(1)
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            workers: worker_count(),
            current: AtomicPtr::new(std::ptr::null_mut()),
            epoch: AtomicU64::new(0),
            active: AtomicUsize::new(0),
            submit: Mutex::new(()),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
            sleeping: AtomicUsize::new(0),
            spin: spin_time(),
        }));
        for k in 0..pool.workers {
            std::thread::Builder::new()
                .name(format!("qso-pool-{k}"))
                .spawn(move || worker(pool))
                .expect("a pool thread");
        }
        pool
    })
}

fn worker(pool: &'static Pool) {
    INSIDE.with(|c| c.set(true));
    let mut seen = pool.epoch.load(SeqCst);
    loop {
        // Wait for a new job: spin, then park.
        let start = Instant::now();
        loop {
            if pool.epoch.load(SeqCst) != seen {
                break;
            }
            if start.elapsed() >= pool.spin {
                let guard = pool.sleep.lock().unwrap_or_else(|e| e.into_inner());
                pool.sleeping.fetch_add(1, SeqCst);
                let _guard = pool
                    .wake
                    .wait_while(guard, |_| pool.epoch.load(SeqCst) == seen)
                    .unwrap_or_else(|e| e.into_inner());
                pool.sleeping.fetch_sub(1, SeqCst);
                break;
            }
            std::hint::spin_loop();
        }
        seen = pool.epoch.load(SeqCst);
        pool.active.fetch_add(1, SeqCst);
        let job = pool.current.load(SeqCst);
        if !job.is_null() {
            // SAFETY: the job outlives every worker counted in `active`
            // that read it from `current` (see `run`).
            unsafe { (*job).work() };
        }
        pool.active.fetch_sub(1, SeqCst);
    }
}

/// Call `f(i)` for every `i in 0..n` on the pool's threads and the calling
/// thread; returns when all have finished. Serial when the pool is busy,
/// when called from inside a task, for a single task, or without workers.
/// A panic in a task is resumed here after the others finished.
pub(crate) fn run<F>(n: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    let serial = |f: &F| (0..n).for_each(f);
    if n <= 1 || INSIDE.with(Cell::get) {
        return serial(&f);
    }
    let pool = pool();
    if pool.workers == 0 {
        return serial(&f);
    }
    let Ok(_submit) = pool.submit.try_lock() else {
        return serial(&f);
    };
    let job = Job {
        f: &f,
        n,
        next: AtomicUsize::new(0),
        finished: AtomicUsize::new(0),
        panic: Mutex::new(None),
    };
    // SAFETY (lifetime erasure): `current` is cleared and `active` drained
    // below before `job` goes out of scope.
    let erased = &job as *const Job<'_> as *mut Job<'static>;
    pool.current.store(erased, SeqCst);
    pool.epoch.fetch_add(1, SeqCst);
    if pool.sleeping.load(SeqCst) > 0 {
        let _guard = pool.sleep.lock().unwrap_or_else(|e| e.into_inner());
        pool.wake.notify_all();
    }
    INSIDE.with(|c| c.set(true));
    job.work();
    INSIDE.with(|c| c.set(false));
    while job.finished.load(SeqCst) < n {
        std::hint::spin_loop();
    }
    // No worker may pick the job up from here on, and those that did must
    // leave it: a worker that increments `active` after this load reads
    // `current` after the store (sequential consistency) and finds null.
    pool.current.store(std::ptr::null_mut(), SeqCst);
    while pool.active.load(SeqCst) != 0 {
        std::hint::spin_loop();
    }
    let payload = job.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(payload) = payload {
        resume_unwind(payload);
    }
}

#[cfg(test)]
mod tests;
