//! Generic synchronization primitives shared across ktstr subsystems.
//!
//! Keeps small, reusable blocking primitives out of feature-specific
//! modules. Callers compose these — they do not carry domain
//! semantics like "probe readiness" or "phase-B attach" in their
//! type or method names.

use std::sync::{Condvar, Mutex};

/// One-shot signal from a producer thread to one or more waiters.
///
/// `set` flips the state and wakes every waiter currently blocked in
/// `wait`; subsequent waiters return immediately. Uses
/// `Mutex<bool> + Condvar` under the hood so waiters block in the
/// kernel instead of spinning. Replaces the `Arc<AtomicBool>` +
/// `while !flag { thread::sleep(10ms) }` pattern callers previously
/// used to hand off readiness between producer and consumer threads.
#[derive(Default)]
pub struct Latch {
    set: Mutex<bool>,
    cv: Condvar,
}

impl Latch {
    /// Create a new latch in the unset state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the latch and wake every waiter. Idempotent: a second call
    /// is a no-op beyond re-notifying, matching the previous
    /// `AtomicBool::store(true, Release)` semantics.
    pub fn set(&self) {
        let mut guard = self.set.lock().unwrap();
        *guard = true;
        self.cv.notify_all();
    }

    /// Block until `set` is called. Returns immediately if already set.
    pub fn wait(&self) {
        let mut guard = self.set.lock().unwrap();
        while !*guard {
            guard = self.cv.wait(guard).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    /// An unset latch blocks `wait` until a producer thread calls
    /// `set`; the waiter must observe `set` before returning.
    #[test]
    fn latch_blocks_until_set_from_producer() {
        let latch = Arc::new(Latch::new());
        let l2 = latch.clone();
        let waiter = std::thread::spawn(move || {
            l2.wait();
        });
        // Give the waiter a chance to reach `cv.wait`.
        std::thread::sleep(Duration::from_millis(20));
        latch.set();
        waiter.join().unwrap();
    }

    /// A latch already in the set state returns from `wait`
    /// immediately — the mutex guards against the condvar missing the
    /// prior `notify_all`.
    #[test]
    fn latch_returns_immediately_when_already_set() {
        let latch = Latch::new();
        latch.set();
        let start = std::time::Instant::now();
        latch.wait();
        assert!(start.elapsed() < Duration::from_millis(50));
    }

    /// Two threads blocked in `wait` must both return after a single
    /// `set` — `notify_all` wakes every waiter in one call.
    #[test]
    fn set_wakes_every_waiter() {
        let latch = Arc::new(Latch::new());
        let a = latch.clone();
        let b = latch.clone();
        let wa = std::thread::spawn(move || a.wait());
        let wb = std::thread::spawn(move || b.wait());
        std::thread::sleep(Duration::from_millis(20));
        latch.set();
        wa.join().unwrap();
        wb.join().unwrap();
    }

    /// Calling `set` twice is idempotent — subsequent `wait` calls
    /// return immediately as they would after a single set.
    #[test]
    fn set_twice_is_idempotent() {
        let latch = Latch::new();
        latch.set();
        latch.set();
        latch.wait();
    }
}
