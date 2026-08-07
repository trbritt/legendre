//! Framework core: state, storage, scheduling, scratch, observation, and the
//! simulation owner. Everything here is grid- and physics-agnostic.

pub mod driver;
pub mod monte_carlo;
pub mod observer;
pub mod scheduler;
pub mod scratch;
pub mod simulation;
pub mod state;
pub mod storage;
