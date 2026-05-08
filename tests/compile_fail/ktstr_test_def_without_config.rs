// A scheduler with `config_file_def` set, paired with a `#[ktstr_test]`
// that omits `config = ...`, must fail at compile time. The macro emits
// a `const _: () = { ... };` block that const-evaluates
// `(scheduler).config_file_def().is_some()` against the macro-known
// `config_set` flag and panics when the def is present but no content
// was supplied.
use ktstr::ktstr_test;
use ktstr::test_support::{Payload, Scheduler};

#[allow(dead_code)]
const SCHED_REQUIRES_CONFIG: Scheduler =
    Scheduler::new("requires_config").config_file_def("--config {file}", "/include-files/cfg.json");

#[allow(dead_code)]
const SCHED_REQUIRES_CONFIG_PAYLOAD: Payload = Payload::from_scheduler(&SCHED_REQUIRES_CONFIG);

#[ktstr_test(scheduler = SCHED_REQUIRES_CONFIG_PAYLOAD)]
fn def_without_config(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
