use ktstr::ktstr_test;
use ktstr::test_support::{OutputFormat, Payload, PayloadKind};

#[allow(dead_code)]
const FIO: Payload = Payload::new(
    "fio",
    PayloadKind::Binary("fio"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
);

// `workloads` must be an array literal `[FIO]`, not a bare path.
#[ktstr_test(workloads = FIO)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
