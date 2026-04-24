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

#[allow(dead_code)]
const STRESS: Payload = Payload::new(
    "stress",
    PayloadKind::Binary("stress-ng"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
);

#[ktstr_test(payload = FIO, payload = STRESS)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
