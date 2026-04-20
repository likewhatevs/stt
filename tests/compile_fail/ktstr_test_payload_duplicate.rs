use ktstr::ktstr_test;
use ktstr::test_support::{OutputFormat, Payload, PayloadKind};

#[allow(dead_code)]
const FIO: Payload = Payload {
    name: "fio",
    kind: PayloadKind::Binary("fio"),
    output: OutputFormat::ExitCode,
    default_args: &[],
    default_checks: &[],
    metrics: &[],
};

#[allow(dead_code)]
const STRESS: Payload = Payload {
    name: "stress",
    kind: PayloadKind::Binary("stress-ng"),
    output: OutputFormat::ExitCode,
    default_args: &[],
    default_checks: &[],
    metrics: &[],
};

#[ktstr_test(payload = FIO, payload = STRESS)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
