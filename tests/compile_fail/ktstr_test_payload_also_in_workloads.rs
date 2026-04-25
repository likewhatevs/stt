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
    None,
);

#[ktstr_test(payload = FIO, workloads = [FIO])]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
