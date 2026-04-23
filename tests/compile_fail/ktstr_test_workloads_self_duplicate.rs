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
    include_files: &[],
    uses_parent_pgrp: false,
    known_flags: None,
};

#[ktstr_test(workloads = [FIO, FIO])]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
