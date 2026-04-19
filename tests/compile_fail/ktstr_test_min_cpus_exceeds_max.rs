use ktstr::ktstr_test;

#[ktstr_test(min_cpus = 64, max_cpus = 32)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
