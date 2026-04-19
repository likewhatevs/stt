use ktstr::ktstr_test;

#[ktstr_test(extra_sched_args = "not-an-array")]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
