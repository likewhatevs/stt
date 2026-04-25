use ktstr::ktstr_test;

#[ktstr_test(cleanup_budget_ms = 0)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
