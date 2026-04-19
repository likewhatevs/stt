use ktstr::ktstr_test;

#[ktstr_test(min_llcs = 8, max_llcs = 4)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
