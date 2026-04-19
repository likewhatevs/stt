use ktstr::ktstr_test;

#[ktstr_test(llcs = 3, numa_nodes = 2)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
