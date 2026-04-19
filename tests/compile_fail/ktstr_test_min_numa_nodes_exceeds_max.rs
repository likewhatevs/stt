use ktstr::ktstr_test;

#[ktstr_test(min_numa_nodes = 4, max_numa_nodes = 2)]
fn bad(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
